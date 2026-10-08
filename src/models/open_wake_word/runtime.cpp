#include "engine/models/open_wake_word/runtime.h"
#ifdef ENGINE_HAVE_NEON
#include "engine/framework/neon/execution.h"
#include "engine/models/open_wake_word/neon_embedding.h"
#endif
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/models/open_wake_word/frontend.h"
#include "engine/models/open_wake_word/graph_weights.h"

#include <algorithm>
#include <cmath>
#include <ggml-alloc.h>
#include <limits>
#include <stdexcept>

namespace engine::models::open_wake_word {
namespace {
#ifdef ENGINE_HAVE_NEON
namespace cpu = engine::neon;
#endif
class Graph {
public:
    explicit Graph(const core::ExecutionContext &execution)
        : execution_(execution), context_(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free),
          allocator_(nullptr, ggml_gallocr_free) {
        if (!context_) {
            throw std::runtime_error("OpenWakeWord graph context allocation failed");
        }
    }
    ~Graph() {
        if (graph_) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
        }
    }
    core::ModuleBuildContext context() {
        return {context_.get(), "open_wake_word", execution_.backend_type()};
    }
    void prepare() {
        ggml_set_output(output.tensor);
        graph_ = ggml_new_graph_custom(context_.get(), 4096, false);
        ggml_build_forward_expand(graph_, output.tensor);
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (!ggml_gallocr_alloc_graph(allocator_.get(), graph_)) {
            throw std::runtime_error("OpenWakeWord graph allocation failed");
        }
        core::prepare_host_graph_plan(execution_, graph_, plan_);
    }
    std::vector<float> run(const std::vector<float> &values) {
        core::write_tensor_f32(input, values);
        if (core::compute_graph(execution_, graph_, plan_, "open_wake_word") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("OpenWakeWord graph failed");
        }
        ggml_backend_synchronize(execution_.backend());
        return core::read_tensor_f32(output.tensor);
    }
    core::TensorValue input, output;

private:
    const core::ExecutionContext &execution_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator_;
    core::HostGraphPlan plan_;
    ggml_cgraph *graph_ = nullptr;
};

void build_graph_embedding(Graph &graph, const Weights &weights, const GraphWeights &bound) {
    auto ctx = graph.context();
    graph.input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, 76, 32}));
    ggml_set_input(graph.input.tensor);
    auto value = graph.input;
    for (size_t i = 0; i < weights.embedding.size(); ++i) {
        const auto &layer = weights.embedding[i];
        if (const auto *conv = std::get_if<ConvLayer>(&layer)) {
            modules::Conv2dConfig config;
            const auto &c = conv->config;
            config.in_channels = c.input_channels;
            config.out_channels = c.output_channels;
            config.kernel_height = c.kernel_height;
            config.kernel_width = c.kernel_width;
            config.stride_height = c.stride_height;
            config.stride_width = c.stride_width;
            config.dilation_height = c.dilation_height;
            config.dilation_width = c.dilation_width;
            config.padding_height = c.padding_height;
            config.padding_width = c.padding_width;
            config.use_bias = !conv->bias.empty();
            value = modules::Conv2dModule(config).build(ctx, value, *bound.embedding[i]);
        } else if (const auto *activation = std::get_if<LeakyReluLayer>(&layer)) {
            value = modules::LeakyReluModule({activation->slope}).build(ctx, value);
        } else if (const auto *clamp = std::get_if<ClampLayer>(&layer)) {
            value = core::wrap_tensor(
                ggml_clamp(ctx.ggml, value.tensor, clamp->minimum, std::numeric_limits<float>::infinity()), value.shape,
                GGML_TYPE_F32);
        } else {
            const auto &pool = std::get<PoolLayer>(layer);
            auto *output = ggml_pool_2d(ctx.ggml, value.tensor, GGML_OP_POOL_MAX, pool.width, pool.height,
                                        pool.stride_width, pool.stride_height, 0, 0);
            value = core::wrap_tensor(
                output, core::TensorShape::from_dims({1, value.shape.dims[1], output->ne[1], output->ne[0]}),
                GGML_TYPE_F32);
        }
    }
    if (ggml_nelements(value.tensor) != 96) {
        throw std::runtime_error("OpenWakeWord embedding must have 96 values");
    }
    graph.output = value;
    graph.prepare();
}

void build_graph_classifier(Graph &graph, const Weights &weights, const GraphWeights &bound) {
    auto ctx = graph.context();
    graph.input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, 1536}));
    ggml_set_input(graph.input.tensor);
    auto value = graph.input;
    for (size_t i = 0; i < weights.classifier.size(); ++i) {
        const auto &layer = weights.classifier[i];
        if (const auto *linear = std::get_if<LinearLayer>(&layer)) {
            value = modules::LinearModule({linear->input_features, linear->output_features, true})
                        .build(ctx, value, *bound.classifier[i]);
        } else if (std::holds_alternative<ReluLayer>(layer)) {
            value = modules::ReluModule().build(ctx, value);
        } else {
            value = modules::SigmoidModule().build(ctx, value);
        }
    }
    graph.output = value;
    graph.prepare();
}

} // namespace

struct Runtime::Impl {
    const core::ExecutionContext &execution;
    Frontend frontend;
    std::optional<GraphWeights> graph_weights;
    std::unique_ptr<Graph> encoder, classifier;
#ifdef ENGINE_HAVE_NEON
    std::unique_ptr<cpu::ExecutionPlan> native_embedding;
#endif
    int suppression_frames;
    std::vector<float> initial, features, mel_history, raw_history, pending;
    int64_t received = 0, processed = 0, frames = 0;

    Impl(const Weights &weights, const core::ExecutionContext &context, bool use_arm)
        : execution(context), frontend(weights.frontend), suppression_frames(weights.suppression_frames),
          initial(weights.initial_features) {
#ifndef ENGINE_HAVE_NEON
        if (use_arm) {
            throw std::runtime_error("OpenWakeWord native embedding requires ENGINE_ENABLE_NEON=ON");
        }
#endif
        graph_weights = bind_graph_weights(weights, execution, !use_arm);
        classifier = std::make_unique<Graph>(execution);
        build_graph_classifier(*classifier, weights, *graph_weights);
#ifdef ENGINE_HAVE_NEON
        if (use_arm) {
            native_embedding = build_native_embedding(weights);
        } else
#endif
        {
            encoder = std::make_unique<Graph>(execution);
            build_graph_embedding(*encoder, weights, *graph_weights);
        }
        reset();
    }

    void reset() {
        features = initial;
        mel_history.assign(76 * 32, 1.0F);
        raw_history.clear();
        pending.clear();
        received = processed = frames = 0;
    }

    std::vector<float> encode(const std::vector<float> &mel) {
        if (mel.size() != 76 * 32) {
            throw std::invalid_argument("OpenWakeWord embedding requires [76,32] input");
        }
        const auto started = std::chrono::steady_clock::now();
        std::vector<float> result;
#ifdef ENGINE_HAVE_NEON
        if (native_embedding) {
            result = native_embedding->run({cpu::TensorView(mel, {1, 76, 32})}, execution.config().threads).values;
        } else
#endif
        {
            result = encoder->run(mel);
        }
        debug::timing_log_scalar("open_wake_word.embedding_ms", debug::elapsed_ms(started));
        return result;
    }

    Score step(const float *pcm) {
        const auto started = std::chrono::steady_clock::now();
        std::vector<float> waveform = raw_history;
        waveform.insert(waveform.end(), pcm, pcm + 1280);
        raw_history.assign(waveform.end() - 480, waveform.end());
        const auto mel = frontend.extract(waveform, execution.config().threads);
        mel_history.insert(mel_history.end(), mel.values.begin(), mel.values.end());
        mel_history.erase(mel_history.begin(), mel_history.end() - 76 * 32);
        debug::timing_log_scalar("open_wake_word.frontend_ms", debug::elapsed_ms(started));
        const auto embedding = encode(mel_history);
        features.erase(features.begin(), features.begin() + 96);
        features.insert(features.end(), embedding.begin(), embedding.end());
        const auto classify_start = std::chrono::steady_clock::now();
        float probability = classifier->run(features).at(0);
        debug::timing_log_scalar("open_wake_word.classifier_ms", debug::elapsed_ms(classify_start));
        if (!std::isfinite(probability)) {
            throw std::runtime_error("OpenWakeWord produced a non-finite probability");
        }
        if (frames++ < suppression_frames) {
            probability = 0;
        }
        processed += 1280;
        return {std::min(processed, received), probability};
    }
};

Runtime::Runtime(const Weights &weights, const core::ExecutionContext &execution, bool arm_embedding)
    : impl_(std::make_unique<Impl>(weights, execution, arm_embedding)) {}
Runtime::~Runtime() = default;
void Runtime::reset() {
    impl_->reset();
}
std::vector<float> Runtime::encode(const std::vector<float> &mel) {
    return impl_->encode(mel);
}

std::vector<Score> Runtime::push(const std::vector<float> &samples) {
    auto &p = *impl_;
    if (!std::all_of(samples.begin(), samples.end(), [](float value) { return std::isfinite(value); })) {
        throw std::invalid_argument("OpenWakeWord requires finite audio");
    }
    for (const float value : samples) {
        p.pending.push_back(static_cast<int16_t>(std::clamp(value * 32768.0F, -32768.0F, 32767.0F)));
    }
    p.received += samples.size();
    std::vector<Score> scores;
    size_t offset = 0;
    while (p.pending.size() - offset >= 1280) {
        scores.push_back(p.step(p.pending.data() + offset));
        offset += 1280;
    }
    p.pending.erase(p.pending.begin(), p.pending.begin() + static_cast<std::ptrdiff_t>(offset));
    return scores;
}

std::vector<Score> Runtime::finish() {
    auto &p = *impl_;
    if (p.pending.empty()) {
        return {};
    }
    p.pending.resize(1280, 0.0F);
    auto score = p.step(p.pending.data());
    p.pending.clear();
    return {score};
}

} // namespace engine::models::open_wake_word
