#include "engine/models/ced/runtime.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/attention/self_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/transformers/transformer_blocks.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace engine::models::ced {
namespace {

struct ContextDeleter {
    void operator()(ggml_context * context) const { ggml_free(context); }
};

struct CedViTWeights {
    std::unique_ptr<core::BackendWeightStore> store;
    modules::ChannelAffineWeights input_norm;
    modules::Conv2dWeights patch;
    core::TensorValue time_position;
    core::TensorValue frequency_position;
    std::vector<modules::TransformerEncoderBlockWeights> layers;
    modules::NormWeights norm;
    modules::NormWeights head_norm;
    modules::LinearWeights head;
};

CedViTWeights load_weights(const assets::TensorSource & source, const CedConfig & c,
                          core::ExecutionContext & execution, assets::TensorStorageType storage) {
    CedViTWeights w;
    w.store = std::make_unique<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "ced.weights", 4ULL * 1024 * 1024);
    auto & store = *w.store;
    w.input_norm = modules::binding::batch_norm_eval_from_source(store, source, "encoder.init_bn", c.mel_bins, 1e-5f);
    w.patch = modules::binding::conv2d_from_source(store, source, "encoder.patch_embed.proj", storage,
                                                 c.hidden, 1, c.patch_size, c.patch_size, true);
    w.time_position = store.load_f32_tensor(source, "encoder.time_pos_embed", {1, c.hidden, 1, c.target_frames / c.patch_stride});
    w.frequency_position = store.load_f32_tensor(source, "encoder.freq_pos_embed", {1, c.hidden, c.mel_bins / c.patch_stride, 1});
    for (int64_t i = 0; i < c.layers; ++i) {
        const auto prefix = "encoder.blocks." + std::to_string(i);
        modules::TransformerEncoderBlockWeights layer;
        layer.norm1 = modules::binding::norm_from_source(store, source, prefix + ".norm1", c.hidden);
        layer.norm2 = modules::binding::norm_from_source(store, source, prefix + ".norm2", c.hidden);
        layer.self_attention.qkv_weight = store.load_tensor(source, prefix + ".attn.qkv.weight", storage, {3 * c.hidden, c.hidden});
        layer.self_attention.qkv_bias = store.load_f32_tensor(source, prefix + ".attn.qkv.bias", {3 * c.hidden});
        const auto out = modules::binding::linear_from_source(store, source, prefix + ".attn.proj", storage, c.hidden, c.hidden, true);
        layer.self_attention.out_weight = out.weight;
        layer.self_attention.out_bias = out.bias;
        const auto fc1 = modules::binding::linear_from_source(store, source, prefix + ".mlp.fc1", storage, c.intermediate, c.hidden, true);
        const auto fc2 = modules::binding::linear_from_source(store, source, prefix + ".mlp.fc2", storage, c.hidden, c.intermediate, true);
        layer.feed_forward = {fc1.weight, fc1.bias, fc2.weight, fc2.bias};
        w.layers.push_back(std::move(layer));
    }
    w.norm = modules::binding::norm_from_source(store, source, "encoder.norm", c.hidden);
    w.head_norm = modules::binding::norm_from_source(store, source, "outputlayer.0", c.hidden);
    w.head = modules::binding::linear_from_source(store, source, "outputlayer.1", storage,
                                                static_cast<int64_t>(c.labels.size()), c.hidden, true);
    store.upload();
    return w;
}

class CedViTGraph {
public:
    CedViTGraph(const CedConfig & c, const CedViTWeights & w, core::ExecutionContext & execution,
                int64_t frames, bool flash)
        : backend_(execution.backend()), frames_(frames) {
        context_.reset(ggml_init({4ULL * 1024 * 1024, nullptr, true}));
        if (!context_) throw std::runtime_error("failed to create CED graph context");
        core::ModuleBuildContext ctx{context_.get(), "ced", execution.backend_type()};
        core::TensorValue output;
        if (frames == 0) {
            auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, c.hidden}));
            input_ = input.tensor;
            output = modules::LayerNormModule({c.hidden, 1e-5f, true, true, false}).build(ctx, input, w.head_norm);
            output = modules::LinearModule({c.hidden, static_cast<int64_t>(c.labels.size()), true}).build(ctx, output, w.head);
        } else {
            auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, c.mel_bins, 1, frames}));
            input_ = input.tensor;
            auto hidden = modules::BatchNorm2dEvalModule({c.mel_bins}).build(ctx, input, {w.input_norm.scale, w.input_norm.bias});
            auto mask = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, 1, frames}));
            mask_ = mask.tensor;
            mask = modules::RepeatModule({hidden.shape}).build(ctx, mask);
            hidden = modules::MulModule().build(ctx, hidden, mask);
            hidden = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, hidden);
            hidden = modules::Conv2dModule({1, c.hidden, c.patch_size, c.patch_size,
                static_cast<int>(c.patch_stride), static_cast<int>(c.patch_stride), 0, 0, 1, 1, true}).build(ctx, hidden, w.patch);
            const int64_t frequency = (c.mel_bins - c.patch_size) / c.patch_stride + 1;
            const int64_t time = (frames - c.patch_size) / c.patch_stride + 1;
            auto positions = modules::SliceModule({3, 0, time}).build(ctx, w.time_position);
            positions = modules::RepeatModule({hidden.shape}).build(ctx, positions);
            hidden = modules::AddModule().build(ctx, hidden, positions);
            auto frequency_positions = modules::RepeatModule({hidden.shape}).build(ctx, w.frequency_position);
            hidden = modules::AddModule().build(ctx, hidden, frequency_positions);
            hidden = modules::ReshapeModule({core::TensorShape::from_dims({1, c.hidden, frequency * time})}).build(ctx, hidden);
            hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, hidden);
            modules::AttentionConfig attention;
            attention.hidden_size = c.hidden;
            attention.num_heads = c.heads;
            attention.use_packed_qkv = true;
            attention.use_flash_attention = flash;
            const modules::LayerNormModule norm({c.hidden, 1e-6f, true, true, false});
            for (const auto & layer : w.layers) {
                auto normalized = norm.build(ctx, hidden, layer.norm1);
                auto attended = modules::SelfAttentionModule(attention).build(ctx, normalized, layer.self_attention);
                hidden = modules::ResidualAddModule().build(ctx, hidden, attended);
                normalized = norm.build(ctx, hidden, layer.norm2);
                auto fed = modules::FeedForwardModule({c.hidden, c.intermediate, true, modules::GeluApproximation::ExactErf})
                    .build(ctx, normalized, layer.feed_forward);
                hidden = modules::ResidualAddModule().build(ctx, hidden, fed);
            }
            hidden = norm.build(ctx, hidden, w.norm);
            output = modules::ReduceMeanModule({1}).build(ctx, hidden);
        }
        output_ = output.tensor;
        ggml_set_input(input_);
        if (mask_) ggml_set_input(mask_);
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(context_.get(), 8192, false);
        ggml_build_forward_expand(graph_, output_);
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (!allocator_ || !ggml_gallocr_reserve(allocator_.get(), graph_) || !ggml_gallocr_alloc_graph(allocator_.get(), graph_)) {
            throw std::runtime_error("failed to allocate CED graph");
        }
        plan_ = core::create_backend_graph_plan_if_host(backend_, graph_);
    }

    ~CedViTGraph() {
        core::release_backend_graph_resources(backend_, graph_);
        if (plan_) core::free_backend_graph_plan(backend_, plan_);
    }

    int64_t frames() const { return frames_; }

    std::vector<float> run(const std::vector<float> & input, int64_t valid_frames = 0) {
        if (input.size() * sizeof(float) != ggml_nbytes(input_)) throw std::runtime_error("CED graph input shape mismatch");
        ggml_backend_tensor_set(input_, input.data(), 0, input.size() * sizeof(float));
        if (mask_) {
            std::vector<float> mask(static_cast<size_t>(frames_), 0.0f);
            std::fill_n(mask.begin(), valid_frames, 1.0f);
            ggml_backend_tensor_set(mask_, mask.data(), 0, mask.size() * sizeof(float));
        }
        if (core::compute_backend_graph(backend_, graph_, plan_, "CED") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("CED graph execution failed");
        }
        std::vector<float> output(static_cast<size_t>(ggml_nelements(output_)));
        ggml_backend_tensor_get(output_, output.data(), 0, output.size() * sizeof(float));
        return output;
    }

private:
    ggml_backend_t backend_;
    int64_t frames_;
    std::unique_ptr<ggml_context, ContextDeleter> context_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator_{nullptr, ggml_gallocr_free};
    ggml_backend_graph_plan_t plan_ = nullptr;
};

}  // namespace

struct CedViTRuntime::Impl {
    Impl(std::shared_ptr<const assets::TensorSource> source, CedConfig c,
         core::ExecutionContext & e, assets::TensorStorageType storage, bool use_flash)
        : config(std::move(c)), execution(e), flash(use_flash), weights(load_weights(*source, config, e, storage)) {
        head = std::make_unique<CedViTGraph>(config, weights, execution, 0, false);
        source->release_storage();
    }

    CedConfig config;
    core::ExecutionContext & execution;
    bool flash;
    CedViTWeights weights;
    std::unique_ptr<CedViTGraph> full_encoder;
    std::unique_ptr<CedViTGraph> short_encoder;
    std::unique_ptr<CedViTGraph> head;
};

CedViTRuntime::CedViTRuntime(std::shared_ptr<const assets::TensorSource> source, CedConfig config,
                             core::ExecutionContext & execution, assets::TensorStorageType storage, bool flash)
    : impl_(std::make_unique<Impl>(std::move(source), std::move(config), execution, storage, flash)) {}
CedViTRuntime::~CedViTRuntime() = default;

std::vector<float> CedViTRuntime::classify(const std::vector<float> & mel, int64_t frames) {
    const auto started = std::chrono::steady_clock::now();
    auto & state = *impl_;
    const auto & c = state.config;
    if (frames < c.patch_size || mel.size() != static_cast<size_t>(c.mel_bins * frames)) {
        throw std::runtime_error("CED requires at least one spectrogram patch");
    }
    core::set_backend_threads(state.execution.backend(), std::max(1, state.execution.config().threads));
    const bool split = frames > c.target_frames;
    const int64_t chunks = split ? (frames + (c.pad_last ? c.target_frames - 1 : 0)) / c.target_frames : 1;
    const int64_t graph_frames = split ? c.target_frames : frames;
    auto & graph = split ? state.full_encoder : state.short_encoder;
    if (!graph || graph->frames() != graph_frames) {
        graph.reset();
        graph = std::make_unique<CedViTGraph>(c, state.weights, state.execution, graph_frames, state.flash);
    }
    std::vector<float> pooled(static_cast<size_t>(c.hidden), 0.0f);
    std::vector<float> input(static_cast<size_t>(c.mel_bins * graph_frames));
    for (int64_t chunk = 0; chunk < chunks; ++chunk) {
        std::fill(input.begin(), input.end(), 0.0f);
        const int64_t offset = chunk * graph_frames;
        const int64_t valid = std::min(graph_frames, frames - offset);
        for (int64_t mel_bin = 0; mel_bin < c.mel_bins; ++mel_bin) {
            std::copy_n(mel.data() + mel_bin * frames + offset, valid, input.data() + mel_bin * graph_frames);
        }
        const auto encoded = graph->run(input, valid);
        for (size_t i = 0; i < pooled.size(); ++i) pooled[i] += encoded[i] / static_cast<float>(chunks);
    }
    auto logits = state.head->run(pooled);
    debug::timing_log_scalar("ced.graph_ms", debug::elapsed_ms(started));
    return logits;
}

}  // namespace engine::models::ced
