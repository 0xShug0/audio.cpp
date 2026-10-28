#include "engine/community_models/lfm3_audio/encoder.h"

#include "../lfm2_audio/weight_stores.h"

#include "engine/community_models/lfm3_audio/debug.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/runtime/errors.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>

#include <chrono>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::lfm3_audio {
namespace {

using core::TensorShape;
using core::TensorValue;

constexpr size_t kWeightContextBytes = 4ull * 1024ull * 1024ull;
constexpr size_t kGraphNodes = 8192;
constexpr float kLayerNormEps = 1e-5f;  // nn.LayerNorm default
// 30 s of features. A longer pass (ASR runs up to 300 s in one) frees its compute buffer afterwards.
constexpr int64_t kMaxRetainedFrames = 3000;

struct EncoderWeights {
    std::unique_ptr<WeightStores> stores;
    modules::Conv2dWeights conv_in, conv_mid, lin_mid, conv_end, lin_end;
    modules::LinearWeights lin_out;
    blocks::StackWeights stack;
    modules::NormWeights projector_norm;
    modules::LinearWeights projector_1, projector_2;
};

EncoderWeights load_weights(const assets::TensorSource & source, const Lfm3EncoderConfig & config, core::ExecutionContext & execution,
                            const Lfm3WeightOptions & options) {
    EncoderWeights out;
    out.stores = std::make_unique<WeightStores>(execution, "lfm3_audio.encoder.weights", kWeightContextBytes, options.cpu_repack);
    auto & store = out.stores->plain();
    const auto backend = execution.backend();
    const int64_t c = config.conv_channels;
    const int64_t d = config.stack.hidden;
    const int64_t o = config.output_size;

    // Conv weights stay F32 whatever the file holds: Metal's depthwise conv is F32 only.
    const auto conv = [&](const std::string & name, int64_t in, int64_t k) {
        const std::string p = "encoder.downsample." + name;
        return modules::Conv2dWeights{store.load_f32_tensor(source, p + ".weight", {c, in, k, k}),
                                      store.load_f32_tensor(source, p + ".bias", {c})};
    };
    const auto linear = [&](const std::string & name, int64_t in, int64_t out_features) {
        return modules::LinearWeights{blocks::load_matmul(*out.stores, backend, source, name + ".weight", {out_features, in}, options),
                                      store.load_f32_tensor(source, name + ".bias", {out_features})};
    };

    out.conv_in = conv("conv_in", 1, 3);
    out.conv_mid = conv("conv_mid", 1, 3);
    out.lin_mid = conv("lin_mid", c, 1);
    out.conv_end = conv("conv_end", 1, 3);
    out.lin_end = conv("lin_end", c, 1);
    out.lin_out = linear("encoder.downsample.lin_out", c * config.mel_after_downsample, d);
    out.stack = blocks::load_stack(*out.stores, backend, source, "encoder.lfm.lfm.", config.stack, options);
    out.projector_norm = {store.load_f32_tensor(source, "mlp.model.0.weight", {d}), store.load_f32_tensor(source, "mlp.model.0.bias", {d})};
    out.projector_1 = linear("mlp.model.1", d, o);
    out.projector_2 = linear("mlp.model.3", o, o);
    out.stores->upload();
    return out;
}

// An output that is a view (a reshape after a linear) also needs its source kept: the allocator frees a view's
// source once the view's consumers ran, whatever the view's own flags.
void keep_output(ggml_tensor * t) {
    for (; t != nullptr; t = t->view_src) {
        ggml_set_output(t);
    }
}

// Conv2d with the bias broadcast over [W, H] instead of repeated to the output size (conv_in's output is the
// largest tensor of the graph).
TensorValue conv2d(core::ModuleBuildContext & ctx, const TensorValue & x, const modules::Conv2dWeights & w, int64_t in, int64_t out,
                   int64_t k, int stride, int padding) {
    auto y = modules::Conv2dModule({in, out, k, k, stride, stride, padding, padding, 1, 1, false}).build(ctx, x, {w.weight, std::nullopt});
    return core::wrap_tensor(ggml_add(ctx.ggml, y.tensor, ggml_reshape_4d(ctx.ggml, w.bias->tensor, 1, 1, out, 1)), y.shape);
}

TensorValue depthwise(core::ModuleBuildContext & ctx, const TensorValue & x, const modules::Conv2dWeights & w, int64_t channels) {
    return modules::DepthwiseConv2dModule({channels, 3, 3, 2, 2, 1, 1}).build(ctx, x, w);
}

// The encoder graph for one feature length; its tensors live in the runtime's compute buffer.
class EncoderGraph {
public:
    EncoderGraph(const EncoderWeights & weights, const Lfm3EncoderConfig & config, core::ExecutionContext & execution,
                 ggml_gallocr_t allocator, const Lfm3WeightOptions & options, int64_t frames)
        : config_(config), execution_(execution), frames_(frames), ctx_(blocks::graph_context(kGraphNodes, "encoder")) {
        auto * g = ctx_.get();
        graph_ = ggml_new_graph_custom(g, kGraphNodes, false);
        core::ModuleBuildContext ctx{g, "lfm3_audio.encoder", execution.backend_type()};
        const int64_t c = config.conv_channels;
        const int64_t d = config.stack.hidden;

        // Lfm3AudioDownsampler: the features as a 1-channel image [mel (H), time (W)], three stride-2 convs.
        // Its length masks only cut batch padding, so a single clip needs none.
        auto input = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 1, config.n_mels, frames}));
        ggml_set_input(input.tensor);
        input_ = input.tensor;
        const modules::SiluModule silu{};
        auto x = silu.build(ctx, conv2d(ctx, input, weights.conv_in, 1, c, 3, 2, 1));
        x = silu.build(ctx, conv2d(ctx, depthwise(ctx, x, weights.conv_mid, c), weights.lin_mid, c, c, 1, 1, 0));
        x = silu.build(ctx, conv2d(ctx, depthwise(ctx, x, weights.conv_end, c), weights.lin_end, c, c, 1, 1, 0));
        tokens_ = x.shape.dims[3];
        if (x.shape.dims[2] != config.mel_after_downsample || tokens_ != lfm3_audio_tokens(frames)) {
            throw std::runtime_error("LFM3-Audio downsampler output does not match the config");
        }

        // flatten(1, 2): feature index c * 16 + f, one row per token.
        auto * flat = ggml_cont(g, ggml_permute(g, x.tensor, 2, 0, 1, 3));
        auto rows = core::wrap_tensor(ggml_reshape_3d(g, flat, c * config.mel_after_downsample, tokens_, 1),
                                      TensorShape::from_dims({1, tokens_, c * config.mel_after_downsample}));
        auto down = modules::LinearModule({c * config.mel_after_downsample, d, true, blocks::kMatmulPrecision}).build(ctx, rows, weights.lin_out);
        down = core::ensure_backend_addressable_layout(ctx, down);
        keep_output(down.tensor);
        downsample_ = down.tensor;

        // The positions and the mask depend on the length only, so they are set once. As outputs, the allocator
        // keeps their memory for the next run.
        auto positions = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({tokens_}));
        auto mask = core::make_tensor(ctx, GGML_TYPE_F16, TensorShape::from_dims({tokens_, tokens_}));
        for (auto * t : {positions.tensor, mask.tensor}) {
            ggml_set_input(t);
            ggml_set_output(t);
        }

        blocks::StepInputs in;
        in.x = down;
        in.positions = positions;
        in.sliding_mask = mask;
        auto encoded = blocks::build_stack(ctx, graph_, config.stack, weights.stack, in, nullptr, options);
        encoded = core::ensure_backend_addressable_layout(ctx, encoded);
        keep_output(encoded.tensor);
        encoder_ = encoded.tensor;

        // Lfm3AudioProjector.
        const int64_t o = config.output_size;
        auto y = modules::LayerNormModule({d, kLayerNormEps}).build(ctx, encoded, weights.projector_norm);
        y = modules::LinearModule({d, o, true, blocks::kMatmulPrecision}).build(ctx, y, weights.projector_1);
        y = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, y);
        y = modules::LinearModule({o, o, true, blocks::kMatmulPrecision}).build(ctx, y, weights.projector_2);
        y = core::ensure_backend_addressable_layout(ctx, y);
        keep_output(y.tensor);
        projector_ = y.tensor;

        ggml_build_forward_expand(graph_, projector_);
        core::validate_backend_graph_with_cpu_extra_buffers(execution.backend(), graph_, "LFM3-Audio encoder graph");
        if (!ggml_gallocr_alloc_graph(allocator, graph_)) {
            throw runtime::CapacityError("LFM3-Audio encoder graph does not fit in device memory for " + std::to_string(frames) +
                                         " feature frames");
        }

        std::vector<int32_t> pos(static_cast<size_t>(tokens_));
        std::iota(pos.begin(), pos.end(), 0);
        const auto window = blocks::window_mask(pos, tokens_, config.stack.sliding_window);
        ggml_backend_tensor_set(positions.tensor, pos.data(), 0, ggml_nbytes(positions.tensor));
        ggml_backend_tensor_set(mask.tensor, window.data(), 0, ggml_nbytes(mask.tensor));
    }

    ~EncoderGraph() { core::release_backend_graph_resources(execution_.backend(), graph_, true); }

    EncoderGraph(const EncoderGraph &) = delete;
    EncoderGraph & operator=(const EncoderGraph &) = delete;

    [[nodiscard]] int64_t frames() const { return frames_; }

    Lfm3AudioEmbeddings run(const Lfm3Features & features, Lfm3EncoderTaps * taps) {
        // Feature-major [mel][time] is the input's ggml layout (ne0 = time).
        ggml_backend_tensor_set(input_, features.values.data(), 0, ggml_nbytes(input_));
        blocks::compute(execution_, graph_, "encoder");

        Lfm3AudioEmbeddings out;
        out.tokens = tokens_;
        out.hidden = config_.output_size;
        out.values = core::read_tensor_f32(projector_);
        if (taps != nullptr) {
            taps->downsample = core::read_tensor_f32(downsample_);
            taps->encoder = core::read_tensor_f32(encoder_);
        }
        return out;
    }

private:
    const Lfm3EncoderConfig & config_;
    core::ExecutionContext & execution_;
    int64_t frames_ = 0;
    int64_t tokens_ = 0;
    ggml_context_ptr ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * downsample_ = nullptr;
    ggml_tensor * encoder_ = nullptr;
    ggml_tensor * projector_ = nullptr;
};

}  // namespace

struct Lfm3AudioEncoderRuntime::Impl {
    Impl(std::shared_ptr<const assets::TensorSource> source_in, const Lfm3EncoderConfig & config_in, core::ExecutionContext & execution_in,
         const Lfm3WeightOptions & options_in)
        : source(std::move(source_in)),
          config(config_in),
          execution(execution_in),
          options(options_in),
          weights(load_weights(*source, config_in, execution_in, options_in)) {}

    Lfm3AudioEmbeddings encode(const Lfm3Features & features, Lfm3EncoderTaps * taps) {
        const auto start = std::chrono::steady_clock::now();
        Lfm3AudioEmbeddings out;
        try {
            if (graph == nullptr || graph->frames() != features.frames) {
                graph.reset();
                if (allocator == nullptr) {
                    allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
                    if (allocator == nullptr) {
                        throw std::runtime_error("failed to create the LFM3-Audio encoder graph allocator");
                    }
                }

                graph = std::make_unique<EncoderGraph>(weights, config, execution, allocator.get(), options, features.frames);
                ++graphs_built;
            }

            out = graph->run(features, taps);
        } catch (...) {
            // A failed allocation keeps the failed graph's plan, so the next call starts afresh.
            reset();
            throw;
        }

        if (features.frames > kMaxRetainedFrames) {
            reset();
        }

        debug::timing_log_scalar("lfm3_audio.encoder.ms", engine::debug::elapsed_ms(start));
        return out;
    }

    void reset() {
        graph.reset();
        allocator.reset();
    }

    std::shared_ptr<const assets::TensorSource> source;
    Lfm3EncoderConfig config;
    core::ExecutionContext & execution;
    Lfm3WeightOptions options;
    EncoderWeights weights;
    int64_t graphs_built = 0;
    // The graph, declared after the allocator, is destroyed first.
    ggml_gallocr_ptr allocator;
    std::unique_ptr<EncoderGraph> graph;
};

Lfm3AudioEncoderRuntime::Lfm3AudioEncoderRuntime(std::shared_ptr<const assets::TensorSource> model, const Lfm3EncoderConfig & config,
                                                 core::ExecutionContext & execution, const Lfm3WeightOptions & options)
    : impl_(std::make_unique<Impl>(std::move(model), config, execution, options)) {}

Lfm3AudioEncoderRuntime::~Lfm3AudioEncoderRuntime() = default;

Lfm3AudioEmbeddings Lfm3AudioEncoderRuntime::encode(const Lfm3Features & features, Lfm3EncoderTaps * taps) {
    const auto & config = impl_->config;
    if (features.frames <= 0 || static_cast<int64_t>(features.values.size()) != config.n_mels * features.frames) {
        throw std::runtime_error("LFM3-Audio encoder features must be [" + std::to_string(config.n_mels) + "][frames]");
    }

    const bool dump = lfm3_dump_enabled();
    Lfm3EncoderTaps dump_taps;
    auto out = impl_->encode(features, taps != nullptr ? taps : dump ? &dump_taps : nullptr);
    if (dump) {
        const auto & t = taps != nullptr ? *taps : dump_taps;
        lfm3_dump("downsample", t.downsample.data(), {out.tokens, config.stack.hidden});
        lfm3_dump("encoder", t.encoder.data(), {out.tokens, config.stack.hidden});
        lfm3_dump("projector", out.values.data(), {out.tokens, out.hidden});
    }
    return out;
}

int64_t Lfm3AudioEncoderRuntime::graphs_built() const noexcept {
    return impl_->graphs_built;
}

}  // namespace engine::community_models::lfm3_audio
