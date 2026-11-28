#include "engine/framework/modules/speech_encoders/lfm3_audio_encoder_runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/transformers/causal_decoder.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/cache_slots.h"

#include <ggml-cpp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::modules {
namespace {
constexpr int64_t kDownsampleStride = 8;
using core::TensorShape;
using core::TensorValue;

// One topology per feature length; all topologies borrow the encoder's one allocator.
class AudioGraph {
public:
    AudioGraph(const Lfm3AudioEncoderConfig & config, const Lfm3AudioEncoderWeights & weights,
               core::ExecutionContext & execution, int64_t frames)
        : execution_(execution), frames_(frames) {
        constexpr size_t nodes = 16384;
        context_.reset(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!context_) { throw std::runtime_error("LFM3 audio: failed to initialize the audio graph context"); }
        auto * g = context_.get();
        core::ModuleBuildContext ctx{g, "framework.lfm3.audio", execution.backend_type()};
        const int64_t c = config.conv_channels, d = config.decoder.hidden_size;
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 1, config.mel_bins, frames}));
        ggml_set_input(input_.tensor);

        // Official downsampler: SiLU after conv_in and after each depthwise/pointwise pair.
        auto x = Conv2dModule({1, c, 3, 3, 2, 2, 1, 1}).build(ctx, input_, weights.conv_in);
        x = SiluModule{}.build(ctx, x);
        x = DepthwiseConv2dModule({c, 3, 3, 2, 2, 1, 1}).build(ctx, x, weights.conv_mid);
        x = Conv2dModule({c, c, 1, 1}).build(ctx, x, weights.lin_mid);
        x = SiluModule{}.build(ctx, x);
        x = DepthwiseConv2dModule({c, 3, 3, 2, 2, 1, 1}).build(ctx, x, weights.conv_end);
        x = Conv2dModule({c, c, 1, 1}).build(ctx, x, weights.lin_end);
        x = SiluModule{}.build(ctx, x);
        tokens_ = x.shape.dims[3];
        if (tokens_ != (frames + kDownsampleStride - 1) / kDownsampleStride ||
            x.shape.dims[2] != (config.mel_bins / kDownsampleStride)) {
            throw std::runtime_error("LFM3 audio: audio downsampler output disagrees with its checkpoint configuration");
        }

        // PyTorch flatten(1,2): channel-major, then mel, for each time position.
        auto * flat = ggml_cont(g, ggml_permute(g, x.tensor, 2, 0, 1, 3));
        x = core::wrap_tensor(ggml_reshape_3d(g, flat, c * (config.mel_bins / kDownsampleStride), tokens_, 1),
                              TensorShape::from_dims({1, tokens_, c * (config.mel_bins / kDownsampleStride)}), GGML_TYPE_F32);
        x = LinearModule({c * (config.mel_bins / kDownsampleStride), d, true, config.decoder.projection_precision})
                .build(ctx, x, weights.lin_out);

        positions_ = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({tokens_}));
        mask_ = core::make_tensor(ctx, GGML_TYPE_F16, TensorShape::from_dims({tokens_, tokens_}));
        ggml_set_input(positions_.tensor);
        ggml_set_input(mask_.tensor);
        LfmDecoderInputs inputs;
        inputs.hidden = x;
        inputs.positions = positions_;
        // Encoder attention uses its configured causal window.
        inputs.full_mask = mask_;
        inputs.sliding_mask = mask_;
        x = build_lfm_decoder(ctx, inputs, weights.decoder, config.decoder).hidden;
        x = LayerNormModule({d, config.projector_norm_eps}).build(ctx, x, weights.projector_norm);
        x = LinearModule({d, config.output_size, true, config.decoder.projection_precision})
                .build(ctx, x, weights.projector_1);
        x = GeluModule({GeluApproximation::ExactErf}).build(ctx, x);
        x = LinearModule({config.output_size, config.output_size, true, config.decoder.projection_precision})
                .build(ctx, x, weights.projector_2);
        // An owning output avoids allocator reuse of the source behind a final reshape/view.
        output_ = core::wrap_tensor(ggml_cont(g, x.tensor), x.shape, x.type);
        ggml_set_output(output_.tensor);
        graph_ = ggml_new_graph_custom(g, nodes, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        core::validate_backend_graph_with_cpu_extra_buffers(execution.backend(), graph_, "LFM3 audio encoder");

        position_values_ = decoder_position_ids(tokens_);
        mask_values_ = causal_prefill_mask_values(1, tokens_);
        for (int64_t q = config.sliding_window; q < tokens_; ++q) {
            std::fill_n(mask_values_.begin() + q * tokens_, q - config.sliding_window + 1, ggml_fp32_to_fp16(-INFINITY));
        }
    }

    ~AudioGraph() { core::release_backend_graph_resources(execution_.backend(), graph_, true); }
    ggml_cgraph * graph() const noexcept { return graph_; }
    AudioGraph(const AudioGraph &) = delete;
    AudioGraph & operator=(const AudioGraph &) = delete;

    // Detach graph-local allocations before another topology can resize shared scratch.
    // Weight storage belongs to the session's separate context and is never detached.
    void unbind() {
        core::release_backend_graph_resources(execution_.backend(), graph_, true);
        for (auto * tensor = ggml_get_first_tensor(context_.get()); tensor; tensor = ggml_get_next_tensor(context_.get(), tensor)) {
            tensor->data = nullptr;
            tensor->buffer = nullptr;
            tensor->extra = nullptr;
        }
    }

    Lfm3AudioEncoding run(const std::vector<float> & features) {
        // Every run writes all inputs: another cached topology may have used the scratch.
        core::write_tensor_f32(input_, features);
        core::write_tensor_i32(positions_, position_values_);
        ggml_backend_tensor_set(mask_.tensor, mask_values_.data(), 0, mask_values_.size() * sizeof(ggml_fp16_t));
        core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const auto status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        if (status != GGML_STATUS_SUCCESS) { throw std::runtime_error("LFM3 audio: audio encoder graph compute failed"); }
        return {output_, frames_, tokens_};
    }

private:
    core::ExecutionContext & execution_;
    int64_t frames_, tokens_ = 0;
    ggml_context_ptr context_;
    ggml_cgraph * graph_ = nullptr;
    TensorValue input_, positions_, mask_, output_;
    std::vector<int32_t> position_values_;
    std::vector<ggml_fp16_t> mask_values_;
};

}  // namespace

struct Lfm3AudioEncoderRuntime::Impl {
    Impl(Lfm3AudioEncoderConfig config_in, Lfm3AudioEncoderWeights weights_in, core::ExecutionContext & execution_in, size_t slots)
        : config(std::move(config_in)), weights(std::move(weights_in)), execution(execution_in),
          allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()))), graphs(slots) {
        if (config.mel_bins <= 0 || config.conv_channels <= 0 || config.output_size <= 0 ||
            config.mel_bins % kDownsampleStride != 0 || config.sliding_window <= 0 ||
            !std::isfinite(config.projector_norm_eps) || config.projector_norm_eps <= 0) {
            throw std::invalid_argument("LFM3 audio encoder requires a stride-8 downsampler and positive dimensions, window and norm epsilon");
        }
        if (!slots) { throw std::invalid_argument("LFM3 audio graph cache requires at least one slot"); }
        if (!allocator) { throw std::runtime_error("LFM3 audio: failed to initialize the audio graph allocator"); }
    }

    Lfm3AudioEncoderConfig config;
    Lfm3AudioEncoderWeights weights;
    core::ExecutionContext & execution;
    // Cached contexts/graphs are destroyed before the allocator they borrow.
    ggml_gallocr_ptr allocator;
    runtime::CacheSlots<int64_t, std::shared_ptr<AudioGraph>> graphs;
    std::shared_ptr<AudioGraph> bound;
    size_t built = 0;

    void release_workspace() {
        if (bound) { bound->unbind(); bound.reset(); }
        allocator.reset();
    }
};

Lfm3AudioEncoderRuntime::Lfm3AudioEncoderRuntime(Lfm3AudioEncoderConfig config, Lfm3AudioEncoderWeights weights,
                                                   core::ExecutionContext & execution, size_t graph_cache_slots)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(weights), execution, graph_cache_slots)) {}

Lfm3AudioEncoderRuntime::~Lfm3AudioEncoderRuntime() = default;

Lfm3AudioEncoding Lfm3AudioEncoderRuntime::encode(const std::vector<float> & features, int64_t frames) {
    auto & m = *impl_;
    if (frames <= 0 || features.size() != static_cast<size_t>(m.config.mel_bins) * static_cast<size_t>(frames)) {
        throw std::invalid_argument("LFM3 audio features must be [mel_bins, frames]");
    }
    const auto encoder_start = std::chrono::steady_clock::now();
    std::shared_ptr<AudioGraph> graph;
    if (auto * cached = m.graphs.find(frames)) {
        graph = *cached;
    } else {
        graph = std::make_shared<AudioGraph>(m.config, m.weights, m.execution, frames);
        m.graphs.put(frames, graph);
        ++m.built;
    }
    if (m.bound != graph) {
        if (m.bound) { m.bound->unbind(); m.bound.reset(); }
        if (!m.allocator) { m.allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.execution.backend()))); }
        if (!m.allocator || !ggml_gallocr_reserve(m.allocator.get(), graph->graph()) ||
            !ggml_gallocr_alloc_graph(m.allocator.get(), graph->graph())) {
            graph->unbind();
            m.allocator.reset();
            throw runtime::CapacityError("LFM3 audio: audio encoder does not fit in device memory at " + std::to_string(frames) + " feature frames");
        }
        m.bound = graph;
    }
    auto out = graph->run(features);
    debug::timing_log_scalar("framework.lfm3.audio.encoder_ms", debug::elapsed_ms(encoder_start));
    debug::timing_log_scalar("framework.lfm3.audio.feature_frames", out.feature_frames);
    debug::timing_log_scalar("framework.lfm3.audio.tokens", out.tokens);
    return out;
}

void Lfm3AudioEncoderRuntime::release_workspace() { impl_->release_workspace(); }
size_t Lfm3AudioEncoderRuntime::graphs_built() const noexcept { return impl_->built; }

} // namespace engine::modules
