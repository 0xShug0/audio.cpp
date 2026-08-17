#include "engine/framework/codecs/mimi_codec_runtime.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/transformer_blocks.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::codecs {
namespace {

constexpr float kCodebookEps = 1.0e-5F;
constexpr float kMaskedAttentionBias = -std::numeric_limits<float>::infinity();
constexpr int64_t kMimiActiveCodebooks = 8;
constexpr int64_t kMimiFrameSamples = 1920;
namespace binding = engine::modules::binding;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

modules::TransformerEncoderBlockWeights as_transformer_layer_weights(
    const MimiCodecTransformerLayerWeights & weights) {
    return {
        weights.norm1,
        weights.self_attn,
        weights.layer_scale1,
        weights.norm2,
        weights.feed_forward,
        weights.layer_scale2,
    };
}

std::vector<float> transformer_attention_mask(int64_t frames, int64_t cache_steps, int64_t context, int64_t current_end) {
    if (frames <= 0 || cache_steps < 0 || context <= 0 || current_end < 0) {
        throw std::runtime_error("Mimi codec transformer attention mask received invalid dimensions");
    }
    std::vector<float> values(static_cast<size_t>(frames * (cache_steps + frames)), kMaskedAttentionBias);
    for (int64_t query = 0; query < frames; ++query) {
        const int64_t query_position = current_end + query;
        for (int64_t slot = 0; slot < cache_steps; ++slot) {
            int64_t key_position = -1;
            if (current_end <= cache_steps) {
                if (slot < current_end) {
                    key_position = slot;
                }
            } else {
                const int64_t end_slot = current_end % cache_steps;
                const int64_t delta = slot - end_slot;
                key_position = delta < 0 ? current_end + delta : current_end + delta - cache_steps;
            }
            const int64_t distance = query_position - key_position;
            if (key_position >= 0 && distance >= 0 && distance < context) {
                values[static_cast<size_t>(query * (cache_steps + frames) + slot)] = 0.0F;
            }
        }
        for (int64_t local_key = 0; local_key <= query; ++local_key) {
            const int64_t key_position = current_end + local_key;
            if (query_position - key_position < context) {
                values[static_cast<size_t>(query * (cache_steps + frames) + cache_steps + local_key)] = 0.0F;
            }
        }
    }
    return values;
}

struct StreamingConv1dState {
    int64_t channels = 0;
    int64_t history_frames = 0;
    bool first = true;
    std::vector<float> previous;
};

struct StreamingConvTranspose1dState {
    int64_t channels = 0;
    int64_t partial_frames = 0;
    std::vector<float> partial;
};

struct MimiTransformerState {
    int64_t current_end = 0;
};

struct MimiDecoderState {
    StreamingConvTranspose1dState encoder_rate_upsample;
    MimiTransformerState transformer;
    StreamingConv1dState input_projection;
    std::vector<StreamingConvTranspose1dState> stage_upsamples;
    std::vector<std::array<StreamingConv1dState, 2>> stage_residual_convs;
    StreamingConv1dState output_projection;
};

struct MimiEncoderState {
    StreamingConv1dState input_projection;
    std::vector<std::array<StreamingConv1dState, 2>> stage_residual_convs;
    std::vector<StreamingConv1dState> stage_downsamples;
    StreamingConv1dState output_projection;
    MimiTransformerState transformer;
    StreamingConv1dState downsample;
};

void release_graph_runtime(
    ggml_gallocr_t & gallocr,
    ggml_backend_buffer_t & params_buffer,
    ggml_context *& context) {
    if (gallocr != nullptr) {
        ggml_gallocr_free(gallocr);
        gallocr = nullptr;
    }
    if (params_buffer != nullptr) {
        ggml_backend_buffer_free(params_buffer);
        params_buffer = nullptr;
    }
    if (context != nullptr) {
        ggml_free(context);
        context = nullptr;
    }
}

StreamingConv1dState make_streaming_conv1d_state(int64_t channels, int64_t kernel_size, int stride, int dilation) {
    StreamingConv1dState state;
    state.channels = channels;
    state.history_frames = std::max<int64_t>(0, (kernel_size - 1) * dilation + 1 - stride);
    state.previous.assign(static_cast<size_t>(state.channels * state.history_frames), 0.0F);
    return state;
}

StreamingConvTranspose1dState make_streaming_convtranspose_state(int64_t channels, int64_t kernel_size, int stride) {
    StreamingConvTranspose1dState state;
    state.channels = channels;
    state.partial_frames = std::max<int64_t>(0, kernel_size - stride);
    state.partial.assign(static_cast<size_t>(state.channels * state.partial_frames), 0.0F);
    return state;
}

MimiDecoderState make_mimi_decoder_state(const MimiCodecConfig & config) {
    MimiDecoderState state;
    state.encoder_rate_upsample = make_streaming_convtranspose_state(config.hidden_size, 4, 2);
    state.input_projection = make_streaming_conv1d_state(config.hidden_size, 7, 1, 1);
    state.stage_upsamples.push_back(make_streaming_convtranspose_state(512, 16, 8));
    state.stage_upsamples.push_back(make_streaming_convtranspose_state(256, 12, 6));
    state.stage_upsamples.push_back(make_streaming_convtranspose_state(128, 10, 5));
    state.stage_upsamples.push_back(make_streaming_convtranspose_state(64, 8, 4));
    state.stage_residual_convs = {
        std::array<StreamingConv1dState, 2>{
            make_streaming_conv1d_state(512, 3, 1, 1),
            make_streaming_conv1d_state(256, 1, 1, 1),
        },
        std::array<StreamingConv1dState, 2>{
            make_streaming_conv1d_state(256, 3, 1, 1),
            make_streaming_conv1d_state(128, 1, 1, 1),
        },
        std::array<StreamingConv1dState, 2>{
            make_streaming_conv1d_state(128, 3, 1, 1),
            make_streaming_conv1d_state(64, 1, 1, 1),
        },
        std::array<StreamingConv1dState, 2>{
            make_streaming_conv1d_state(64, 3, 1, 1),
            make_streaming_conv1d_state(32, 1, 1, 1),
        },
    };
    state.output_projection = make_streaming_conv1d_state(64, 3, 1, 1);
    return state;
}

MimiEncoderState make_mimi_encoder_state(const MimiCodecConfig & config) {
    MimiEncoderState state;
    state.input_projection = make_streaming_conv1d_state(config.channels, 7, 1, 1);
    state.stage_residual_convs = {
        std::array<StreamingConv1dState, 2>{
            make_streaming_conv1d_state(64, 3, 1, 1),
            make_streaming_conv1d_state(32, 1, 1, 1),
        },
        std::array<StreamingConv1dState, 2>{
            make_streaming_conv1d_state(128, 3, 1, 1),
            make_streaming_conv1d_state(64, 1, 1, 1),
        },
        std::array<StreamingConv1dState, 2>{
            make_streaming_conv1d_state(256, 3, 1, 1),
            make_streaming_conv1d_state(128, 1, 1, 1),
        },
        std::array<StreamingConv1dState, 2>{
            make_streaming_conv1d_state(512, 3, 1, 1),
            make_streaming_conv1d_state(256, 1, 1, 1),
        },
    };
    state.stage_downsamples.push_back(make_streaming_conv1d_state(64, 8, 4, 1));
    state.stage_downsamples.push_back(make_streaming_conv1d_state(128, 10, 5, 1));
    state.stage_downsamples.push_back(make_streaming_conv1d_state(256, 12, 6, 1));
    state.stage_downsamples.push_back(make_streaming_conv1d_state(512, 16, 8, 1));
    state.output_projection = make_streaming_conv1d_state(1024, 3, 1, 1);
    state.downsample = make_streaming_conv1d_state(config.hidden_size, 4, 2, 1);
    return state;
}

struct StatefulConv1dOutput {
    core::TensorValue output;
    std::optional<core::TensorValue> next_history;
};

StatefulConv1dOutput build_stateful_conv1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::optional<core::TensorValue> & history,
    const modules::StreamingConv1dWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int stride,
    int dilation,
    bool use_bias) {
    auto full_input = input;
    if (history.has_value()) {
        full_input = modules::ConcatModule({2}).build(ctx, *history, input);
    }
    auto output = modules::Conv1dModule({
        in_channels,
        out_channels,
        kernel_size,
        stride,
        0,
        dilation,
        use_bias,
    }).build(ctx, full_input, weights);
    std::optional<core::TensorValue> next_history = std::nullopt;
    if (history.has_value()) {
        next_history = modules::SliceModule({
            2,
            full_input.shape.dims[2] - history->shape.dims[2],
            history->shape.dims[2],
        }).build(ctx, full_input);
    }
    return {output, next_history};
}

struct StatefulConvTranspose1dOutput {
    core::TensorValue output;
    std::optional<core::TensorValue> next_partial;
};

StatefulConvTranspose1dOutput build_stateful_convtranspose1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::optional<core::TensorValue> & partial,
    const modules::ConvTranspose1dWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int stride,
    bool use_bias) {
    auto raw = modules::ConvTranspose1dModule({
        in_channels,
        out_channels,
        kernel_size,
        stride,
        0,
        1,
        use_bias,
    }).build(ctx, input, weights);
    if (!partial.has_value()) {
        return {raw, std::nullopt};
    }
    const int64_t partial_frames = partial->shape.dims[2];
    auto prefix = modules::SliceModule({2, 0, partial_frames}).build(ctx, raw);
    prefix = modules::AddModule().build(ctx, prefix, *partial);
    auto suffix = modules::SliceModule({2, partial_frames, raw.shape.dims[2] - partial_frames}).build(ctx, raw);
    auto combined = modules::ConcatModule({2}).build(ctx, prefix, suffix);
    auto output = modules::SliceModule({2, 0, raw.shape.dims[2] - partial_frames}).build(ctx, combined);
    auto next_partial = modules::SliceModule({
        2,
        raw.shape.dims[2] - partial_frames,
        partial_frames,
    }).build(ctx, combined);
    return {output, next_partial};
}

StatefulConvTranspose1dOutput build_stateful_depthwise_convtranspose1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::optional<core::TensorValue> & partial,
    const modules::DepthwiseConvTranspose1dWeights & weights,
    int64_t channels,
    int64_t kernel_size,
    int stride) {
    auto raw = modules::DepthwiseConvTranspose1dModule({
        channels,
        kernel_size,
        stride,
        false,
    }).build(ctx, input, weights);
    if (!partial.has_value()) {
        return {raw, std::nullopt};
    }
    const int64_t partial_frames = partial->shape.dims[2];
    auto prefix = modules::SliceModule({2, 0, partial_frames}).build(ctx, raw);
    prefix = modules::AddModule().build(ctx, prefix, *partial);
    auto suffix = modules::SliceModule({2, partial_frames, raw.shape.dims[2] - partial_frames}).build(ctx, raw);
    auto combined = modules::ConcatModule({2}).build(ctx, prefix, suffix);
    auto output = modules::SliceModule({2, 0, raw.shape.dims[2] - partial_frames}).build(ctx, combined);
    auto next_partial = modules::SliceModule({
        2,
        raw.shape.dims[2] - partial_frames,
        partial_frames,
    }).build(ctx, combined);
    return {output, next_partial};
}

core::TensorValue build_mimi_residual_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const MimiCodecResidualWeights & weights,
    int64_t channels,
    int64_t hidden_channels,
    const std::optional<core::TensorValue> & conv1_history,
    const std::optional<core::TensorValue> & conv2_history,
    std::vector<core::TensorValue> & next_histories) {
    auto x = modules::EluModule().build(ctx, input);
    auto conv1 = build_stateful_conv1d(
        ctx,
        x,
        conv1_history,
        weights.conv1,
        channels,
        hidden_channels,
        3,
        1,
        1,
        weights.conv1.bias.has_value());
    if (conv1.next_history.has_value()) {
        next_histories.push_back(*conv1.next_history);
    }
    x = modules::EluModule().build(ctx, conv1.output);
    auto conv2 = build_stateful_conv1d(
        ctx,
        x,
        conv2_history,
        weights.conv2,
        hidden_channels,
        channels,
        1,
        1,
        1,
        weights.conv2.bias.has_value());
    if (conv2.next_history.has_value()) {
        next_histories.push_back(*conv2.next_history);
    }
    return modules::ResidualAddModule().build(ctx, input, conv2.output);
}

core::TensorValue build_seanet_residual_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const SEANetResidualWeights & weights,
    int64_t channels,
    int64_t hidden_channels,
    const std::optional<core::TensorValue> & conv1_history,
    const std::optional<core::TensorValue> & conv2_history,
    std::vector<core::TensorValue> & next_histories) {
    auto x = modules::EluModule().build(ctx, input);
    auto conv1 = build_stateful_conv1d(
        ctx,
        x,
        conv1_history,
        weights.conv1,
        channels,
        hidden_channels,
        3,
        1,
        1,
        weights.conv1.bias.has_value());
    if (conv1.next_history.has_value()) {
        next_histories.push_back(*conv1.next_history);
    }
    x = modules::EluModule().build(ctx, conv1.output);
    auto conv2 = build_stateful_conv1d(
        ctx,
        x,
        conv2_history,
        weights.conv2,
        hidden_channels,
        channels,
        1,
        1,
        1,
        weights.conv2.bias.has_value());
    if (conv2.next_history.has_value()) {
        next_histories.push_back(*conv2.next_history);
    }
    return modules::ResidualAddModule().build(ctx, input, conv2.output);
}

void write_streaming_history(
    const core::TensorValue & tensor,
    const StreamingConv1dState & state,
    const std::vector<float> & current,
    int64_t channels,
    int64_t frames,
    modules::StreamingPadMode pad_mode) {
    if (state.history_frames <= 0) {
        return;
    }
    std::vector<float> history(static_cast<size_t>(channels * state.history_frames), 0.0F);
    if (pad_mode == modules::StreamingPadMode::Replicate && state.first) {
        for (int64_t channel = 0; channel < channels; ++channel) {
            std::fill_n(
                history.data() + static_cast<size_t>(channel * state.history_frames),
                static_cast<size_t>(state.history_frames),
                current[static_cast<size_t>(channel * frames)]);
        }
    } else {
        history = state.previous;
    }
    core::write_tensor_f32(tensor, history);
}

void read_streaming_history(
    const core::TensorValue & tensor,
    StreamingConv1dState & state) {
    if (state.history_frames <= 0) {
        state.first = false;
        return;
    }
    state.previous = core::read_tensor_f32(tensor.tensor);
    state.first = false;
}

void write_streaming_partial(
    const core::TensorValue & tensor,
    const StreamingConvTranspose1dState & state) {
    if (state.partial_frames > 0) {
        core::write_tensor_f32(tensor, state.partial);
    }
}

void read_streaming_partial(
    const core::TensorValue & tensor,
    StreamingConvTranspose1dState & state,
    const std::vector<float> & bias_values) {
    if (state.partial_frames <= 0) {
        return;
    }
    state.partial = core::read_tensor_f32(tensor.tensor);
    if (!bias_values.empty()) {
        for (int64_t channel = 0; channel < state.channels; ++channel) {
            const float bias = bias_values[static_cast<size_t>(channel)];
            for (int64_t frame = 0; frame < state.partial_frames; ++frame) {
                state.partial[static_cast<size_t>(channel * state.partial_frames + frame)] -= bias;
            }
        }
    }
}

class MimiEncoderPreTransformerGraph {
public:
    MimiEncoderPreTransformerGraph(
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        size_t graph_context_bytes,
        std::shared_ptr<const MimiCodecWeights> weights,
        MimiCodecConfig config)
        : backend_(backend),
          threads_(threads),
          weights_(std::move(weights)),
          config_(std::move(config)) {
        if (weights_ == nullptr) {
            throw std::runtime_error("Mimi codec encoder pre-transformer graph requires weights");
        }
        ggml_ctx_ = ggml_init({graph_context_bytes, nullptr, true});
        if (ggml_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Mimi codec encoder pre-transformer graph context");
        }
        core::ModuleBuildContext ctx{ggml_ctx_, "mimi_codec.encoder_pre_transformer", backend_type};
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config_.channels, kMimiFrameSamples}));

        std::vector<core::TensorValue> residual_histories;
        auto input_history = make_history_tensor(ctx, config_.channels, 6);
        auto input_projection = build_stateful_conv1d(
            ctx,
            input_,
            input_history,
            weights_->encoder.input_projection,
            config_.channels,
            64,
            7,
            1,
            1,
            weights_->encoder.input_projection.bias.has_value());
        next_histories_.push_back(*input_projection.next_history);
        auto x = input_projection.output;

        const int64_t channels[] = {64, 128, 256, 512};
        const int64_t hidden_channels[] = {32, 64, 128, 256};
        const int64_t downsample_kernels[] = {8, 10, 12, 16};
        const int downsample_strides[] = {4, 5, 6, 8};
        for (size_t stage = 0; stage < 4; ++stage) {
            x = build_mimi_residual_block(
                ctx,
                x,
                weights_->encoder.residual_blocks[stage],
                channels[stage],
                hidden_channels[stage],
                make_history_tensor(ctx, channels[stage], 2),
                make_history_tensor(ctx, hidden_channels[stage], 0),
                residual_histories);
            x = modules::EluModule().build(ctx, x);
            auto downsample = build_stateful_conv1d(
                ctx,
                x,
                make_history_tensor(ctx, channels[stage], downsample_kernels[stage] - downsample_strides[stage]),
                weights_->encoder.downsample_layers[stage],
                channels[stage],
                channels[stage] * 2,
                downsample_kernels[stage],
                downsample_strides[stage],
                1,
                weights_->encoder.downsample_layers[stage].bias.has_value());
            x = downsample.output;
            next_histories_.push_back(*downsample.next_history);
        }
        next_histories_.insert(next_histories_.end(), residual_histories.begin(), residual_histories.end());

        x = modules::EluModule().build(ctx, x);
        auto output_projection = build_stateful_conv1d(
            ctx,
            x,
            make_history_tensor(ctx, 1024, 2),
            weights_->encoder.output_projection,
            1024,
            config_.hidden_size,
            3,
            1,
            1,
            weights_->encoder.output_projection.bias.has_value());
        next_histories_.push_back(*output_projection.next_history);
        output_ = output_projection.output;
        build_runtime(ctx);
    }

    ~MimiEncoderPreTransformerGraph() {
        release_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
    }

    std::vector<float> run(const std::vector<float> & input, MimiEncoderState & state) const {
        if (static_cast<int64_t>(input.size()) != config_.channels * kMimiFrameSamples) {
            throw std::runtime_error("Mimi codec encoder pre-transformer input size mismatch");
        }
        core::write_tensor_f32(input_, input);
        size_t history = 0;
        write_streaming_history(history_tensors_[history++], state.input_projection, input, config_.channels, kMimiFrameSamples, modules::StreamingPadMode::Constant);
        for (size_t stage = 0; stage < 4; ++stage) {
            write_streaming_history(history_tensors_[history++], state.stage_residual_convs[stage][0], {}, 0, 0, modules::StreamingPadMode::Constant);
            write_streaming_history(history_tensors_[history++], state.stage_downsamples[stage], {}, 0, 0, modules::StreamingPadMode::Constant);
        }
        write_streaming_history(history_tensors_[history++], state.output_projection, {}, 0, 0, modules::StreamingPadMode::Constant);
        if (core::compute_backend_graph(backend_, graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Mimi codec encoder pre-transformer graph compute failed");
        }
        size_t output_history = 0;
        read_streaming_history(next_histories_[output_history++], state.input_projection);
        for (size_t stage = 0; stage < 4; ++stage) {
            read_streaming_history(next_histories_[output_history++], state.stage_downsamples[stage]);
        }
        for (size_t stage = 0; stage < 4; ++stage) {
            read_streaming_history(next_histories_[output_history++], state.stage_residual_convs[stage][0]);
        }
        read_streaming_history(next_histories_[output_history++], state.output_projection);
        return core::read_tensor_f32(output_.tensor);
    }

private:
    std::optional<core::TensorValue> make_history_tensor(
        core::ModuleBuildContext & ctx,
        int64_t channels,
        int64_t frames) {
        if (frames <= 0) {
            return std::nullopt;
        }
        history_tensors_.push_back(core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, channels, frames})));
        return history_tensors_.back();
    }

    void build_runtime(core::ModuleBuildContext &) {
        if (core::is_host_backend(backend_)) {
            params_buffer_ = ggml_backend_alloc_ctx_tensors(ggml_ctx_, backend_);
            if (params_buffer_ == nullptr) {
                release_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
                throw std::runtime_error("Mimi codec encoder pre-transformer context tensor allocation failed");
            }
        }
        core::set_backend_threads(backend_, threads_);
        graph_ = ggml_new_graph_custom(ggml_ctx_, 65536, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        for (const auto & history : next_histories_) {
            ggml_build_forward_expand(graph_, history.tensor);
        }
        galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (galloc_ == nullptr || !ggml_gallocr_reserve(galloc_, graph_) || !ggml_gallocr_alloc_graph(galloc_, graph_)) {
            release_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
            throw std::runtime_error("Mimi codec encoder pre-transformer graph allocation failed");
        }
    }

    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
    std::shared_ptr<const MimiCodecWeights> weights_;
    MimiCodecConfig config_;
    ggml_context * ggml_ctx_ = nullptr;
    ggml_backend_buffer_t params_buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t galloc_ = nullptr;
    core::TensorValue input_;
    std::vector<core::TensorValue> history_tensors_;
    std::vector<core::TensorValue> next_histories_;
    core::TensorValue output_;
};

class MimiEncoderPostTransformerGraph {
public:
    MimiEncoderPostTransformerGraph(
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        size_t graph_context_bytes,
        std::shared_ptr<const MimiCodecWeights> weights,
        MimiCodecConfig config,
        int64_t frames)
        : backend_(backend),
          threads_(threads),
          weights_(std::move(weights)),
          config_(std::move(config)),
          frames_(frames) {
        if (weights_ == nullptr || frames_ <= 0) {
            throw std::runtime_error("Mimi codec encoder post-transformer graph requires weights and frames");
        }
        ggml_ctx_ = ggml_init({graph_context_bytes, nullptr, true});
        if (ggml_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Mimi codec encoder post-transformer graph context");
        }
        core::ModuleBuildContext ctx{ggml_ctx_, "mimi_codec.encoder_post_transformer", backend_type};
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config_.hidden_size, frames_}));
        downsample_history_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config_.hidden_size, 2}));
        auto downsample = build_stateful_conv1d(
            ctx,
            input_,
            downsample_history_,
            weights_->encoder.downsample_conv,
            config_.hidden_size,
            config_.hidden_size,
            4,
            2,
            1,
            weights_->encoder.downsample_conv.bias.has_value());
        next_downsample_history_ = *downsample.next_history;
        if (downsample.output.shape.dims[2] != 1) {
            throw std::runtime_error("Mimi codec encoder post-transformer expected one latent frame");
        }
        auto latent = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, downsample.output);
        latent = modules::ReshapeModule({core::TensorShape::from_dims({1, config_.hidden_size})}).build(ctx, latent);
        const auto semantic_weight = core::reshape_tensor(
            ctx,
            weights_->quantizer.semantic_input_proj.weight,
            core::TensorShape::from_dims({config_.latent_size, config_.hidden_size}));
        semantic_ = modules::LinearModule({
            config_.hidden_size,
            config_.latent_size,
            weights_->quantizer.semantic_input_proj.bias.has_value(),
            GGML_PREC_F32,
        }).build(ctx, latent, modules::LinearWeights{semantic_weight, weights_->quantizer.semantic_input_proj.bias});
        const auto acoustic_weight = core::reshape_tensor(
            ctx,
            weights_->quantizer.acoustic_input_proj.weight,
            core::TensorShape::from_dims({config_.latent_size, config_.hidden_size}));
        acoustic_ = modules::LinearModule({
            config_.hidden_size,
            config_.latent_size,
            weights_->quantizer.acoustic_input_proj.bias.has_value(),
            GGML_PREC_F32,
        }).build(ctx, latent, modules::LinearWeights{acoustic_weight, weights_->quantizer.acoustic_input_proj.bias});
        build_runtime();
    }

    ~MimiEncoderPostTransformerGraph() {
        release_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
    }

    std::pair<std::vector<float>, std::vector<float>> run(
        const std::vector<float> & input,
        MimiEncoderState & state) const {
        if (static_cast<int64_t>(input.size()) != config_.hidden_size * frames_) {
            throw std::runtime_error("Mimi codec encoder post-transformer input size mismatch");
        }
        core::write_tensor_f32(input_, input);
        write_streaming_history(
            downsample_history_,
            state.downsample,
            input,
            config_.hidden_size,
            frames_,
            modules::StreamingPadMode::Replicate);
        if (core::compute_backend_graph(backend_, graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Mimi codec encoder post-transformer graph compute failed");
        }
        read_streaming_history(next_downsample_history_, state.downsample);
        return {core::read_tensor_f32(semantic_.tensor), core::read_tensor_f32(acoustic_.tensor)};
    }

private:
    void build_runtime() {
        if (core::is_host_backend(backend_)) {
            params_buffer_ = ggml_backend_alloc_ctx_tensors(ggml_ctx_, backend_);
            if (params_buffer_ == nullptr) {
                release_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
                throw std::runtime_error("Mimi codec encoder post-transformer context tensor allocation failed");
            }
        }
        core::set_backend_threads(backend_, threads_);
        graph_ = ggml_new_graph_custom(ggml_ctx_, 32768, false);
        ggml_build_forward_expand(graph_, semantic_.tensor);
        ggml_build_forward_expand(graph_, acoustic_.tensor);
        ggml_build_forward_expand(graph_, next_downsample_history_.tensor);
        galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (galloc_ == nullptr || !ggml_gallocr_reserve(galloc_, graph_) || !ggml_gallocr_alloc_graph(galloc_, graph_)) {
            release_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
            throw std::runtime_error("Mimi codec encoder post-transformer graph allocation failed");
        }
    }

    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
    std::shared_ptr<const MimiCodecWeights> weights_;
    MimiCodecConfig config_;
    int64_t frames_ = 0;
    ggml_context * ggml_ctx_ = nullptr;
    ggml_backend_buffer_t params_buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t galloc_ = nullptr;
    core::TensorValue input_;
    core::TensorValue downsample_history_;
    core::TensorValue next_downsample_history_;
    core::TensorValue semantic_;
    core::TensorValue acoustic_;
};

class MimiDecoderPreTransformerGraph {
public:
    MimiDecoderPreTransformerGraph(
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        size_t graph_context_bytes,
        std::shared_ptr<const MimiCodecWeights> weights,
        MimiCodecConfig config)
        : backend_(backend),
          threads_(threads),
          weights_(std::move(weights)),
          config_(std::move(config)) {
        if (weights_ == nullptr) {
            throw std::runtime_error("Mimi codec decoder pre-transformer graph requires weights");
        }
        ggml_ctx_ = ggml_init({graph_context_bytes, nullptr, true});
        if (ggml_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Mimi codec decoder pre-transformer graph context");
        }
        core::ModuleBuildContext ctx{ggml_ctx_, "mimi_codec.decoder_pre_transformer", backend_type};
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config_.hidden_size, 1}));
        partial_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config_.hidden_size, 2}));
        auto upsample = build_stateful_depthwise_convtranspose1d(
            ctx,
            input_,
            partial_,
            weights_->decoder.upsample_conv,
            config_.hidden_size,
            4,
            2);
        output_ = upsample.output;
        next_partial_ = *upsample.next_partial;
        build_runtime();
    }

    ~MimiDecoderPreTransformerGraph() {
        release_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
    }

    std::vector<float> run(const std::vector<float> & input, MimiDecoderState & state) const {
        if (static_cast<int64_t>(input.size()) != config_.hidden_size) {
            throw std::runtime_error("Mimi codec decoder pre-transformer input size mismatch");
        }
        core::write_tensor_f32(input_, input);
        write_streaming_partial(partial_, state.encoder_rate_upsample);
        if (core::compute_backend_graph(backend_, graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Mimi codec decoder pre-transformer graph compute failed");
        }
        read_streaming_partial(next_partial_, state.encoder_rate_upsample, {});
        return core::read_tensor_f32(output_.tensor);
    }

private:
    void build_runtime() {
        if (core::is_host_backend(backend_)) {
            params_buffer_ = ggml_backend_alloc_ctx_tensors(ggml_ctx_, backend_);
            if (params_buffer_ == nullptr) {
                release_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
                throw std::runtime_error("Mimi codec decoder pre-transformer context tensor allocation failed");
            }
        }
        core::set_backend_threads(backend_, threads_);
        graph_ = ggml_new_graph_custom(ggml_ctx_, 32768, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        ggml_build_forward_expand(graph_, next_partial_.tensor);
        galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (galloc_ == nullptr || !ggml_gallocr_reserve(galloc_, graph_) || !ggml_gallocr_alloc_graph(galloc_, graph_)) {
            release_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
            throw std::runtime_error("Mimi codec decoder pre-transformer graph allocation failed");
        }
    }

    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
    std::shared_ptr<const MimiCodecWeights> weights_;
    MimiCodecConfig config_;
    ggml_context * ggml_ctx_ = nullptr;
    ggml_backend_buffer_t params_buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t galloc_ = nullptr;
    core::TensorValue input_;
    core::TensorValue partial_;
    core::TensorValue next_partial_;
    core::TensorValue output_;
};

class MimiDecoderPostTransformerGraph {
public:
    MimiDecoderPostTransformerGraph(
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        size_t graph_context_bytes,
        std::shared_ptr<const MimiCodecWeights> weights,
        MimiCodecConfig config,
        int64_t frames)
        : backend_(backend),
          threads_(threads),
          weights_(std::move(weights)),
          config_(std::move(config)),
          frames_(frames) {
        if (weights_ == nullptr || frames_ <= 0) {
            throw std::runtime_error("Mimi codec decoder post-transformer graph requires weights and frames");
        }
        ggml_ctx_ = ggml_init({graph_context_bytes, nullptr, true});
        if (ggml_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Mimi codec decoder post-transformer graph context");
        }
        core::ModuleBuildContext ctx{ggml_ctx_, "mimi_codec.decoder_post_transformer", backend_type};
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config_.hidden_size, frames_}));
        auto input_projection = build_stateful_conv1d(
            ctx,
            input_,
            make_history_tensor(ctx, config_.hidden_size, 6),
            weights_->decoder.seanet.input_projection,
            config_.hidden_size,
            1024,
            7,
            1,
            1,
            weights_->decoder.seanet.input_projection.bias.has_value());
        next_histories_.push_back(*input_projection.next_history);
        auto x = input_projection.output;

        const int64_t stage_in_channels[] = {1024, 512, 256, 128};
        const int64_t stage_out_channels[] = {512, 256, 128, 64};
        const int64_t stage_hidden_channels[] = {256, 128, 64, 32};
        const int64_t stage_kernel_sizes[] = {16, 12, 10, 8};
        const int stage_strides[] = {8, 6, 5, 4};
        std::vector<core::TensorValue> residual_histories;
        for (size_t stage = 0; stage < 4; ++stage) {
            x = modules::EluModule().build(ctx, x);
            auto upsample = build_stateful_convtranspose1d(
                ctx,
                x,
                make_partial_tensor(ctx, stage_out_channels[stage], stage_kernel_sizes[stage] - stage_strides[stage]),
                weights_->decoder.seanet.stages[stage].upsample,
                stage_in_channels[stage],
                stage_out_channels[stage],
                stage_kernel_sizes[stage],
                stage_strides[stage],
                weights_->decoder.seanet.stages[stage].upsample.bias.has_value());
            next_partials_.push_back(*upsample.next_partial);
            x = build_seanet_residual_block(
                ctx,
                upsample.output,
                weights_->decoder.seanet.stages[stage].residual_blocks.front(),
                stage_out_channels[stage],
                stage_hidden_channels[stage],
                make_history_tensor(ctx, stage_out_channels[stage], 2),
                make_history_tensor(ctx, stage_hidden_channels[stage], 0),
                residual_histories);
        }
        next_histories_.insert(next_histories_.end(), residual_histories.begin(), residual_histories.end());
        x = modules::EluModule().build(ctx, x);
        auto output_projection = build_stateful_conv1d(
            ctx,
            x,
            make_history_tensor(ctx, 64, 2),
            weights_->decoder.seanet.output_projection,
            64,
            config_.channels,
            3,
            1,
            1,
            weights_->decoder.seanet.output_projection.bias.has_value());
        next_histories_.push_back(*output_projection.next_history);
        output_ = output_projection.output;
        build_runtime();
    }

    ~MimiDecoderPostTransformerGraph() {
        release_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
    }

    std::vector<float> run(const std::vector<float> & input, MimiDecoderState & state) const {
        if (static_cast<int64_t>(input.size()) != config_.hidden_size * frames_) {
            throw std::runtime_error("Mimi codec decoder post-transformer input size mismatch");
        }
        core::write_tensor_f32(input_, input);
        size_t history = 0;
        write_streaming_history(history_tensors_[history++], state.input_projection, input, config_.hidden_size, frames_, modules::StreamingPadMode::Constant);
        for (size_t stage = 0; stage < 4; ++stage) {
            write_streaming_partial(partial_tensors_[stage], state.stage_upsamples[stage]);
            write_streaming_history(history_tensors_[history++], state.stage_residual_convs[stage][0], {}, 0, 0, modules::StreamingPadMode::Constant);
        }
        write_streaming_history(history_tensors_[history++], state.output_projection, {}, 0, 0, modules::StreamingPadMode::Constant);
        if (core::compute_backend_graph(backend_, graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Mimi codec decoder post-transformer graph compute failed");
        }
        size_t output_history = 0;
        read_streaming_history(next_histories_[output_history++], state.input_projection);
        for (size_t stage = 0; stage < 4; ++stage) {
            read_streaming_history(next_histories_[output_history++], state.stage_residual_convs[stage][0]);
        }
        read_streaming_history(next_histories_[output_history++], state.output_projection);
        for (size_t stage = 0; stage < 4; ++stage) {
            read_streaming_partial(next_partials_[stage], state.stage_upsamples[stage], weights_->decoder.seanet_upsample_bias_values[stage]);
        }
        auto output = core::read_tensor_f32(output_.tensor);
        for (float & sample : output) {
            sample = std::clamp(sample, -1.0F, 1.0F);
        }
        return output;
    }

private:
    std::optional<core::TensorValue> make_history_tensor(
        core::ModuleBuildContext & ctx,
        int64_t channels,
        int64_t frames) {
        if (frames <= 0) {
            return std::nullopt;
        }
        history_tensors_.push_back(core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, channels, frames})));
        return history_tensors_.back();
    }

    std::optional<core::TensorValue> make_partial_tensor(
        core::ModuleBuildContext & ctx,
        int64_t channels,
        int64_t frames) {
        if (frames <= 0) {
            return std::nullopt;
        }
        partial_tensors_.push_back(core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, channels, frames})));
        return partial_tensors_.back();
    }

    void build_runtime() {
        if (core::is_host_backend(backend_)) {
            params_buffer_ = ggml_backend_alloc_ctx_tensors(ggml_ctx_, backend_);
            if (params_buffer_ == nullptr) {
                release_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
                throw std::runtime_error("Mimi codec decoder post-transformer context tensor allocation failed");
            }
        }
        core::set_backend_threads(backend_, threads_);
        graph_ = ggml_new_graph_custom(ggml_ctx_, 65536, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        for (const auto & history : next_histories_) {
            ggml_build_forward_expand(graph_, history.tensor);
        }
        for (const auto & partial : next_partials_) {
            ggml_build_forward_expand(graph_, partial.tensor);
        }
        galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (galloc_ == nullptr || !ggml_gallocr_reserve(galloc_, graph_) || !ggml_gallocr_alloc_graph(galloc_, graph_)) {
            release_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
            throw std::runtime_error("Mimi codec decoder post-transformer graph allocation failed");
        }
    }

    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
    std::shared_ptr<const MimiCodecWeights> weights_;
    MimiCodecConfig config_;
    int64_t frames_ = 0;
    ggml_context * ggml_ctx_ = nullptr;
    ggml_backend_buffer_t params_buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t galloc_ = nullptr;
    core::TensorValue input_;
    std::vector<core::TensorValue> history_tensors_;
    std::vector<core::TensorValue> partial_tensors_;
    std::vector<core::TensorValue> next_histories_;
    std::vector<core::TensorValue> next_partials_;
    core::TensorValue output_;
};

class MimiTransformerRuntime {
public:
    MimiTransformerRuntime(
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        size_t graph_context_bytes,
        const std::vector<MimiCodecTransformerLayerWeights> & layers,
        const MimiCodecConfig & config,
        int64_t frames)
        : backend_(backend),
          threads_(threads),
          config_(config),
          frames_(frames),
          cache_steps_(std::max<int64_t>(0, config.context - frames)),
          head_dim_(config.hidden_size / config.num_heads) {
        ggml_ctx_ = ggml_init({graph_context_bytes, nullptr, true});
        if (ggml_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Mimi codec transformer runtime context");
        }
        core::ModuleBuildContext ctx{ggml_ctx_, "mimi_codec.transformer", backend_type};
        input_bct_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config.hidden_size, frames}));
        positions_buffer_.resize(static_cast<size_t>(frames));
        positions_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({frames}));
        attention_mask_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, frames, cache_steps_ + frames}));
        auto x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input_bct_);
        for (int64_t layer = 0; layer < config.transformer_layers; ++layer) {
            std::optional<core::TensorValue> prefix_key = std::nullopt;
            std::optional<core::TensorValue> prefix_value = std::nullopt;
            if (cache_steps_ > 0) {
                prefix_key = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config.num_heads, cache_steps_, head_dim_}));
                prefix_value = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config.num_heads, cache_steps_, head_dim_}));
                prefix_keys_.push_back(*prefix_key);
                prefix_values_.push_back(*prefix_value);
            }
            auto outputs = modules::StreamingTransformerEncoderBlockModule({
                config.hidden_size,
                config.num_heads,
                config.intermediate_size,
                1.0e-5F,
                false,
                GGML_PREC_F32,
                GGML_PREC_F32,
                modules::AttentionPrefixCacheLayout::HeadsSequence,
            }).build(
                ctx,
                x,
                positions_,
                as_transformer_layer_weights(layers.at(static_cast<size_t>(layer))),
                prefix_key,
                prefix_value,
                attention_mask_);
            x = outputs.output;
            keys_.push_back(outputs.key);
            values_.push_back(outputs.value);
        }
        output_bct_ = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        build_transfer_views();
        if (core::is_host_backend(backend_)) {
            params_buffer_ = ggml_backend_alloc_ctx_tensors(ggml_ctx_, backend_);
            if (params_buffer_ == nullptr) {
                release_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
                throw std::runtime_error("Mimi codec transformer context tensor allocation failed");
            }
        }
        core::set_backend_threads(backend_, threads_);
        graph_ = ggml_new_graph_custom(ggml_ctx_, 65536, false);
        ggml_build_forward_expand(graph_, output_bct_.tensor);
        for (size_t layer = 0; layer < keys_.size(); ++layer) {
            ggml_build_forward_expand(graph_, keys_[layer].tensor);
            ggml_build_forward_expand(graph_, values_[layer].tensor);
        }
        for (auto * view : key_update_sources_) {
            ggml_build_forward_expand(graph_, view);
        }
        for (auto * view : value_update_sources_) {
            ggml_build_forward_expand(graph_, view);
        }
        for (auto * view : key_update_destinations_) {
            ggml_build_forward_expand(graph_, view);
        }
        for (auto * view : value_update_destinations_) {
            ggml_build_forward_expand(graph_, view);
        }
        galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (galloc_ == nullptr || !ggml_gallocr_reserve(galloc_, graph_) || !ggml_gallocr_alloc_graph(galloc_, graph_)) {
            release_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
            throw std::runtime_error("Mimi codec transformer graph allocation failed");
        }
        core::write_tensor_f32(input_bct_, std::vector<float>(static_cast<size_t>(config.hidden_size * frames), 0.0F));
        core::write_tensor_f32(attention_mask_, transformer_attention_mask(frames_, cache_steps_, config_.context, 0));
        if (cache_steps_ > 0) {
            const std::vector<float> zero_prefix(static_cast<size_t>(cache_steps_ * config.num_heads * head_dim_), 0.0F);
            for (size_t layer = 0; layer < prefix_keys_.size(); ++layer) {
                core::write_tensor_f32(prefix_keys_[layer], zero_prefix);
                core::write_tensor_f32(prefix_values_[layer], zero_prefix);
            }
        }
    }

    ~MimiTransformerRuntime() {
        release_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
    }

    void reset_sequence(int64_t current_end) {
        current_end_ = current_end;
        sequence_initialized_ = true;
        for (size_t layer = 0; layer < prefix_keys_.size(); ++layer) {
            const std::vector<float> zero_prefix(static_cast<size_t>(cache_steps_ * config_.num_heads * head_dim_), 0.0F);
            core::write_tensor_f32(prefix_keys_[layer], zero_prefix);
            core::write_tensor_f32(prefix_values_[layer], zero_prefix);
        }
    }

    std::vector<float> run(const std::vector<float> & input_bct) {
        if (static_cast<int64_t>(input_bct.size()) != config_.hidden_size * frames_) {
            throw std::runtime_error("Mimi codec transformer input size mismatch");
        }
        if (!sequence_initialized_) {
            throw std::runtime_error("Mimi codec transformer must be reset before run");
        }
        core::write_tensor_f32(input_bct_, input_bct);
        for (int64_t index = 0; index < frames_; ++index) {
            positions_buffer_[static_cast<size_t>(index)] = static_cast<int32_t>(current_end_ + index);
        }
        core::write_tensor_i32(positions_, positions_buffer_);
        core::write_tensor_f32(attention_mask_, transformer_attention_mask(frames_, cache_steps_, config_.context, current_end_));
        if (core::compute_backend_graph(backend_, graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Mimi codec transformer graph compute failed");
        }
        auto output = core::read_tensor_f32(output_bct_.tensor);
        if (cache_steps_ > 0) {
            for (size_t layer = 0; layer < prefix_keys_.size(); ++layer) {
                for (int64_t frame = 0; frame < frames_; ++frame) {
                    const int64_t slot = (current_end_ + frame) % cache_steps_;
                    for (int64_t head = 0; head < config_.num_heads; ++head) {
                        const size_t view_index = static_cast<size_t>((frame * config_.num_heads + head) * config_.transformer_layers + static_cast<int64_t>(layer));
                        const size_t dst_index = static_cast<size_t>((slot * config_.num_heads + head) * config_.transformer_layers + static_cast<int64_t>(layer));
                        ggml_backend_tensor_copy(key_update_sources_[view_index], key_update_destinations_[dst_index]);
                        ggml_backend_tensor_copy(value_update_sources_[view_index], value_update_destinations_[dst_index]);
                    }
                }
            }
        }
        current_end_ += frames_;
        return output;
    }

private:
    void build_transfer_views() {
        if (cache_steps_ <= 0) {
            return;
        }
        const size_t source_count = static_cast<size_t>(frames_ * config_.num_heads * config_.transformer_layers);
        const size_t destination_count = static_cast<size_t>(cache_steps_ * config_.num_heads * config_.transformer_layers);
        key_update_sources_.assign(source_count, nullptr);
        value_update_sources_.assign(source_count, nullptr);
        key_update_destinations_.assign(destination_count, nullptr);
        value_update_destinations_.assign(destination_count, nullptr);
        for (int64_t frame = 0; frame < frames_; ++frame) {
            for (int64_t head = 0; head < config_.num_heads; ++head) {
                const size_t source_offset = static_cast<size_t>((head * frames_ + frame) * head_dim_) * sizeof(float);
                for (size_t layer = 0; layer < keys_.size(); ++layer) {
                    const size_t index = static_cast<size_t>(
                        (frame * config_.num_heads + head) * config_.transformer_layers + static_cast<int64_t>(layer));
                    key_update_sources_[index] = ggml_view_1d(ggml_ctx_, keys_[layer].tensor, head_dim_, source_offset);
                    value_update_sources_[index] = ggml_view_1d(ggml_ctx_, values_[layer].tensor, head_dim_, source_offset);
                }
            }
        }
        for (int64_t slot = 0; slot < cache_steps_; ++slot) {
            for (int64_t head = 0; head < config_.num_heads; ++head) {
                const size_t destination_offset = static_cast<size_t>((head * cache_steps_ + slot) * head_dim_) * sizeof(float);
                for (size_t layer = 0; layer < prefix_keys_.size(); ++layer) {
                    const size_t index = static_cast<size_t>(
                        (slot * config_.num_heads + head) * config_.transformer_layers + static_cast<int64_t>(layer));
                    key_update_destinations_[index] = ggml_view_1d(
                        ggml_ctx_,
                        prefix_keys_[layer].tensor,
                        head_dim_,
                        destination_offset);
                    value_update_destinations_[index] = ggml_view_1d(
                        ggml_ctx_,
                        prefix_values_[layer].tensor,
                        head_dim_,
                        destination_offset);
                }
            }
        }
    }

    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
    MimiCodecConfig config_;
    int64_t frames_ = 0;
    int64_t cache_steps_ = 0;
    int64_t head_dim_ = 0;
    ggml_context * ggml_ctx_ = nullptr;
    ggml_backend_buffer_t params_buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t galloc_ = nullptr;
    core::TensorValue input_bct_;
    core::TensorValue positions_;
    core::TensorValue attention_mask_;
    std::vector<core::TensorValue> prefix_keys_;
    std::vector<core::TensorValue> prefix_values_;
    std::vector<core::TensorValue> keys_;
    std::vector<core::TensorValue> values_;
    std::vector<int32_t> positions_buffer_;
    std::vector<ggml_tensor *> key_update_sources_;
    std::vector<ggml_tensor *> value_update_sources_;
    std::vector<ggml_tensor *> key_update_destinations_;
    std::vector<ggml_tensor *> value_update_destinations_;
    core::TensorValue output_bct_;
    int64_t current_end_ = 0;
    bool sequence_initialized_ = false;
};

core::TensorValue make_qkv_slice(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & name,
    assets::TensorStorageType storage_type,
    int64_t hidden_size,
    int64_t slice_index) {
    const auto packed = source.require_f32(name, {3 * hidden_size, hidden_size});
    std::vector<float> slice(static_cast<size_t>(hidden_size * hidden_size));
    const size_t offset = static_cast<size_t>(slice_index * hidden_size * hidden_size);
    std::copy_n(packed.begin() + static_cast<std::ptrdiff_t>(offset), slice.size(), slice.begin());
    return store.make_from_f32(
        core::TensorShape::from_dims({hidden_size, hidden_size}),
        storage_type,
        std::move(slice));
}

modules::AttentionWeights load_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t hidden_size) {
    const std::string packed_name = prefix + "self_attn.in_proj_weight";
    modules::AttentionWeights weights;
    weights.q_weight = make_qkv_slice(store, source, packed_name, storage_type, hidden_size, 0);
    weights.k_weight = make_qkv_slice(store, source, packed_name, storage_type, hidden_size, 1);
    weights.v_weight = make_qkv_slice(store, source, packed_name, storage_type, hidden_size, 2);
    weights.out_weight = binding::tensor_from_named_source(
        store,
        source,
        prefix + "self_attn.out_proj.weight",
        storage_type);
    return weights;
}

MimiCodecTransformerLayerWeights load_transformer_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t hidden_size) {
    MimiCodecTransformerLayerWeights layer;
    layer.norm1 = binding::norm_from_named_source(store, source, prefix + "norm1.weight", prefix + "norm1.bias");
    layer.self_attn = load_attention(store, source, prefix, storage_type, hidden_size);
    layer.layer_scale1 = binding::layer_scale_from_named_source(store, source, prefix + "layer_scale_1.scale");
    layer.norm2 = binding::norm_from_named_source(store, source, prefix + "norm2.weight", prefix + "norm2.bias");
    layer.feed_forward = modules::FeedForwardWeights{
        binding::tensor_from_named_source(store, source, prefix + "linear1.weight", storage_type),
        std::nullopt,
        binding::tensor_from_named_source(store, source, prefix + "linear2.weight", storage_type),
        std::nullopt,
    };
    layer.layer_scale2 = binding::layer_scale_from_named_source(store, source, prefix + "layer_scale_2.scale");
    return layer;
}

MimiCodecResidualWeights load_residual_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type) {
    return {
        binding::conv1d_from_named_source(store, source, prefix + ".block.1.conv.conv.weight", prefix + ".block.1.conv.conv.bias", storage_type),
        binding::conv1d_from_named_source(store, source, prefix + ".block.3.conv.conv.weight", prefix + ".block.3.conv.conv.bias", storage_type),
    };
}

std::vector<float> normalized_codebook(
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t size,
    int64_t dim) {
    const auto cluster_usage = source.require_f32(prefix + "cluster_usage", {size});
    const auto embedding_sum = source.require_f32(prefix + "embedding_sum", {size, dim});
    std::vector<float> embedding(embedding_sum.size(), 0.0F);
    for (int64_t code = 0; code < size; ++code) {
        const float denom = std::max(cluster_usage[static_cast<size_t>(code)], kCodebookEps);
        for (int64_t col = 0; col < dim; ++col) {
            const size_t offset = static_cast<size_t>(code * dim + col);
            embedding[offset] = embedding_sum[offset] / denom;
        }
    }
    return embedding;
}

MimiCodecCodebookWeights load_codebook(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t size,
    int64_t dim) {
    auto embedding = normalized_codebook(source, prefix, size, dim);
    return {
        store.make_f32(
            core::TensorShape::from_dims({size, dim}),
            std::vector<float>(embedding)),
        std::move(embedding),
    };
}

void load_encoder_weights(
    MimiCodecEncoderWeights & weights,
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const MimiCodecConfig & config,
    const MimiCodecWeightBinding & weight_binding,
    assets::TensorStorageType storage_type) {
    weights.input_projection = binding::conv1d_from_named_source(
        store,
        source,
        weight_binding.encoder_input_projection_weight,
        weight_binding.encoder_input_projection_bias,
        storage_type);
    weights.residual_blocks.reserve(weight_binding.encoder_residual_prefixes.size());
    for (const auto & prefix : weight_binding.encoder_residual_prefixes) {
        weights.residual_blocks.push_back(load_residual_block(store, source, prefix, storage_type));
    }
    if (weight_binding.encoder_downsample_weight_names.size() != weight_binding.encoder_downsample_bias_names.size()) {
        throw std::runtime_error("Mimi codec encoder downsample binding size mismatch");
    }
    weights.downsample_layers.reserve(weight_binding.encoder_downsample_weight_names.size());
    for (size_t index = 0; index < weight_binding.encoder_downsample_weight_names.size(); ++index) {
        weights.downsample_layers.push_back(binding::conv1d_from_named_source(
            store,
            source,
            weight_binding.encoder_downsample_weight_names[index],
            weight_binding.encoder_downsample_bias_names[index],
            storage_type));
    }
    weights.output_projection = binding::conv1d_from_named_source(
        store,
        source,
        weight_binding.encoder_output_projection_weight,
        weight_binding.encoder_output_projection_bias,
        storage_type);
    weights.transformer_layers.reserve(static_cast<size_t>(config.transformer_layers));
    for (int64_t layer = 0; layer < config.transformer_layers; ++layer) {
        weights.transformer_layers.push_back(load_transformer_layer(
            store,
            source,
            weight_binding.encoder_transformer_layer_prefix + std::to_string(layer) + ".",
            storage_type,
            config.hidden_size));
    }
    weights.downsample_conv = binding::conv1d_from_named_source(
        store,
        source,
        weight_binding.downsample_weight,
        std::nullopt,
        storage_type);
}

void load_quantizer_weights(
    MimiCodecQuantizerWeights & weights,
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const MimiCodecConfig & config,
    const MimiCodecWeightBinding & weight_binding,
    assets::TensorStorageType storage_type) {
    weights.semantic_codebooks.push_back(load_codebook(
        store,
        source,
        weight_binding.semantic_codebook_prefix,
        config.codebook_size,
        config.latent_size));
    weights.acoustic_codebooks.reserve(static_cast<size_t>(config.total_codebooks - config.codebooks));
    for (int64_t layer = 0; layer < config.total_codebooks - 1; ++layer) {
        weights.acoustic_codebooks.push_back(load_codebook(
            store,
            source,
            weight_binding.acoustic_codebook_layer_prefix + std::to_string(layer) + weight_binding.acoustic_codebook_suffix,
            config.codebook_size,
            config.latent_size));
    }
    weights.semantic_input_proj = binding::conv1d_from_named_source(
        store,
        source,
        weight_binding.semantic_input_projection_weight,
        std::nullopt,
        storage_type);
    weights.acoustic_input_proj = binding::conv1d_from_named_source(
        store,
        source,
        weight_binding.acoustic_input_projection_weight,
        std::nullopt,
        storage_type);
    weights.semantic_output_proj = binding::conv1d_from_named_source(
        store,
        source,
        weight_binding.semantic_output_projection_weight,
        std::nullopt,
        storage_type);
    weights.acoustic_output_proj = binding::conv1d_from_named_source(
        store,
        source,
        weight_binding.acoustic_output_projection_weight,
        std::nullopt,
        storage_type);
}

SEANetResidualWeights to_seanet_residual(const MimiCodecResidualWeights & weights) {
    return {weights.conv1, weights.conv2};
}

core::TensorValue codebook_decode(
    core::ModuleBuildContext & ctx,
    ggml_tensor * codes_t_q_b,
    const MimiCodecCodebookWeights & codebook,
    const MimiCodecConfig & config,
    int64_t codebook_index) {
    auto * code_slice = ggml_view_2d(
        ctx.ggml,
        codes_t_q_b,
        codes_t_q_b->ne[0],
        codes_t_q_b->ne[2],
        codes_t_q_b->nb[2],
        static_cast<size_t>(codebook_index) * codes_t_q_b->nb[1]);
    auto indices = core::wrap_tensor(
        code_slice,
        core::TensorShape::from_dims({code_slice->ne[1], code_slice->ne[0]}),
        GGML_TYPE_I32);
    indices = core::ensure_backend_addressable_layout(ctx, indices);
    return modules::CodebookLookupModule({config.codebook_size, config.latent_size}).build(
        ctx,
        indices,
        codebook.embedding);
}

core::TensorValue project_quantized_latent(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & quantized_btd,
    const modules::Conv1dWeights & weights,
    const MimiCodecConfig & config) {
    auto bct = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, quantized_btd);
    return modules::Conv1dModule({
        config.latent_size,
        config.hidden_size,
        1,
        1,
        0,
        1,
        false,
    }).build(ctx, bct, weights);
}

core::TensorValue quantizer_decode(
    core::ModuleBuildContext & ctx,
    ggml_tensor * codes_t_q_b,
    const MimiCodecWeights & weights,
    const MimiCodecConfig & config) {
    auto semantic = codebook_decode(
        ctx,
        codes_t_q_b,
        weights.quantizer.semantic_codebooks.front(),
        config,
        0);
    semantic = project_quantized_latent(ctx, semantic, weights.quantizer.semantic_output_proj, config);

    core::TensorValue acoustic;
    for (int64_t codebook = 1; codebook < kMimiActiveCodebooks; ++codebook) {
        auto decoded = codebook_decode(
            ctx,
            codes_t_q_b,
            weights.quantizer.acoustic_codebooks[static_cast<size_t>(codebook - 1)],
            config,
            codebook);
        acoustic = acoustic.valid() ? modules::AddModule{}.build(ctx, acoustic, decoded) : decoded;
    }
    acoustic = project_quantized_latent(ctx, acoustic, weights.quantizer.acoustic_output_proj, config);
    return modules::AddModule{}.build(ctx, semantic, acoustic);
}

std::vector<int32_t> quantize_projected(
    const std::vector<float> & projected,
    int64_t frames,
    const std::vector<MimiCodecCodebookWeights> & codebooks,
    int64_t latent_size) {
    std::vector<float> residual(projected);
    std::vector<int32_t> codes(static_cast<size_t>(frames * static_cast<int64_t>(codebooks.size())));
    for (size_t q = 0; q < codebooks.size(); ++q) {
        const auto & table = codebooks[q].host_embedding;
        for (int64_t frame = 0; frame < frames; ++frame) {
            const size_t frame_offset = static_cast<size_t>(frame * latent_size);
            int32_t best = 0;
            float best_dist = std::numeric_limits<float>::infinity();
            for (int64_t code = 0; code < static_cast<int64_t>(table.size()) / latent_size; ++code) {
                const size_t code_offset = static_cast<size_t>(code * latent_size);
                float dist = 0.0F;
                for (int64_t dim = 0; dim < latent_size; ++dim) {
                    const float delta = residual[frame_offset + static_cast<size_t>(dim)] -
                        table[code_offset + static_cast<size_t>(dim)];
                    dist += delta * delta;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best = static_cast<int32_t>(code);
                }
            }
            codes[static_cast<size_t>(frame * static_cast<int64_t>(codebooks.size()) + static_cast<int64_t>(q))] = best;
            const size_t code_offset = static_cast<size_t>(best) * static_cast<size_t>(latent_size);
            for (int64_t dim = 0; dim < latent_size; ++dim) {
                residual[frame_offset + static_cast<size_t>(dim)] -= table[code_offset + static_cast<size_t>(dim)];
            }
        }
    }
    return codes;
}

void load_decoder_weights(
    MimiCodecDecoderWeights & weights,
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const MimiCodecConfig & config,
    const MimiCodecWeightBinding & weight_binding,
    assets::TensorStorageType storage_type) {
    weights.transformer_layers.reserve(static_cast<size_t>(config.transformer_layers));
    for (int64_t layer = 0; layer < config.transformer_layers; ++layer) {
        weights.transformer_layers.push_back(load_transformer_layer(
            store,
            source,
            weight_binding.decoder_transformer_layer_prefix + std::to_string(layer) + ".",
            storage_type,
            config.hidden_size));
    }
    auto upsample_weight = source.require_f32(
        weight_binding.decoder_upsample_weight,
        {config.hidden_size, 1, 2 * 2});
    for (int64_t channel = 0; channel < config.hidden_size; ++channel) {
        auto * kernel = upsample_weight.data() + static_cast<size_t>(channel) * 4;
        std::reverse(kernel, kernel + 4);
    }
    weights.upsample_conv = {
        store.make_from_f32(
            core::TensorShape::from_dims({config.hidden_size, 1, 1, 2 * 2}),
            storage_type,
            upsample_weight),
        std::nullopt,
    };
    weights.seanet.input_projection = binding::conv1d_from_named_source(
        store,
        source,
        weight_binding.decoder_input_projection_weight,
        weight_binding.decoder_input_projection_bias,
        storage_type);
    if (weight_binding.decoder_upsample_weight_names.size() != weight_binding.decoder_upsample_bias_names.size() ||
        weight_binding.decoder_upsample_weight_names.size() != weight_binding.decoder_residual_prefixes.size()) {
        throw std::runtime_error("Mimi codec decoder stage binding size mismatch");
    }
    weights.seanet.stages.reserve(weight_binding.decoder_upsample_weight_names.size());
    for (size_t stage = 0; stage < weight_binding.decoder_upsample_weight_names.size(); ++stage) {
        SEANetDecoderStageWeights stage_weights;
        stage_weights.upsample = binding::conv_transpose1d_from_named_source(
            store,
            source,
            weight_binding.decoder_upsample_weight_names[stage],
            weight_binding.decoder_upsample_bias_names[stage],
            storage_type);
        weights.seanet_upsample_bias_values.push_back(source.require_f32(
            weight_binding.decoder_upsample_bias_names[stage],
            {stage_weights.upsample.bias->shape.dims[0]}));
        stage_weights.residual_blocks.push_back(to_seanet_residual(load_residual_block(
            store,
            source,
            weight_binding.decoder_residual_prefixes[stage],
            storage_type)));
        weights.seanet.stages.push_back(std::move(stage_weights));
    }
    weights.seanet.output_projection = binding::conv1d_from_named_source(
        store,
        source,
        weight_binding.decoder_output_projection_weight,
        weight_binding.decoder_output_projection_bias,
        storage_type);
}

}  // namespace

std::shared_ptr<const MimiCodecWeights> load_mimi_codec_weights(
    const assets::TensorSource & source,
    const MimiCodecConfig & config,
    MimiCodecWeightBinding weight_binding,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t context_bytes,
    assets::TensorStorageType storage_type) {
    auto weights = std::make_shared<MimiCodecWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "MimiCodec.weights",
        context_bytes);
    load_encoder_weights(weights->encoder, *weights->store, source, config, weight_binding, storage_type);
    load_quantizer_weights(weights->quantizer, *weights->store, source, config, weight_binding, storage_type);
    load_decoder_weights(weights->decoder, *weights->store, source, config, weight_binding, storage_type);
    weights->store->upload();
    return weights;
}

MimiCodecComponent MimiCodecComponent::load_from_tensor_source(
    std::shared_ptr<const assets::TensorSource> source,
    MimiCodecConfig config,
    MimiCodecWeightBinding weight_binding,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t context_bytes,
    assets::TensorStorageType storage_type) {
    if (source == nullptr) {
        throw std::runtime_error("MimiCodecComponent requires a tensor source");
    }
    auto weights = load_mimi_codec_weights(
        *source,
        config,
        std::move(weight_binding),
        backend,
        backend_type,
        context_bytes,
        storage_type);
    return MimiCodecComponent(std::move(config), std::move(weights));
}

MimiCodecComponent::MimiCodecComponent(
    MimiCodecConfig config,
    std::shared_ptr<const MimiCodecWeights> weights)
    : config_(std::move(config)),
      weights_(std::move(weights)) {
    if (weights_ == nullptr) {
        throw std::runtime_error("MimiCodecComponent requires loaded weights");
    }
}

const MimiCodecConfig & MimiCodecComponent::config() const noexcept {
    return config_;
}

const std::shared_ptr<const MimiCodecWeights> & MimiCodecComponent::weights() const noexcept {
    return weights_;
}

struct MimiDecoderRuntime::Impl {
    struct QuantizerGraph {
        QuantizerGraph(
            std::shared_ptr<const MimiCodecWeights> weights,
            MimiCodecConfig config,
            ggml_backend_t backend,
            core::BackendType backend_type,
            int threads,
            size_t graph_arena_bytes,
            int64_t code_frames)
            : weights(std::move(weights)),
              config(std::move(config)),
              backend(backend),
              backend_type(backend_type),
              threads(std::max(1, threads)),
              code_frames(code_frames) {
            if (this->weights == nullptr) {
                throw std::runtime_error("Mimi codec quantizer decoder requires weights");
            }
            if (code_frames <= 0) {
                throw std::runtime_error("Mimi codec quantizer decoder requires positive frame count");
            }
            ctx.reset(ggml_init({graph_arena_bytes, nullptr, true}));
            if (ctx == nullptr) {
                throw std::runtime_error("failed to initialize Mimi codec quantizer decoder graph context");
            }
            core::ModuleBuildContext build_ctx{
                ctx.get(),
                "mimi_codec.quantizer_decode",
                backend_type,
            };
            codes = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I32, code_frames, kMimiActiveCodebooks, 1);
            ggml_set_input(codes);
            auto hidden = core::ensure_backend_addressable_layout(
                build_ctx,
                quantizer_decode(build_ctx, codes, *this->weights, this->config));
            output = hidden.tensor;
            ggml_set_output(output);

            graph = ggml_new_graph_custom(ctx.get(), 65536, false);
            ggml_build_forward_expand(graph, output);
            if (core::is_host_backend(backend)) {
                params_buffer = ggml_backend_alloc_ctx_tensors(ctx.get(), backend);
                if (params_buffer == nullptr) {
                    throw std::runtime_error("Mimi codec quantizer decoder context tensor allocation failed");
                }
            }
            galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (galloc == nullptr ||
                !ggml_gallocr_reserve(galloc, graph) ||
                !ggml_gallocr_alloc_graph(galloc, graph)) {
                throw std::runtime_error("failed to allocate Mimi codec quantizer decoder graph");
            }
        }

        ~QuantizerGraph() {
            engine::core::release_backend_graph_resources(backend, graph);
            if (galloc != nullptr) {
                ggml_gallocr_free(galloc);
            }
            if (params_buffer != nullptr) {
                ggml_backend_buffer_free(params_buffer);
            }
        }

        std::vector<float> run(const std::vector<int32_t> & input_codes) {
            if (static_cast<int64_t>(input_codes.size()) != code_frames * kMimiActiveCodebooks) {
                throw std::runtime_error("Mimi codec quantizer decoder code tensor size mismatch");
            }
            std::vector<int32_t> tensor_codes(static_cast<size_t>(code_frames * kMimiActiveCodebooks));
            for (int64_t frame = 0; frame < code_frames; ++frame) {
                for (int64_t codebook = 0; codebook < kMimiActiveCodebooks; ++codebook) {
                    tensor_codes[static_cast<size_t>(frame + code_frames * codebook)] =
                        input_codes[static_cast<size_t>(frame * kMimiActiveCodebooks + codebook)];
                }
            }
            ggml_backend_tensor_set(codes, tensor_codes.data(), 0, tensor_codes.size() * sizeof(int32_t));
            core::set_backend_threads(backend, threads);
            const ggml_status status = core::compute_backend_graph(backend, graph, nullptr, "mimi_codec.quantizer_decode");
            ggml_backend_synchronize(backend);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Mimi codec quantizer decoder graph compute failed");
            }
            return core::read_tensor_f32(output);
        }

        std::shared_ptr<const MimiCodecWeights> weights;
        MimiCodecConfig config;
        ggml_backend_t backend = nullptr;
        core::BackendType backend_type = core::BackendType::Cpu;
        int threads = 1;
        int64_t code_frames = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
        ggml_tensor * codes = nullptr;
        ggml_tensor * output = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_gallocr_t galloc = nullptr;
        ggml_backend_buffer_t params_buffer = nullptr;
    };

    struct RuntimeCache {
        std::unique_ptr<QuantizerGraph> quantizer_graph;
        int64_t quantizer_frames = -1;
        std::unique_ptr<MimiDecoderPreTransformerGraph> pre_transformer_graph;
        std::unique_ptr<MimiTransformerRuntime> transformer_runtime;
        int64_t transformer_frames = -1;
        std::unique_ptr<MimiDecoderPostTransformerGraph> post_transformer_graph;
        int64_t post_transformer_frames = -1;
    };

    runtime::AudioBuffer decode_with_state(
        const std::vector<int32_t> & codes,
        int64_t frames,
        MimiDecoderState & state,
        bool & transformer_initialized);

    std::shared_ptr<const MimiCodecWeights> weights;
    MimiCodecConfig config;
    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    int threads = 1;
    size_t graph_arena_bytes = 0;
    RuntimeCache cache;
    std::optional<MimiDecoderState> streaming_state;
    bool streaming_transformer_initialized = false;
};

MimiDecoderRuntime::MimiDecoderRuntime(
    std::shared_ptr<const MimiCodecWeights> weights,
    MimiCodecConfig config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    int threads,
    size_t graph_arena_bytes)
    : impl_(std::make_unique<Impl>()) {
    impl_->weights = std::move(weights);
    impl_->config = std::move(config);
    impl_->backend = backend;
    impl_->backend_type = backend_type;
    impl_->threads = std::max(1, threads);
    impl_->graph_arena_bytes = graph_arena_bytes;
}

MimiDecoderRuntime::~MimiDecoderRuntime() = default;

runtime::AudioBuffer MimiDecoderRuntime::Impl::decode_with_state(
    const std::vector<int32_t> & codes,
    int64_t frames,
    MimiDecoderState & state,
    bool & transformer_initialized) {
    if (frames <= 0 || static_cast<int64_t>(codes.size()) != frames * kMimiActiveCodebooks) {
        throw std::runtime_error("Mimi codec decode requires frames * 8 codes");
    }
    if (cache.quantizer_graph == nullptr || cache.quantizer_frames != frames) {
        cache.quantizer_graph = std::make_unique<Impl::QuantizerGraph>(
            weights,
            config,
            backend,
            backend_type,
            threads,
            graph_arena_bytes,
            frames);
        cache.quantizer_frames = frames;
    }

    const auto & config = this->config;
    auto latents_bct = cache.quantizer_graph->run(codes);
    std::vector<float> audio;
    audio.reserve(static_cast<size_t>(frames * kMimiFrameSamples));

    std::vector<float> latent_step(static_cast<size_t>(config.hidden_size), 0.0F);

    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t channel = 0; channel < config.hidden_size; ++channel) {
            latent_step[static_cast<size_t>(channel)] =
                latents_bct[static_cast<size_t>(channel * frames + frame)];
        }
        if (cache.pre_transformer_graph == nullptr) {
            cache.pre_transformer_graph = std::make_unique<MimiDecoderPreTransformerGraph>(
                backend,
                backend_type,
                threads,
                graph_arena_bytes,
                this->weights,
                config);
        }
        auto x = cache.pre_transformer_graph->run(latent_step, state);
        const int64_t encoder_frames = static_cast<int64_t>(x.size()) / config.hidden_size;
        if (cache.transformer_runtime == nullptr || cache.transformer_frames != encoder_frames) {
            cache.transformer_runtime = std::make_unique<MimiTransformerRuntime>(
                backend,
                backend_type,
                threads,
                graph_arena_bytes,
                weights->decoder.transformer_layers,
                config,
                encoder_frames);
            cache.transformer_frames = encoder_frames;
        }
        if (!transformer_initialized) {
            cache.transformer_runtime->reset_sequence(state.transformer.current_end);
            transformer_initialized = true;
        }
        x = cache.transformer_runtime->run(x);
        if (cache.post_transformer_graph == nullptr || cache.post_transformer_frames != encoder_frames) {
            cache.post_transformer_graph = std::make_unique<MimiDecoderPostTransformerGraph>(
                backend,
                backend_type,
                threads,
                graph_arena_bytes,
                this->weights,
                config,
                encoder_frames);
            cache.post_transformer_frames = encoder_frames;
        }
        x = cache.post_transformer_graph->run(x, state);
        audio.insert(audio.end(), x.begin(), x.end());
    }
    return runtime::AudioBuffer{config.sample_rate, static_cast<int>(config.channels), std::move(audio)};
}

runtime::AudioBuffer MimiDecoderRuntime::decode(const std::vector<int32_t> & codes, int64_t frames) {
    auto state = make_mimi_decoder_state(impl_->config);
    bool transformer_initialized = false;
    return impl_->decode_with_state(codes, frames, state, transformer_initialized);
}

void MimiDecoderRuntime::reset_streaming() {
    impl_->streaming_state = make_mimi_decoder_state(impl_->config);
    impl_->streaming_transformer_initialized = false;
}

runtime::AudioBuffer MimiDecoderRuntime::decode_streaming(const std::vector<int32_t> & codes, int64_t frames) {
    if (!impl_->streaming_state.has_value()) {
        reset_streaming();
    }
    return impl_->decode_with_state(
        codes,
        frames,
        *impl_->streaming_state,
        impl_->streaming_transformer_initialized);
}

struct MimiEncoderRuntime::Impl {
    struct RuntimeCache {
        std::unique_ptr<MimiEncoderPreTransformerGraph> pre_transformer_graph;
        std::unique_ptr<MimiTransformerRuntime> transformer_runtime;
        int64_t transformer_frames = -1;
        std::unique_ptr<MimiEncoderPostTransformerGraph> post_transformer_graph;
        int64_t post_transformer_frames = -1;
    };

    std::vector<int32_t> encode_streaming_with_state(
        const std::vector<float> & mono,
        MimiEncoderState & state,
        bool & transformer_initialized) {
        if (mono.empty() || static_cast<int64_t>(mono.size()) % kMimiFrameSamples != 0) {
            throw std::runtime_error("Mimi codec streaming encoder requires full Mimi frames");
        }
        const auto & config = this->config;
        const auto & weights = *this->weights;

        const int64_t code_frames = static_cast<int64_t>(mono.size()) / kMimiFrameSamples;
        std::vector<int32_t> out(static_cast<size_t>(code_frames * kMimiActiveCodebooks));
        std::vector<MimiCodecCodebookWeights> acoustic_codebooks;
        acoustic_codebooks.reserve(static_cast<size_t>(kMimiActiveCodebooks - 1));
        for (int64_t index = 0; index < kMimiActiveCodebooks - 1; ++index) {
            acoustic_codebooks.push_back(weights.quantizer.acoustic_codebooks[static_cast<size_t>(index)]);
        }

        for (int64_t frame = 0; frame < code_frames; ++frame) {
            std::vector<float> x(
                mono.begin() + static_cast<std::ptrdiff_t>(frame * kMimiFrameSamples),
                mono.begin() + static_cast<std::ptrdiff_t>((frame + 1) * kMimiFrameSamples));
            if (cache.pre_transformer_graph == nullptr) {
                cache.pre_transformer_graph = std::make_unique<MimiEncoderPreTransformerGraph>(
                    backend,
                    backend_type,
                    threads,
                    graph_arena_bytes,
                    this->weights,
                    config);
            }
            x = cache.pre_transformer_graph->run(x, state);

            const int64_t transformer_frames = static_cast<int64_t>(x.size()) / config.hidden_size;
            if (cache.transformer_runtime == nullptr || cache.transformer_frames != transformer_frames) {
                cache.transformer_runtime = std::make_unique<MimiTransformerRuntime>(
                    backend,
                    backend_type,
                    threads,
                    graph_arena_bytes,
                    weights.encoder.transformer_layers,
                    config,
                    transformer_frames);
                cache.transformer_frames = transformer_frames;
            }
            if (!transformer_initialized) {
                cache.transformer_runtime->reset_sequence(state.transformer.current_end);
                transformer_initialized = true;
            }
            x = cache.transformer_runtime->run(x);

            if (cache.post_transformer_graph == nullptr || cache.post_transformer_frames != transformer_frames) {
                cache.post_transformer_graph = std::make_unique<MimiEncoderPostTransformerGraph>(
                    backend,
                    backend_type,
                    threads,
                    graph_arena_bytes,
                    this->weights,
                    config,
                    transformer_frames);
                cache.post_transformer_frames = transformer_frames;
            }
            const auto [semantic, acoustic] = cache.post_transformer_graph->run(x, state);
            const auto semantic_codes = quantize_projected(
                semantic,
                1,
                weights.quantizer.semantic_codebooks,
                config.latent_size);
            const auto acoustic_codes = quantize_projected(
                acoustic,
                1,
                acoustic_codebooks,
                config.latent_size);
            out[static_cast<size_t>(frame * kMimiActiveCodebooks)] = semantic_codes.front();
            for (int64_t q = 1; q < kMimiActiveCodebooks; ++q) {
                out[static_cast<size_t>(frame * kMimiActiveCodebooks + q)] =
                    acoustic_codes[static_cast<size_t>(q - 1)];
            }
        }
        return out;
    }

    std::shared_ptr<const MimiCodecWeights> weights;
    MimiCodecConfig config;
    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    int threads = 1;
    size_t graph_arena_bytes = 0;
    RuntimeCache cache;
    std::optional<MimiEncoderState> streaming_state;
    bool streaming_transformer_initialized = false;
    std::vector<float> streaming_partial;
};

MimiEncoderRuntime::MimiEncoderRuntime(
    std::shared_ptr<const MimiCodecWeights> weights,
    MimiCodecConfig config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    int threads,
    size_t graph_arena_bytes)
    : impl_(std::make_unique<Impl>()) {
    impl_->weights = std::move(weights);
    impl_->config = std::move(config);
    impl_->backend = backend;
    impl_->backend_type = backend_type;
    impl_->threads = std::max(1, threads);
    impl_->graph_arena_bytes = graph_arena_bytes;
}

MimiEncoderRuntime::~MimiEncoderRuntime() = default;

std::vector<int32_t> MimiEncoderRuntime::encode(const runtime::AudioBuffer & audio) {
    auto mono = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        impl_->config.sample_rate);
    if (mono.empty()) {
        throw std::runtime_error("Mimi codec encoder received empty audio");
    }
    const int64_t padded_samples =
        ((static_cast<int64_t>(mono.size()) + kMimiFrameSamples - 1) / kMimiFrameSamples) * kMimiFrameSamples;
    mono.resize(static_cast<size_t>(padded_samples), 0.0F);
    auto state = make_mimi_encoder_state(impl_->config);
    bool transformer_initialized = false;
    return impl_->encode_streaming_with_state(mono, state, transformer_initialized);
}

void MimiEncoderRuntime::reset_streaming() {
    impl_->streaming_state = make_mimi_encoder_state(impl_->config);
    impl_->streaming_transformer_initialized = false;
    impl_->streaming_partial.clear();
}

std::vector<int32_t> MimiEncoderRuntime::encode_streaming(const runtime::AudioBuffer & audio, bool flush) {
    if (!impl_->streaming_state.has_value()) {
        reset_streaming();
    }
    if (audio.sample_rate <= 0 || audio.channels <= 0) {
        throw std::runtime_error("Mimi codec streaming encoder received invalid audio format");
    }
    auto mono = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        impl_->config.sample_rate);
    impl_->streaming_partial.insert(impl_->streaming_partial.end(), mono.begin(), mono.end());
    const int64_t available = static_cast<int64_t>(impl_->streaming_partial.size());
    int64_t encode_samples = (available / kMimiFrameSamples) * kMimiFrameSamples;
    if (flush && available > encode_samples) {
        encode_samples += kMimiFrameSamples;
    }
    if (encode_samples == 0) {
        return {};
    }
    std::vector<float> encode_input(
        impl_->streaming_partial.begin(),
        impl_->streaming_partial.begin() + static_cast<std::ptrdiff_t>(
            std::min<int64_t>(available, encode_samples)));
    encode_input.resize(static_cast<size_t>(encode_samples), 0.0F);
    impl_->streaming_partial.erase(
        impl_->streaming_partial.begin(),
        impl_->streaming_partial.begin() + static_cast<std::ptrdiff_t>(
            std::min<int64_t>(available, encode_samples)));
    return impl_->encode_streaming_with_state(
        encode_input,
        *impl_->streaming_state,
        impl_->streaming_transformer_initialized);
}

}  // namespace engine::codecs
