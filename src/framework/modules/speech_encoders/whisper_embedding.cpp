#include "engine/framework/modules/speech_encoders/whisper_embedding.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <cmath>
#include <stdexcept>

namespace engine::modules {
namespace {

core::TensorValue ensure_contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return core::ensure_backend_addressable_layout(ctx, value);
}

core::TensorValue permute(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    std::array<int, core::kMaxTensorRank> axes) {
    return TransposeModule({axes, input.shape.rank}).build(ctx, input);
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t head_dim) {
    return core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, input),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, head_dim}));
}

core::TensorValue whisper_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const WhisperAttentionWeights & weights,
    int64_t hidden_size,
    int64_t heads) {
    const int64_t head_dim = hidden_size / heads;
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    const LinearModule query({hidden_size, hidden_size, true});
    const LinearModule key({hidden_size, hidden_size, false});
    const LinearModule value({hidden_size, hidden_size, true});
    const LinearModule out({hidden_size, hidden_size, true});
    const MatMulModule matmul;

    auto q = reshape_heads(ctx, query.build(ctx, input, weights.query), heads, head_dim);
    auto k = reshape_heads(ctx, key.build(ctx, input, weights.key), heads, head_dim);
    auto v = reshape_heads(ctx, value.build(ctx, input, weights.value), heads, head_dim);

    auto q_heads = permute(ctx, q, {0, 2, 1, 3});
    auto k_heads = permute(ctx, k, {0, 2, 1, 3});
    auto v_heads = permute(ctx, v, {0, 2, 1, 3});
    auto scores = matmul.build(ctx, q_heads, permute(ctx, k_heads, {0, 1, 3, 2}));
    // Matches the active OpenAI Whisper SDPA path, which applies the head scale to q @ k^T.
    scores = core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, scale), scores.shape, GGML_TYPE_F32);
    auto attn = core::wrap_tensor(
        ggml_soft_max(ctx.ggml, ensure_contiguous(ctx, scores).tensor),
        scores.shape,
        GGML_TYPE_F32);
    auto context = matmul.build(ctx, attn, v_heads);
    context = permute(ctx, context, {0, 2, 1, 3});
    context = core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, context),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], hidden_size}));
    return out.build(ctx, context, weights.out);
}

core::TensorValue encoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const WhisperEncoderLayerWeights & weights,
    const WhisperEmbeddingConfig & config) {
    const LayerNormModule norm({config.n_audio_state, config.layer_norm_eps, true, true});
    const FeedForwardModule mlp({
        config.n_audio_state,
        config.n_audio_state * 4,
        true,
        GeluApproximation::ExactErf,
    });
    auto attn_input = norm.build(ctx, input, weights.attention_norm);
    auto attn_out = whisper_attention(ctx, attn_input, weights.attention, config.n_audio_state, config.n_audio_head);
    auto x = AddModule().build(ctx, input, attn_out);
    auto mlp_input = norm.build(ctx, x, weights.mlp_norm);
    auto mlp_out = mlp.build(ctx, mlp_input, weights.mlp);
    return AddModule().build(ctx, x, mlp_out);
}

void validate_config(const WhisperEmbeddingConfig & config) {
    if (config.n_mels <= 0 || config.n_audio_ctx <= 0 || config.n_audio_state <= 0 ||
        config.n_audio_head <= 0 || config.n_audio_layer <= 0) {
        throw std::runtime_error("WhisperEmbeddingConfig dimensions must be positive");
    }
    if (config.n_audio_state % config.n_audio_head != 0) {
        throw std::runtime_error("WhisperEmbeddingConfig n_audio_state must be divisible by n_audio_head");
    }
}

}  // namespace

WhisperEmbeddingModule::WhisperEmbeddingModule(WhisperEmbeddingConfig config) : config_(config) {
    validate_config(config_);
}

const WhisperEmbeddingConfig & WhisperEmbeddingModule::config() const noexcept {
    return config_;
}

core::TensorValue WhisperEmbeddingModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & log_mel,
    const WhisperEmbeddingWeights & weights) const {
    auto x = build_pre_norm(ctx, log_mel, weights);
    return LayerNormModule({config_.n_audio_state, config_.layer_norm_eps, true, true}).build(ctx, x, weights.final_norm);
}

core::TensorValue WhisperEmbeddingModule::build_pre_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & log_mel,
    const WhisperEmbeddingWeights & weights) const {
    if (log_mel.shape.rank != 3 ||
        log_mel.shape.dims[0] <= 0 ||
        log_mel.shape.dims[1] != config_.n_mels ||
        log_mel.shape.dims[2] <= 0 ||
        log_mel.shape.dims[2] > config_.n_audio_ctx * 2) {
        throw std::runtime_error(
            "Whisper log_mel must have shape [batch, n_mels, frames] with "
            "frames in the configured audio context");
    }
    core::validate_shape(
        weights.positional_embedding,
        core::TensorShape::from_dims({config_.n_audio_ctx, config_.n_audio_state}),
        "positional_embedding");
    if (static_cast<int64_t>(weights.layers.size()) != config_.n_audio_layer) {
        throw std::runtime_error("WhisperEmbeddingWeights layer count mismatch");
    }

    auto x = Conv1dModule({config_.n_mels, config_.n_audio_state, 3, 1, 1, 1, true})
                 .build(ctx, log_mel, weights.conv1);
    x = GeluModule({GeluApproximation::ExactErf}).build(ctx, x);
    x = Conv1dModule({config_.n_audio_state, config_.n_audio_state, 3, 2, 1, 1, true})
            .build(ctx, x, weights.conv2);
    x = GeluModule({GeluApproximation::ExactErf}).build(ctx, x);
    x = permute(ctx, x, {0, 2, 1});
    x = ensure_contiguous(ctx, x);

    const int64_t encoded_frames = x.shape.dims[1];
    auto position_rows = encoded_frames == config_.n_audio_ctx
        ? weights.positional_embedding
        : SliceModule({0, 0, encoded_frames}).build(
              ctx, weights.positional_embedding);
    auto pos = core::reshape_tensor(
        ctx,
        position_rows,
        core::TensorShape::from_dims(
            {1, encoded_frames, config_.n_audio_state}));
    if (log_mel.shape.dims[0] > 1) {
        pos = RepeatModule({core::TensorShape::from_dims(
                                {log_mel.shape.dims[0],
                                 encoded_frames,
                                 config_.n_audio_state})})
                  .build(ctx, pos);
    }
    x = AddModule().build(ctx, x, pos);

    for (const auto & layer : weights.layers) {
        x = encoder_layer(ctx, x, layer, config_);
    }
    return x;
}

}  // namespace engine::modules
