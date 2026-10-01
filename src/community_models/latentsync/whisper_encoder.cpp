#include "engine/community_models/latentsync/whisper_encoder.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <stdexcept>

namespace engine::models::latentsync {

modules::WhisperEmbeddingWeights load_whisper_encoder_weights(
    core::BackendWeightStore & store, const assets::TensorSource & source,
    const modules::WhisperEmbeddingConfig & config, assets::TensorStorageType type) {
    namespace binding = modules::binding;
    modules::WhisperEmbeddingWeights weights;
    const int64_t channels = config.n_audio_state;
    weights.conv1 = binding::conv1d_from_source(
        store, source, "whisper.encoder.conv1", type, channels, config.n_mels, 3, true);
    weights.conv2 = binding::conv1d_from_source(
        store, source, "whisper.encoder.conv2", type, channels, channels, 3, true);
    weights.positional_embedding = store.load_f32_tensor(source,
        "whisper.encoder.positional_embedding", {config.n_audio_ctx, channels});
    for (int64_t index = 0; index < config.n_audio_layer; ++index) {
        const auto prefix = "whisper.encoder.blocks." + std::to_string(index);
        modules::WhisperEncoderLayerWeights layer;
        layer.attention_norm = binding::norm_from_source(store, source, prefix + ".attn_ln", channels);
        layer.mlp_norm = binding::norm_from_source(store, source, prefix + ".mlp_ln", channels);
        layer.attention.query = binding::linear_from_source(store, source, prefix + ".attn.query", type, channels, channels, true);
        layer.attention.key = binding::linear_from_source(store, source, prefix + ".attn.key", type, channels, channels, false);
        layer.attention.value = binding::linear_from_source(store, source, prefix + ".attn.value", type, channels, channels, true);
        layer.attention.out = binding::linear_from_source(store, source, prefix + ".attn.out", type, channels, channels, true);
        const auto fc1 = binding::linear_from_source(store, source, prefix + ".mlp.0", type, channels * 4, channels, true);
        const auto fc2 = binding::linear_from_source(store, source, prefix + ".mlp.2", type, channels, channels * 4, true);
        layer.mlp = {fc1.weight, fc1.bias, fc2.weight, fc2.bias};
        weights.layers.push_back(std::move(layer));
    }
    return weights;
}

LatentSyncWhisperEncoder::LatentSyncWhisperEncoder(modules::WhisperEmbeddingConfig config)
    : config_(config) {
    if (config_.n_audio_state <= 0 || config_.n_audio_head <= 0 ||
        config_.n_audio_state % config_.n_audio_head != 0 || config_.n_audio_ctx <= 0 ||
        config_.n_mels <= 0 || config_.n_audio_layer <= 0) {
        throw std::runtime_error("Invalid LatentSync Whisper encoder dimensions");
    }
}

core::TensorValue LatentSyncWhisperEncoder::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & mel,
    const modules::WhisperEmbeddingWeights & weights) const {
    using namespace modules;
    core::validate_shape(mel,
        core::TensorShape::from_dims({1, config_.n_mels, 2 * config_.n_audio_ctx}), "mel");
    if (weights.layers.size() != static_cast<size_t>(config_.n_audio_layer)) {
        throw std::runtime_error("LatentSync Whisper layer count does not match weights");
    }
    const int64_t channels = config_.n_audio_state;
    const int64_t frames = config_.n_audio_ctx;
    const int64_t heads = config_.n_audio_head;
    const int64_t head_dim = channels / heads;
    const GeluModule gelu({GeluApproximation::ExactErf});
    auto x = Conv1dModule({config_.n_mels, channels, 3, 1, 1, 1, true})
                 .build(ctx, mel, weights.conv1);
    x = gelu.build(ctx, x);
    x = Conv1dModule({channels, channels, 3, 2, 1, 1, true}).build(ctx, x, weights.conv2);
    x = gelu.build(ctx, x);
    x = TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
    x = core::ensure_backend_addressable_layout(ctx, x);
    const auto positions = core::reshape_tensor(ctx, weights.positional_embedding,
        core::TensorShape::from_dims({1, frames, channels}));
    x = AddModule().build(ctx, x, positions);

    auto embeddings = core::reshape_tensor(ctx, x,
        core::TensorShape::from_dims({1, frames, 1, channels}));
    const LayerNormModule norm({channels, config_.layer_norm_eps, true, true});
    const FeedForwardModule mlp({channels, channels * 4, true, GeluApproximation::ExactErf});
    const ScaledDotProductAttentionModule attention({
        head_dim, ScaledDotProductAttentionLowering::Flash, GGML_PREC_F32});
    for (const auto & layer : weights.layers) {
        const auto normalized = norm.build(ctx, x, layer.attention_norm);
        const auto project_heads = [&](const LinearWeights & projection, bool bias) {
            auto projected = LinearModule({channels, channels, bias}).build(ctx, normalized, projection);
            projected = core::reshape_tensor(ctx, projected,
                core::TensorShape::from_dims({1, frames, heads, head_dim}));
            return TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, projected);
        };
        const auto q = project_heads(layer.attention.query, true);
        const auto k = project_heads(layer.attention.key, false);
        const auto v = project_heads(layer.attention.value, true);
        auto attended = attention.build(ctx, q, k, v);
        attended = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, attended),
            core::TensorShape::from_dims({1, frames, channels}));
        attended = LinearModule({channels, channels, true}).build(ctx, attended, layer.attention.out);
        x = AddModule().build(ctx, x, attended);
        x = AddModule().build(ctx, x, mlp.build(ctx, norm.build(ctx, x, layer.mlp_norm), layer.mlp));
        const auto layer_embedding = core::reshape_tensor(ctx, x,
            core::TensorShape::from_dims({1, frames, 1, channels}));
        embeddings = ConcatModule({2}).build(ctx, embeddings, layer_embedding);
    }
    return embeddings;
}

}  // namespace engine::models::latentsync
