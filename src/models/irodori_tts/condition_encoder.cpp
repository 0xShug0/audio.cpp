#include "engine/models/irodori_tts/condition_encoder.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::irodori_tts {
namespace {

namespace binding = modules::binding;

constexpr size_t kConditionEncoderWeightContextBytes = 512ull * 1024ull * 1024ull;

int64_t text_ffn_dim(const IrodoriModelConfig & config) {
    return static_cast<int64_t>(static_cast<double>(config.text_dim) * static_cast<double>(config.text_mlp_ratio));
}

int64_t speaker_ffn_dim(const IrodoriModelConfig & config) {
    return static_cast<int64_t>(static_cast<double>(config.speaker_dim) * static_cast<double>(config.speaker_mlp_ratio));
}

int64_t caption_ffn_dim(const IrodoriModelConfig & config) {
    return static_cast<int64_t>(
        static_cast<double>(config.caption_dim_resolved()) *
        static_cast<double>(config.caption_mlp_ratio_resolved()));
}

modules::LinearWeights load_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_features,
    int64_t in_features,
    bool use_bias) {
    modules::LinearWeights weights;
    weights.weight = store.load_tensor(source, prefix + ".weight", storage_type, {out_features, in_features});
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    }
    return weights;
}

IrodoriSelfAttentionWeights load_self_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t hidden_size,
    int64_t heads) {
    IrodoriSelfAttentionWeights weights;
    const int64_t head_dim = hidden_size / heads;
    weights.wq = load_linear(store, source, prefix + ".wq", storage_type, hidden_size, hidden_size, false);
    weights.wk = load_linear(store, source, prefix + ".wk", storage_type, hidden_size, hidden_size, false);
    weights.wv = load_linear(store, source, prefix + ".wv", storage_type, hidden_size, hidden_size, false);
    weights.wo = load_linear(store, source, prefix + ".wo", storage_type, hidden_size, hidden_size, false);
    weights.gate = load_linear(store, source, prefix + ".gate", storage_type, hidden_size, hidden_size, false);
    weights.q_norm = store.load_f32_tensor(source, prefix + ".q_norm.weight", {heads, head_dim});
    weights.k_norm = store.load_f32_tensor(source, prefix + ".k_norm.weight", {heads, head_dim});
    return weights;
}

IrodoriTextBlockWeights load_text_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t hidden_size,
    int64_t heads,
    int64_t ffn_dim) {
    IrodoriTextBlockWeights weights;
    weights.attention_norm = store.load_f32_tensor(source, prefix + ".attention_norm.weight", {hidden_size});
    weights.attention = load_self_attention(store, source, prefix + ".attention", storage_type, hidden_size, heads);
    weights.mlp_norm = store.load_f32_tensor(source, prefix + ".mlp_norm.weight", {hidden_size});
    weights.mlp_w1 = load_linear(store, source, prefix + ".mlp.w1", storage_type, ffn_dim, hidden_size, false);
    weights.mlp_w2 = load_linear(store, source, prefix + ".mlp.w2", storage_type, hidden_size, ffn_dim, false);
    weights.mlp_w3 = load_linear(store, source, prefix + ".mlp.w3", storage_type, ffn_dim, hidden_size, false);
    return weights;
}

IrodoriDurationBlockWeights load_duration_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    const IrodoriModelConfig & config) {
    IrodoriDurationBlockWeights weights;
    weights.norm = store.load_f32_tensor(source, prefix + ".norm.weight", {config.duration_hidden_dim});
    weights.mlp_w1 = load_linear(
        store,
        source,
        prefix + ".mlp.w1",
        storage_type,
        config.duration_hidden_dim,
        config.duration_hidden_dim,
        false);
    weights.mlp_w2 = load_linear(
        store,
        source,
        prefix + ".mlp.w2",
        storage_type,
        config.duration_hidden_dim,
        config.duration_hidden_dim,
        false);
    weights.mlp_w3 = load_linear(
        store,
        source,
        prefix + ".mlp.w3",
        storage_type,
        config.duration_hidden_dim,
        config.duration_hidden_dim,
        false);
    weights.modulation = load_linear(
        store,
        source,
        prefix + ".modulation",
        storage_type,
        3 * config.duration_hidden_dim,
        config.speaker_dim,
        true);
    if (config.use_caption_condition) {
        weights.caption_modulation = load_linear(
            store,
            source,
            prefix + ".caption_modulation",
            storage_type,
            3 * config.duration_hidden_dim,
            config.caption_dim_resolved(),
            true);
    }
    return weights;
}

core::TensorValue ensure_contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    return core::ensure_backend_addressable_layout(ctx, input);
}

core::TensorValue repeat_like(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    const core::TensorValue & like) {
    return core::wrap_tensor(ggml_repeat(ctx.ggml, value.tensor, like.tensor), like.shape, GGML_TYPE_F32);
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t head_dim) {
    auto contiguous = ensure_contiguous(ctx, input);
    return core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, head_dim}));
}

core::TensorValue head_rms_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & weight,
    float eps) {
    core::validate_rank_between(input, 4, 4, "head_rms_norm.input");
    core::validate_shape(
        weight,
        core::TensorShape::from_dims({input.shape.dims[2], input.shape.dims[3]}),
        "head_rms_norm.weight");
    auto normalized = core::wrap_tensor(
        ggml_rms_norm(ctx.ggml, ensure_contiguous(ctx, input).tensor, eps),
        input.shape,
        GGML_TYPE_F32);
    auto weight_view = core::reshape_tensor(
        ctx,
        weight,
        core::TensorShape::from_dims({1, 1, input.shape.dims[2], input.shape.dims[3]}));
    return core::wrap_tensor(
        ggml_mul(ctx.ggml, normalized.tensor, repeat_like(ctx, weight_view, normalized).tensor),
        input.shape,
        GGML_TYPE_F32);
}

core::TensorValue attention_from_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t head_dim,
    const core::TensorValue & attention_mask) {
    const modules::MatMulModule matmul;
    auto scores = matmul.build(
        ctx,
        q_heads,
        modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    scores = core::wrap_tensor(
        ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(static_cast<float>(head_dim))),
        scores.shape,
        GGML_TYPE_F32);
    scores = modules::AddModule{}.build(ctx, scores, attention_mask);
    scores = ensure_contiguous(ctx, scores);
    auto attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, scores.tensor), scores.shape, GGML_TYPE_F32);
    return matmul.build(ctx, attn, v_heads);
}

core::TensorValue build_self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & attention_mask,
    const core::TensorValue & positions,
    const IrodoriSelfAttentionWeights & weights,
    int64_t hidden_size,
    int64_t heads,
    float norm_eps) {
    const int64_t head_dim = hidden_size / heads;
    auto q = modules::LinearModule(binding::linear_config(hidden_size, hidden_size, false)).build(ctx, input, weights.wq);
    auto k = modules::LinearModule(binding::linear_config(hidden_size, hidden_size, false)).build(ctx, input, weights.wk);
    auto v = modules::LinearModule(binding::linear_config(hidden_size, hidden_size, false)).build(ctx, input, weights.wv);
    const auto gate =
        modules::SigmoidModule{}.build(ctx, modules::LinearModule(binding::linear_config(hidden_size, hidden_size, false)).build(ctx, input, weights.gate));

    q = head_rms_norm(ctx, reshape_heads(ctx, q, heads, head_dim), weights.q_norm, norm_eps);
    k = head_rms_norm(ctx, reshape_heads(ctx, k, heads, head_dim), weights.k_norm, norm_eps);
    v = reshape_heads(ctx, v, heads, head_dim);
    q = modules::RoPEModule({head_dim, GGML_ROPE_TYPE_NORMAL, 10000.0F}).build(ctx, q, positions);
    k = modules::RoPEModule({head_dim, GGML_ROPE_TYPE_NORMAL, 10000.0F}).build(ctx, k, positions);

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context = attention_from_heads(ctx, q_heads, k_heads, v_heads, head_dim, attention_mask);
    context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = ensure_contiguous(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], hidden_size}));
    context = modules::MulModule{}.build(ctx, context, gate);
    return modules::LinearModule(binding::linear_config(hidden_size, hidden_size, false)).build(ctx, context, weights.wo);
}

core::TensorValue build_swiglu(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::LinearWeights & w1,
    const modules::LinearWeights & w2,
    const modules::LinearWeights & w3,
    int64_t hidden_size,
    int64_t ffn_dim) {
    auto gate = modules::LinearModule(binding::linear_config(hidden_size, ffn_dim, false)).build(ctx, input, w1);
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule(binding::linear_config(hidden_size, ffn_dim, false)).build(ctx, input, w3);
    auto gated = modules::MulModule{}.build(ctx, gate, up);
    return modules::LinearModule(binding::linear_config(ffn_dim, hidden_size, false)).build(ctx, gated, w2);
}

core::TensorValue build_text_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & mask,
    const core::TensorValue & attention_mask,
    const core::TensorValue & positions,
    const IrodoriTextBlockWeights & weights,
    int64_t hidden_size,
    int64_t heads,
    int64_t ffn_dim,
    float norm_eps) {
    auto attn_in = modules::RMSNormModule({hidden_size, norm_eps, true, false})
                       .build(ctx, input, {weights.attention_norm, std::nullopt});
    auto x = modules::AddModule{}.build(
        ctx,
        input,
        build_self_attention(ctx, attn_in, attention_mask, positions, weights.attention, hidden_size, heads, norm_eps));
    auto mlp_in = modules::RMSNormModule({hidden_size, norm_eps, true, false})
                      .build(ctx, x, {weights.mlp_norm, std::nullopt});
    x = modules::AddModule{}.build(
        ctx,
        x,
        build_swiglu(ctx, mlp_in, weights.mlp_w1, weights.mlp_w2, weights.mlp_w3, hidden_size, ffn_dim));
    return modules::MaskingModule{}.build(ctx, x, mask);
}

core::TensorValue select_speaker_vec(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & speaker_state,
    const core::TensorValue & has_speaker,
    const core::TensorValue & null_speaker) {
    auto speaker_first = modules::SliceModule({1, 0, 1}).build(ctx, speaker_state);
    speaker_first = core::reshape_tensor(
        ctx,
        speaker_first,
        core::TensorShape::from_dims({speaker_state.shape.dims[0], speaker_state.shape.dims[2]}));
    auto null_view = core::reshape_tensor(
        ctx,
        null_speaker,
        core::TensorShape::from_dims({1, null_speaker.shape.dims[0]}));
    auto null_batch = modules::RepeatModule({speaker_first.shape}).build(ctx, null_view);
    auto flag_f32 = core::wrap_tensor(ggml_cast(ctx.ggml, has_speaker.tensor, GGML_TYPE_F32), has_speaker.shape, GGML_TYPE_F32);
    auto flag = core::reshape_tensor(ctx, flag_f32, core::TensorShape::from_dims({has_speaker.shape.dims[0], 1}));
    auto flag_full = repeat_like(ctx, flag, speaker_first);
    auto inv_flag = core::wrap_tensor(ggml_scale_bias(ctx.ggml, flag_full.tensor, -1.0F, 1.0F), flag_full.shape, GGML_TYPE_F32);
    return modules::AddModule{}.build(
        ctx,
        modules::MulModule{}.build(ctx, speaker_first, flag_full),
        modules::MulModule{}.build(ctx, null_batch, inv_flag));
}

core::TensorValue select_caption_vec(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & caption_state,
    const core::TensorValue & caption_mask,
    const core::TensorValue & has_caption,
    const core::TensorValue & null_caption) {
    const int64_t batch = caption_state.shape.dims[0];
    const int64_t tokens = caption_state.shape.dims[1];
    const int64_t dim = caption_state.shape.dims[2];
    auto mask_f32 = core::wrap_tensor(
        ggml_cast(ctx.ggml, caption_mask.tensor, GGML_TYPE_F32),
        caption_mask.shape,
        GGML_TYPE_F32);
    auto has_f32 = core::wrap_tensor(
        ggml_cast(ctx.ggml, has_caption.tensor, GGML_TYPE_F32),
        has_caption.shape,
        GGML_TYPE_F32);
    auto has_seq = core::reshape_tensor(ctx, has_f32, core::TensorShape::from_dims({batch, 1}));
    has_seq = modules::RepeatModule({core::TensorShape::from_dims({batch, tokens})}).build(ctx, has_seq);
    mask_f32 = modules::MulModule{}.build(ctx, mask_f32, has_seq);
    auto mask_3d = core::reshape_tensor(ctx, mask_f32, core::TensorShape::from_dims({batch, tokens, 1}));
    auto mask_full = repeat_like(ctx, mask_3d, caption_state);
    auto masked_state = modules::MulModule{}.build(ctx, caption_state, mask_full);
    auto summed = modules::ReduceSumModule({1}).build(ctx, masked_state);
    auto denom = modules::ReduceSumModule({1}).build(ctx, mask_3d);
    denom = core::wrap_tensor(
        ggml_clamp(ctx.ggml, denom.tensor, 1.0F, std::numeric_limits<float>::infinity()),
        denom.shape,
        GGML_TYPE_F32);
    auto pooled = core::wrap_tensor(
        ggml_div(ctx.ggml, summed.tensor, repeat_like(ctx, denom, summed).tensor),
        summed.shape,
        GGML_TYPE_F32);
    pooled = core::reshape_tensor(ctx, pooled, core::TensorShape::from_dims({batch, dim}));
    auto null_view = core::reshape_tensor(ctx, null_caption, core::TensorShape::from_dims({1, dim}));
    auto null_batch = modules::RepeatModule({pooled.shape}).build(ctx, null_view);
    auto flag = core::reshape_tensor(ctx, has_f32, core::TensorShape::from_dims({batch, 1}));
    auto flag_full = repeat_like(ctx, flag, pooled);
    auto inv_flag = core::wrap_tensor(ggml_scale_bias(ctx.ggml, flag_full.tensor, -1.0F, 1.0F), flag_full.shape, GGML_TYPE_F32);
    return modules::AddModule{}.build(
        ctx,
        modules::MulModule{}.build(ctx, pooled, flag_full),
        modules::MulModule{}.build(ctx, null_batch, inv_flag));
}

core::TensorValue softplus(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input) {
    auto one = core::wrap_tensor(ggml_scale_bias(ctx.ggml, ensure_contiguous(ctx, input).tensor, 0.0F, 1.0F), input.shape, GGML_TYPE_F32);
    auto exp_value = core::wrap_tensor(ggml_exp(ctx.ggml, input.tensor), input.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_log(ctx.ggml, modules::AddModule{}.build(ctx, exp_value, one).tensor), input.shape, GGML_TYPE_F32);
}

core::TensorValue masked_token_sum(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & values,
    const core::TensorValue & mask) {
    core::validate_shape(mask, values.shape, "duration_token_mask");
    auto mask_f32 = core::wrap_tensor(ggml_cast(ctx.ggml, mask.tensor, GGML_TYPE_F32), mask.shape, GGML_TYPE_F32);
    auto masked = modules::MulModule{}.build(ctx, values, mask_f32);
    auto sum = core::wrap_tensor(
        ggml_sum_rows(ctx.ggml, ensure_contiguous(ctx, masked).tensor),
        core::TensorShape::from_dims({values.shape.dims[0], 1}),
        GGML_TYPE_F32);
    return core::reshape_tensor(ctx, sum, core::TensorShape::from_dims({values.shape.dims[0]}));
}

core::TensorValue log1p_tensor(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input) {
    auto one = core::wrap_tensor(ggml_scale_bias(ctx.ggml, ensure_contiguous(ctx, input).tensor, 0.0F, 1.0F), input.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_log(ctx.ggml, modules::AddModule{}.build(ctx, input, one).tensor), input.shape, GGML_TYPE_F32);
}

}  // namespace

IrodoriConditionEncoderWeights load_irodori_condition_encoder_weights(
    const IrodoriAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type) {
    const auto & config = assets.config;
    const auto & source = *assets.model_weights;
    IrodoriConditionEncoderWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "irodori_tts.condition_encoder.weights",
        weight_context_bytes == 0 ? kConditionEncoderWeightContextBytes : weight_context_bytes);
    weights.text_embedding = weights.store->load_tensor(
        source,
        "text_encoder.text_embedding.weight",
        weight_storage_type,
        {config.text_vocab_size, config.text_dim});
    weights.text_blocks.reserve(static_cast<size_t>(config.text_layers));
    for (int64_t layer = 0; layer < config.text_layers; ++layer) {
        weights.text_blocks.push_back(load_text_block(
            *weights.store,
            source,
            "text_encoder.blocks." + std::to_string(layer),
            weight_storage_type,
            config.text_dim,
            config.text_heads,
            text_ffn_dim(config)));
    }
    weights.text_norm = weights.store->load_f32_tensor(source, "text_norm.weight", {config.text_dim});

    weights.speaker_in_proj = load_linear(
        *weights.store,
        source,
        "speaker_encoder.in_proj",
        weight_storage_type,
        config.speaker_dim,
        config.speaker_patched_latent_dim(),
        true);
    weights.speaker_blocks.reserve(static_cast<size_t>(config.speaker_layers));
    for (int64_t layer = 0; layer < config.speaker_layers; ++layer) {
        weights.speaker_blocks.push_back(load_text_block(
            *weights.store,
            source,
            "speaker_encoder.blocks." + std::to_string(layer),
            weight_storage_type,
            config.speaker_dim,
            config.speaker_heads,
            speaker_ffn_dim(config)));
    }
    weights.speaker_norm = weights.store->load_f32_tensor(source, "speaker_norm.weight", {config.speaker_dim});

    weights.duration.null_speaker = weights.store->load_f32_tensor(source, "duration_predictor.null_speaker", {config.speaker_dim});
    if (config.use_caption_condition) {
        weights.caption_embedding = weights.store->load_tensor(
            source,
            "caption_encoder.text_embedding.weight",
            weight_storage_type,
            {config.caption_vocab_size_resolved(), config.caption_dim_resolved()});
        weights.caption_blocks.reserve(static_cast<size_t>(config.caption_layers_resolved()));
        for (int64_t layer = 0; layer < config.caption_layers_resolved(); ++layer) {
            weights.caption_blocks.push_back(load_text_block(
                *weights.store,
                source,
                "caption_encoder.blocks." + std::to_string(layer),
                weight_storage_type,
                config.caption_dim_resolved(),
                config.caption_heads_resolved(),
                caption_ffn_dim(config)));
        }
        weights.caption_norm =
            weights.store->load_f32_tensor(source, "caption_norm.weight", {config.caption_dim_resolved()});
        weights.duration.null_caption =
            weights.store->load_f32_tensor(source, "duration_predictor.null_caption", {config.caption_dim_resolved()});
    }
    weights.duration.token_input_proj = load_linear(
        *weights.store,
        source,
        "duration_predictor.token_input_proj",
        weight_storage_type,
        config.duration_hidden_dim,
        config.text_dim,
        true);
    weights.duration.token_blocks.reserve(static_cast<size_t>(config.duration_layers));
    for (int64_t layer = 0; layer < config.duration_layers; ++layer) {
        weights.duration.token_blocks.push_back(load_duration_block(
            *weights.store,
            source,
            "duration_predictor.token_blocks." + std::to_string(layer),
            weight_storage_type,
            config));
    }
    weights.duration.token_out_norm =
        weights.store->load_f32_tensor(source, "duration_predictor.token_out_norm.weight", {config.duration_hidden_dim});
    weights.duration.token_out_proj = load_linear(
        *weights.store,
        source,
        "duration_predictor.token_out_proj",
        weight_storage_type,
        1,
        config.duration_hidden_dim,
        true);
    weights.store->upload();
    return weights;
}

core::TensorValue build_irodori_text_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_ids,
    const core::TensorValue & text_mask,
    const core::TensorValue & attention_mask,
    const core::TensorValue & positions,
    const IrodoriConditionEncoderWeights & weights,
    const IrodoriModelConfig & config) {
    auto hidden = modules::EmbeddingModule({config.text_vocab_size, config.text_dim}).build(ctx, input_ids, weights.text_embedding);
    hidden = modules::MaskingModule{}.build(ctx, hidden, text_mask);
    for (const auto & block : weights.text_blocks) {
        hidden = build_text_block(
            ctx,
            hidden,
            text_mask,
            attention_mask,
            positions,
            block,
            config.text_dim,
            config.text_heads,
            text_ffn_dim(config),
            config.norm_eps);
    }
    hidden = modules::RMSNormModule({config.text_dim, config.norm_eps, true, false})
                 .build(ctx, hidden, {weights.text_norm, std::nullopt});
    return modules::MaskingModule{}.build(ctx, hidden, text_mask);
}

core::TensorValue build_irodori_reference_latent_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & ref_latent,
    const core::TensorValue & ref_mask,
    const core::TensorValue & attention_mask,
    const core::TensorValue & positions,
    const IrodoriConditionEncoderWeights & weights,
    const IrodoriModelConfig & config) {
    auto hidden = modules::LinearModule(binding::linear_config(
                                            config.speaker_patched_latent_dim(),
                                            config.speaker_dim,
                                            true))
                      .build(ctx, ref_latent, weights.speaker_in_proj);
    hidden = core::wrap_tensor(ggml_scale(ctx.ggml, hidden.tensor, 1.0F / 6.0F), hidden.shape, GGML_TYPE_F32);
    hidden = modules::MaskingModule{}.build(ctx, hidden, ref_mask);
    for (const auto & block : weights.speaker_blocks) {
        hidden = build_text_block(
            ctx,
            hidden,
            ref_mask,
            attention_mask,
            positions,
            block,
            config.speaker_dim,
            config.speaker_heads,
            speaker_ffn_dim(config),
            config.norm_eps);
    }
    hidden = modules::RMSNormModule({config.speaker_dim, config.norm_eps, true, false})
                 .build(ctx, hidden, {weights.speaker_norm, std::nullopt});
    return modules::MaskingModule{}.build(ctx, hidden, ref_mask);
}

core::TensorValue build_irodori_caption_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_ids,
    const core::TensorValue & caption_mask,
    const core::TensorValue & attention_mask,
    const core::TensorValue & positions,
    const IrodoriConditionEncoderWeights & weights,
    const IrodoriModelConfig & config) {
    auto hidden = modules::EmbeddingModule({config.caption_vocab_size_resolved(), config.caption_dim_resolved()})
                      .build(ctx, input_ids, weights.caption_embedding);
    hidden = modules::MaskingModule{}.build(ctx, hidden, caption_mask);
    for (const auto & block : weights.caption_blocks) {
        hidden = build_text_block(
            ctx,
            hidden,
            caption_mask,
            attention_mask,
            positions,
            block,
            config.caption_dim_resolved(),
            config.caption_heads_resolved(),
            caption_ffn_dim(config),
            config.norm_eps);
    }
    hidden = modules::RMSNormModule({config.caption_dim_resolved(), config.norm_eps, true, false})
                 .build(ctx, hidden, {weights.caption_norm, std::nullopt});
    return modules::MaskingModule{}.build(ctx, hidden, caption_mask);
}

core::TensorValue build_irodori_duration_predictor(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & text_state,
    const core::TensorValue & text_mask,
    const core::TensorValue & speaker_state,
    const core::TensorValue & has_speaker,
    const core::TensorValue & caption_state,
    const core::TensorValue & caption_mask,
    const core::TensorValue & has_caption,
    const IrodoriConditionEncoderWeights & weights,
    const IrodoriModelConfig & config) {
    auto speaker_vec = select_speaker_vec(ctx, speaker_state, has_speaker, weights.duration.null_speaker);
    core::TensorValue caption_vec;
    if (config.use_caption_condition) {
        caption_vec = select_caption_vec(
            ctx,
            caption_state,
            caption_mask,
            has_caption,
            weights.duration.null_caption);
    }
    auto hidden = modules::LinearModule(binding::linear_config(config.text_dim, config.duration_hidden_dim, true))
                      .build(ctx, text_state, weights.duration.token_input_proj);
    for (const auto & block : weights.duration.token_blocks) {
        auto normed = modules::RMSNormModule({config.duration_hidden_dim, config.norm_eps, true, false})
                          .build(ctx, hidden, {block.norm, std::nullopt});
        auto modulation = modules::LinearModule(binding::linear_config(config.speaker_dim, 3 * config.duration_hidden_dim, true))
                              .build(ctx, modules::SiluModule{}.build(ctx, speaker_vec), block.modulation);
        auto shift = modules::SliceModule({1, 0, config.duration_hidden_dim}).build(ctx, modulation);
        auto scale = modules::SliceModule({1, config.duration_hidden_dim, config.duration_hidden_dim}).build(ctx, modulation);
        auto gate = modules::SliceModule({1, 2 * config.duration_hidden_dim, config.duration_hidden_dim}).build(ctx, modulation);
        if (config.use_caption_condition) {
            auto caption_modulation =
                modules::LinearModule(binding::linear_config(config.caption_dim_resolved(), 3 * config.duration_hidden_dim, true))
                    .build(ctx, modules::SiluModule{}.build(ctx, caption_vec), block.caption_modulation);
            shift = modules::AddModule{}.build(
                ctx,
                shift,
                modules::SliceModule({1, 0, config.duration_hidden_dim}).build(ctx, caption_modulation));
            scale = modules::AddModule{}.build(
                ctx,
                scale,
                modules::SliceModule({1, config.duration_hidden_dim, config.duration_hidden_dim}).build(ctx, caption_modulation));
            gate = modules::AddModule{}.build(
                ctx,
                gate,
                modules::SliceModule({1, 2 * config.duration_hidden_dim, config.duration_hidden_dim}).build(ctx, caption_modulation));
        }
        auto shift_seq = core::reshape_tensor(ctx, shift, core::TensorShape::from_dims({text_state.shape.dims[0], 1, config.duration_hidden_dim}));
        auto scale_seq = core::reshape_tensor(ctx, scale, core::TensorShape::from_dims({text_state.shape.dims[0], 1, config.duration_hidden_dim}));
        auto gate_seq = core::reshape_tensor(ctx, gate, core::TensorShape::from_dims({text_state.shape.dims[0], 1, config.duration_hidden_dim}));
        shift_seq = repeat_like(ctx, shift_seq, normed);
        scale_seq = repeat_like(ctx, scale_seq, normed);
        gate_seq = repeat_like(ctx, gate_seq, normed);
        normed = modules::AddModule{}.build(
            ctx,
            modules::MulModule{}.build(
                ctx,
                normed,
                core::wrap_tensor(ggml_scale_bias(ctx.ggml, scale_seq.tensor, 1.0F, 1.0F), scale_seq.shape, GGML_TYPE_F32)),
            shift_seq);
        auto mlp = build_swiglu(
            ctx,
            normed,
            block.mlp_w1,
            block.mlp_w2,
            block.mlp_w3,
            config.duration_hidden_dim,
            config.duration_hidden_dim);
        hidden = modules::AddModule{}.build(
            ctx,
            hidden,
            modules::MulModule{}.build(ctx, modules::TanhModule{}.build(ctx, gate_seq), mlp));
    }
    hidden = modules::RMSNormModule({config.duration_hidden_dim, config.norm_eps, true, false})
                 .build(ctx, hidden, {weights.duration.token_out_norm, std::nullopt});
    auto logits = modules::LinearModule(binding::linear_config(config.duration_hidden_dim, 1, true))
                      .build(ctx, hidden, weights.duration.token_out_proj);
    logits = core::reshape_tensor(ctx, logits, core::TensorShape::from_dims({text_state.shape.dims[0], text_state.shape.dims[1]}));
    auto token_frames = softplus(ctx, logits);
    return log1p_tensor(ctx, masked_token_sum(ctx, token_frames, text_mask));
}

}  // namespace engine::models::irodori_tts
