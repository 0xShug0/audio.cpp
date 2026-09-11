#include "engine/models/stable_audio/rf_dit.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/sampling/torch_random.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace engine::models::stable_audio {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kRfWeightContextBytesSmall = 1800ull * 1024ull * 1024ull;
constexpr size_t kRfWeightContextBytesMedium = 3600ull * 1024ull * 1024ull;
constexpr int64_t kTimestepFeaturesDim = 256;
constexpr float kFourierMinFreq = 0.5F;
constexpr float kFourierMaxFreq = 10000.0F;
constexpr float kPi = 3.14159265358979323846F;
constexpr float kTransformerNormEps = 1.0e-5F;
constexpr float kQkNormEps = 1.0e-6F;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

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

modules::LinearWeights load_linear_as_shape(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & tensor_name,
    assets::TensorStorageType storage_type,
    int64_t out_features,
    int64_t in_features,
    bool use_bias) {
    modules::LinearWeights weights;
    weights.weight = store.load_tensor_as_shape(
        source,
        tensor_name + ".weight",
        storage_type,
        {out_features, in_features, 1},
        core::TensorShape::from_dims({out_features, in_features}));
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, tensor_name + ".bias", {out_features});
    }
    return weights;
}

int64_t head_dim(const StableAudioConfig & config) {
    if (config.num_heads <= 0 || config.embed_dim % config.num_heads != 0) {
        throw std::runtime_error("Stable Audio RF DiT head dimensions are invalid");
    }
    return config.embed_dim / config.num_heads;
}

int64_t ff_inner_dim(const StableAudioConfig & config) {
    return config.embed_dim * 4;
}

core::TensorValue ensure_contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    return core::ensure_backend_addressable_layout(ctx, input);
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    return core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, input),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

core::TensorValue rms_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & gamma,
    float eps) {
    return modules::RMSNormModule({input.shape.last_dim(), eps, true, false})
        .build(ctx, input, {gamma, std::nullopt});
}

core::TensorValue one_plus(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    return core::wrap_tensor(ggml_scale_bias(ctx.ggml, input.tensor, 1.0F, 1.0F), input.shape, GGML_TYPE_F32);
}

core::TensorValue sigmoid_one_minus(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    auto shifted = core::wrap_tensor(ggml_scale_bias(ctx.ggml, input.tensor, -1.0F, 1.0F), input.shape, GGML_TYPE_F32);
    return modules::SigmoidModule{}.build(ctx, shifted);
}

core::TensorValue scale_bias_tensor(core::ModuleBuildContext & ctx, const core::TensorValue & input, float scale, float bias) {
    return core::wrap_tensor(ggml_scale_bias(ctx.ggml, ensure_contiguous(ctx, input).tensor, scale, bias), input.shape, GGML_TYPE_F32);
}

core::TensorValue sub_tensor(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    return core::wrap_tensor(
        ggml_sub(ctx.ggml, ensure_contiguous(ctx, lhs).tensor, ensure_contiguous(ctx, rhs).tensor),
        lhs.shape,
        GGML_TYPE_F32);
}

core::TensorValue div_tensor(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    return core::wrap_tensor(
        ggml_div(ctx.ggml, ensure_contiguous(ctx, lhs).tensor, ensure_contiguous(ctx, rhs).tensor),
        lhs.shape,
        GGML_TYPE_F32);
}

core::TensorValue repeat_like(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & like) {
    return core::wrap_tensor(ggml_repeat(ctx.ggml, input.tensor, like.tensor), like.shape, GGML_TYPE_F32);
}

std::vector<float> make_fourier_freqs() {
    const int64_t half = kTimestepFeaturesDim / 2;
    std::vector<float> freqs(static_cast<size_t>(half));
    for (int64_t i = 0; i < half; ++i) {
        const float ramp = static_cast<float>(i) / static_cast<float>(half - 1);
        freqs[static_cast<size_t>(i)] =
            std::exp(ramp * (std::log(kFourierMaxFreq) - std::log(kFourierMinFreq)) + std::log(kFourierMinFreq)) *
            2.0F * kPi;
    }
    return freqs;
}

core::TensorValue build_timestep_embedding(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & features,
    const StableAudioRfDitWeights & weights,
    const StableAudioConfig & config) {
    auto hidden = features;
    hidden = modules::LinearModule({kTimestepFeaturesDim, config.embed_dim, true, GGML_PREC_F32})
        .build(ctx, hidden, weights.timestep_embed_0);
    hidden = modules::SiluModule{}.build(ctx, hidden);
    hidden = modules::LinearModule({config.embed_dim, config.embed_dim, true, GGML_PREC_F32})
        .build(ctx, hidden, weights.timestep_embed_2);
    return hidden;
}

core::TensorValue rf_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim) {
    const auto q = ensure_contiguous(ctx, q_heads);
    const auto k = ensure_contiguous(ctx, k_heads);
    const auto v = ensure_contiguous(ctx, v_heads);
    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q.tensor,
        k.tensor,
        v.tensor,
        nullptr,
        1.0F / std::sqrt(static_cast<float>(dim)),
        0.0F,
        0.0F);
    ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
    return core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[2], q.shape.dims[1], dim}),
        GGML_TYPE_F32);
}

core::TensorValue apply_padding_to_v(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & v_heads,
    const core::TensorValue & padding_mask) {
    const auto mask = modules::RepeatModule({v_heads.shape}).build(ctx, padding_mask);
    return modules::MulModule{}.build(ctx, v_heads, mask);
}

core::TensorValue self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue & padding_mask,
    const StableAudioRfAttentionWeights & weights,
    const StableAudioConfig & config) {
    const int64_t dim = head_dim(config);
    auto qkv = modules::LinearModule({config.embed_dim, config.embed_dim * (weights.differential ? 5 : 3), false, GGML_PREC_F32})
        .build(ctx, input, weights.to_qkv);
    auto q = modules::SliceModule({2, 0, config.embed_dim}).build(ctx, qkv);
    auto k = modules::SliceModule({2, config.embed_dim, config.embed_dim}).build(ctx, qkv);
    auto v = modules::SliceModule({2, 2 * config.embed_dim, config.embed_dim}).build(ctx, qkv);
    core::TensorValue q_diff;
    core::TensorValue k_diff;
    if (weights.differential) {
        q_diff = modules::SliceModule({2, 3 * config.embed_dim, config.embed_dim}).build(ctx, qkv);
        k_diff = modules::SliceModule({2, 4 * config.embed_dim, config.embed_dim}).build(ctx, qkv);
    }
    q = rms_norm(ctx, reshape_heads(ctx, q, config.num_heads, dim), weights.q_norm, kQkNormEps);
    k = rms_norm(ctx, reshape_heads(ctx, k, config.num_heads, dim), weights.k_norm, kQkNormEps);
    v = reshape_heads(ctx, v, config.num_heads, dim);
    q = modules::RoPEModule({std::max<int64_t>(dim / 2, 32), GGML_ROPE_TYPE_NEOX, 10000.0F}).build(ctx, q, positions);
    k = modules::RoPEModule({std::max<int64_t>(dim / 2, 32), GGML_ROPE_TYPE_NEOX, 10000.0F}).build(ctx, k, positions);
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    v_heads = apply_padding_to_v(ctx, v_heads, padding_mask);
    auto context = rf_attention(ctx, q_heads, k_heads, v_heads, dim);
    if (weights.differential) {
        q_diff = rms_norm(ctx, reshape_heads(ctx, q_diff, config.num_heads, dim), weights.q_norm, kQkNormEps);
        k_diff = rms_norm(ctx, reshape_heads(ctx, k_diff, config.num_heads, dim), weights.k_norm, kQkNormEps);
        q_diff = modules::RoPEModule({std::max<int64_t>(dim / 2, 32), GGML_ROPE_TYPE_NEOX, 10000.0F}).build(ctx, q_diff, positions);
        k_diff = modules::RoPEModule({std::max<int64_t>(dim / 2, 32), GGML_ROPE_TYPE_NEOX, 10000.0F}).build(ctx, k_diff, positions);
        auto qd_heads = modules::TransposeModule({{0, 2, 1, 3}, q_diff.shape.rank}).build(ctx, q_diff);
        auto kd_heads = modules::TransposeModule({{0, 2, 1, 3}, k_diff.shape.rank}).build(ctx, k_diff);
        auto diff_context = rf_attention(ctx, qd_heads, kd_heads, v_heads, dim);
        context = sub_tensor(ctx, context, diff_context);
    }
    context = core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, context),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.embed_dim}));
    return modules::LinearModule({config.embed_dim, config.embed_dim, false, GGML_PREC_F32})
        .build(ctx, context, weights.to_out);
}

core::TensorValue cross_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & context_input,
    const StableAudioRfAttentionWeights & weights,
    const StableAudioConfig & config) {
    const int64_t dim = head_dim(config);
    auto q = modules::LinearModule({config.embed_dim, config.embed_dim * (weights.differential ? 2 : 1), false, GGML_PREC_F32})
        .build(ctx, input, weights.to_q);
    auto kv = modules::LinearModule({config.embed_dim, config.embed_dim * (weights.differential ? 3 : 2), false, GGML_PREC_F32})
        .build(ctx, context_input, weights.to_kv);
    core::TensorValue q_diff;
    core::TensorValue k_diff;
    if (weights.differential) {
        q_diff = modules::SliceModule({2, config.embed_dim, config.embed_dim}).build(ctx, q);
        q = modules::SliceModule({2, 0, config.embed_dim}).build(ctx, q);
        k_diff = modules::SliceModule({2, config.embed_dim, config.embed_dim}).build(ctx, kv);
        auto v = modules::SliceModule({2, 2 * config.embed_dim, config.embed_dim}).build(ctx, kv);
        auto k = modules::SliceModule({2, 0, config.embed_dim}).build(ctx, kv);
        q = rms_norm(ctx, reshape_heads(ctx, q, config.num_heads, dim), weights.q_norm, kQkNormEps);
        k = rms_norm(ctx, reshape_heads(ctx, k, config.num_heads, dim), weights.k_norm, kQkNormEps);
        q_diff = rms_norm(ctx, reshape_heads(ctx, q_diff, config.num_heads, dim), weights.q_norm, kQkNormEps);
        k_diff = rms_norm(ctx, reshape_heads(ctx, k_diff, config.num_heads, dim), weights.k_norm, kQkNormEps);
        v = reshape_heads(ctx, v, config.num_heads, dim);
        auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
        auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
        auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
        auto context = rf_attention(ctx, q_heads, k_heads, v_heads, dim);
        auto qd_heads = modules::TransposeModule({{0, 2, 1, 3}, q_diff.shape.rank}).build(ctx, q_diff);
        auto kd_heads = modules::TransposeModule({{0, 2, 1, 3}, k_diff.shape.rank}).build(ctx, k_diff);
        auto diff_context = rf_attention(ctx, qd_heads, kd_heads, v_heads, dim);
        context = sub_tensor(ctx, context, diff_context);
        context = core::reshape_tensor(ctx, ensure_contiguous(ctx, context), core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.embed_dim}));
        return modules::LinearModule({config.embed_dim, config.embed_dim, false, GGML_PREC_F32}).build(ctx, context, weights.to_out);
    }
    auto k = modules::SliceModule({2, 0, config.embed_dim}).build(ctx, kv);
    auto v = modules::SliceModule({2, config.embed_dim, config.embed_dim}).build(ctx, kv);
    q = rms_norm(ctx, reshape_heads(ctx, q, config.num_heads, dim), weights.q_norm, kQkNormEps);
    k = rms_norm(ctx, reshape_heads(ctx, k, config.num_heads, dim), weights.k_norm, kQkNormEps);
    v = reshape_heads(ctx, v, config.num_heads, dim);
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context = rf_attention(ctx, q_heads, k_heads, v_heads, dim);
    context = core::reshape_tensor(ctx, ensure_contiguous(ctx, context), core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.embed_dim}));
    return modules::LinearModule({config.embed_dim, config.embed_dim, false, GGML_PREC_F32}).build(ctx, context, weights.to_out);
}

core::TensorValue swiglu_ff(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const StableAudioRfFeedForwardWeights & weights,
    const StableAudioConfig & config) {
    auto projected = modules::LinearModule({config.embed_dim, 2 * ff_inner_dim(config), true, GGML_PREC_F32})
        .build(ctx, input, weights.in_proj);
    auto value = modules::SliceModule({2, 0, ff_inner_dim(config)}).build(ctx, projected);
    auto gate = modules::SliceModule({2, ff_inner_dim(config), ff_inner_dim(config)}).build(ctx, projected);
    gate = modules::SiluModule{}.build(ctx, gate);
    auto hidden = modules::MulModule{}.build(ctx, value, gate);
    return modules::LinearModule({ff_inner_dim(config), config.embed_dim, true, GGML_PREC_F32})
        .build(ctx, hidden, weights.out_proj);
}

core::TensorValue build_local_embedding(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const core::TensorValue & local_add,
    const StableAudioRfLayerWeights & weights,
    const StableAudioConfig & config) {
    auto local = modules::LinearModule({config.local_add_cond_dim, config.embed_dim, true, GGML_PREC_F32})
        .build(ctx, local_add, weights.local_embed_in);
    local = modules::SiluModule{}.build(ctx, local);
    local = modules::LinearModule({config.embed_dim, config.embed_dim, true, GGML_PREC_F32})
        .build(ctx, local, weights.local_embed_out);
    auto prefix = modules::SliceModule({1, 0, config.num_memory_tokens}).build(ctx, x);
    prefix = sub_tensor(ctx, prefix, prefix);
    return modules::ConcatModule({1}).build(ctx, prefix, local);
}

core::TensorValue build_rf_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & cross_context,
    const core::TensorValue & global_six,
    const core::TensorValue & local_add,
    const core::TensorValue & positions,
    const core::TensorValue & padding_mask,
    const StableAudioRfLayerWeights & weights,
    const StableAudioConfig & config) {
    auto gate_bias = core::reshape_tensor(ctx, weights.scale_shift_gate, core::TensorShape::from_dims({1, 6 * config.embed_dim}));
    gate_bias = modules::RepeatModule({core::TensorShape::from_dims({input.shape.dims[0], 6 * config.embed_dim})}).build(ctx, gate_bias);
    auto adaln = modules::AddModule{}.build(ctx, global_six, gate_bias);
    adaln = core::reshape_tensor(ctx, adaln, core::TensorShape::from_dims({input.shape.dims[0], 1, 6 * config.embed_dim}));
    auto scale_self = modules::SliceModule({2, 0, config.embed_dim}).build(ctx, adaln);
    auto shift_self = modules::SliceModule({2, config.embed_dim, config.embed_dim}).build(ctx, adaln);
    auto gate_self = modules::SliceModule({2, 2 * config.embed_dim, config.embed_dim}).build(ctx, adaln);
    auto scale_ff = modules::SliceModule({2, 3 * config.embed_dim, config.embed_dim}).build(ctx, adaln);
    auto shift_ff = modules::SliceModule({2, 4 * config.embed_dim, config.embed_dim}).build(ctx, adaln);
    auto gate_ff = modules::SliceModule({2, 5 * config.embed_dim, config.embed_dim}).build(ctx, adaln);
    scale_self = modules::RepeatModule({input.shape}).build(ctx, scale_self);
    shift_self = modules::RepeatModule({input.shape}).build(ctx, shift_self);
    gate_self = modules::RepeatModule({input.shape}).build(ctx, gate_self);
    scale_ff = modules::RepeatModule({input.shape}).build(ctx, scale_ff);
    shift_ff = modules::RepeatModule({input.shape}).build(ctx, shift_ff);
    gate_ff = modules::RepeatModule({input.shape}).build(ctx, gate_ff);

    auto hidden = rms_norm(ctx, input, weights.pre_norm_gamma, kTransformerNormEps);
    hidden = modules::AddModule{}.build(ctx, modules::MulModule{}.build(ctx, hidden, one_plus(ctx, scale_self)), shift_self);
    auto attn = self_attention(ctx, hidden, positions, padding_mask, weights.self_attn, config);
    attn = modules::MulModule{}.build(ctx, attn, sigmoid_one_minus(ctx, gate_self));
    hidden = modules::AddModule{}.build(ctx, input, attn);

    auto cross_norm = rms_norm(ctx, hidden, weights.cross_attend_norm_gamma, kTransformerNormEps);
    auto cross = cross_attention(ctx, cross_norm, cross_context, weights.cross_attn, config);
    hidden = modules::AddModule{}.build(ctx, hidden, cross);
    auto local = build_local_embedding(ctx, hidden, local_add, weights, config);
    hidden = modules::AddModule{}.build(ctx, hidden, local);

    auto ff_in = rms_norm(ctx, hidden, weights.ff_norm_gamma, kTransformerNormEps);
    ff_in = modules::AddModule{}.build(ctx, modules::MulModule{}.build(ctx, ff_in, one_plus(ctx, scale_ff)), shift_ff);
    auto ff = swiglu_ff(ctx, ff_in, weights.ff, config);
    ff = modules::MulModule{}.build(ctx, ff, sigmoid_one_minus(ctx, gate_ff));
    return modules::AddModule{}.build(ctx, hidden, ff);
}

StableAudioRfAttentionWeights load_self_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const StableAudioConfig & config,
    assets::TensorStorageType storage_type) {
    StableAudioRfAttentionWeights weights;
    weights.differential = config.differential_attention;
    weights.to_qkv = load_linear(
        store,
        source,
        prefix + ".to_qkv",
        storage_type,
        config.embed_dim * (config.differential_attention ? 5 : 3),
        config.embed_dim,
        false);
    weights.to_out = load_linear(store, source, prefix + ".to_out", storage_type, config.embed_dim, config.embed_dim, false);
    weights.q_norm = store.load_f32_tensor(source, prefix + ".q_norm.gamma", {head_dim(config)});
    weights.k_norm = store.load_f32_tensor(source, prefix + ".k_norm.gamma", {head_dim(config)});
    return weights;
}

StableAudioRfAttentionWeights load_cross_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const StableAudioConfig & config,
    assets::TensorStorageType storage_type) {
    StableAudioRfAttentionWeights weights;
    weights.differential = config.differential_attention;
    weights.to_q = load_linear(
        store,
        source,
        prefix + ".to_q",
        storage_type,
        config.embed_dim * (config.differential_attention ? 2 : 1),
        config.embed_dim,
        false);
    weights.to_kv = load_linear(
        store,
        source,
        prefix + ".to_kv",
        storage_type,
        config.embed_dim * (config.differential_attention ? 3 : 2),
        config.embed_dim,
        false);
    weights.to_out = load_linear(store, source, prefix + ".to_out", storage_type, config.embed_dim, config.embed_dim, false);
    weights.q_norm = store.load_f32_tensor(source, prefix + ".q_norm.gamma", {head_dim(config)});
    weights.k_norm = store.load_f32_tensor(source, prefix + ".k_norm.gamma", {head_dim(config)});
    return weights;
}

StableAudioRfFeedForwardWeights load_feed_forward(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const StableAudioConfig & config,
    assets::TensorStorageType storage_type) {
    StableAudioRfFeedForwardWeights weights;
    weights.in_proj = load_linear(
        store,
        source,
        prefix + ".ff.0.proj",
        storage_type,
        ff_inner_dim(config) * 2,
        config.embed_dim,
        true);
    weights.out_proj = load_linear(
        store,
        source,
        prefix + ".ff.2",
        storage_type,
        config.embed_dim,
        ff_inner_dim(config),
        true);
    return weights;
}

StableAudioRfLayerWeights load_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const StableAudioConfig & config,
    assets::TensorStorageType storage_type) {
    StableAudioRfLayerWeights weights;
    weights.pre_norm_gamma = store.load_f32_tensor(source, prefix + ".pre_norm.gamma", {config.embed_dim});
    weights.cross_attend_norm_gamma = store.load_f32_tensor(source, prefix + ".cross_attend_norm.gamma", {config.embed_dim});
    weights.ff_norm_gamma = store.load_f32_tensor(source, prefix + ".ff_norm.gamma", {config.embed_dim});
    weights.self_attn = load_self_attention(store, source, prefix + ".self_attn", config, storage_type);
    weights.cross_attn = load_cross_attention(store, source, prefix + ".cross_attn", config, storage_type);
    weights.ff = load_feed_forward(store, source, prefix + ".ff", config, storage_type);
    weights.local_embed_in = load_linear(
        store,
        source,
        prefix + ".to_local_embed.0",
        storage_type,
        config.embed_dim,
        config.local_add_cond_dim,
        true);
    weights.local_embed_out = load_linear(
        store,
        source,
        prefix + ".to_local_embed.2",
        storage_type,
        config.embed_dim,
        config.embed_dim,
        true);
    weights.scale_shift_gate = store.load_f32_tensor(source, prefix + ".to_scale_shift_gate", {6 * config.embed_dim});
    return weights;
}

}  // namespace

StableAudioRfDitWeights load_stable_audio_rf_dit_weights(
    const StableAudioAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type) {
    const auto & config = assets.config;
    const auto & source = *assets.model_weights;
    StableAudioRfDitWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "stable_audio.rf_dit.weights",
        weight_context_bytes != 0 ? weight_context_bytes :
            (config.medium_architecture ? kRfWeightContextBytesMedium : kRfWeightContextBytesSmall));
    weights.to_cond_embed_0 = load_linear(*weights.store, source, "model.model.to_cond_embed.0", weight_storage_type, config.embed_dim, config.cond_token_dim, false);
    weights.to_cond_embed_2 = load_linear(*weights.store, source, "model.model.to_cond_embed.2", weight_storage_type, config.embed_dim, config.embed_dim, false);
    weights.to_global_embed_0 = load_linear(*weights.store, source, "model.model.to_global_embed.0", weight_storage_type, config.embed_dim, config.global_cond_dim, false);
    weights.to_global_embed_2 = load_linear(*weights.store, source, "model.model.to_global_embed.2", weight_storage_type, config.embed_dim, config.embed_dim, false);
    weights.timestep_embed_0 = load_linear(*weights.store, source, "model.model.to_timestep_embed.0", weight_storage_type, config.embed_dim, 256, true);
    weights.timestep_embed_2 = load_linear(*weights.store, source, "model.model.to_timestep_embed.2", weight_storage_type, config.embed_dim, config.embed_dim, true);
    weights.preprocess_conv = load_linear_as_shape(*weights.store, source, "model.model.preprocess_conv", weight_storage_type, config.io_channels, config.io_channels, false);
    weights.postprocess_conv = load_linear_as_shape(*weights.store, source, "model.model.postprocess_conv", weight_storage_type, config.io_channels, config.io_channels, false);
    weights.project_in = load_linear(*weights.store, source, "model.model.transformer.project_in", weight_storage_type, config.embed_dim, config.io_channels, false);
    weights.project_out = load_linear(*weights.store, source, "model.model.transformer.project_out", weight_storage_type, config.io_channels, config.embed_dim, false);
    weights.memory_tokens = weights.store->load_f32_tensor(source, "model.model.transformer.memory_tokens", {config.num_memory_tokens, config.embed_dim});
    weights.global_cond_embedder_0 = load_linear(*weights.store, source, "model.model.transformer.global_cond_embedder.0", weight_storage_type, config.embed_dim, config.embed_dim, true);
    weights.global_cond_embedder_2 = load_linear(*weights.store, source, "model.model.transformer.global_cond_embedder.2", weight_storage_type, 6 * config.embed_dim, config.embed_dim, true);
    weights.layers.reserve(static_cast<size_t>(config.depth));
    for (int64_t layer = 0; layer < config.depth; ++layer) {
        weights.layers.push_back(load_layer(
            *weights.store,
            source,
            "model.model.transformer.layers." + std::to_string(layer),
            config,
            weight_storage_type));
    }
    weights.store->upload();
    return weights;
}

class StableAudioRfDitRuntime::Graph {
public:
    Graph(
        core::ExecutionContext & execution,
        std::shared_ptr<const StableAudioAssets> assets,
        const StableAudioRfDitWeights & weights,
        const StableAudioSamplingState & sampling,
        bool use_cfg)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          assets_(std::move(assets)),
          batch_(sampling.batch),
          latent_tokens_(sampling.latent_sample_size),
          use_cfg_(use_cfg),
          timestep_freqs_(make_fourier_freqs()),
          weights_(weights) {
        if (backend_ == nullptr) {
            throw std::runtime_error("Stable Audio RF DiT backend initialization failed");
        }
        if (batch_ <= 0 || latent_tokens_ <= 0) {
            throw std::runtime_error("Stable Audio RF DiT graph requires positive batch and latent length");
        }
        build();
    }

    ~Graph() {
        if (backend_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(backend_type_, backend_, graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(const StableAudioSamplingState & sampling) const noexcept {
        return sampling.batch == batch_ && sampling.latent_sample_size == latent_tokens_;
    }

    bool matches_cfg(bool use_cfg) const noexcept {
        return use_cfg_ == use_cfg;
    }

    std::vector<float> sample(
        const StableAudioSamplingState & sampling,
        const StableAudioConditioningInputs & conditioning,
        uint64_t seed,
        uint64_t & rng_offset_blocks,
        const engine::sampling::TorchCudaSamplingPolicy & rng_policy,
        std::string_view sampler_type,
        float cfg_scale,
        float apg_scale) const {
        validate_inputs(sampling, conditioning);
        const bool pingpong_sampler = sampler_type.empty() || sampler_type == "pingpong";
        if (!pingpong_sampler && sampler_type != "euler") {
            throw std::runtime_error("Stable Audio sampler_type must be pingpong or euler");
        }
        const auto total_start = Clock::now();
        std::vector<float> latent = sampling.noise;
        std::vector<float> graph_x(static_cast<size_t>(graph_batch() * latent_tokens_ * assets_->config.io_channels));
        const auto cross = make_cross_conditioning(conditioning);
        const auto global = make_global_conditioning(conditioning);
        const auto local = make_local_conditioning(sampling);
        const auto padding = make_padding_mask(sampling);
        core::write_tensor_f32(cross_, cross);
        core::write_tensor_f32(global_, global);
        core::write_tensor_f32(local_add_, local);
        core::write_tensor_f32(padding_mask_, padding);
        if (use_cfg_) {
            core::write_tensor_f32(cfg_scale_, std::vector<float>{cfg_scale});
            core::write_tensor_f32(apg_scale_, std::vector<float>{apg_scale});
            core::write_tensor_f32(latent_padding_mask_, sampling.padding_mask);
        }
        core::set_backend_threads(backend_, threads_);
        std::vector<float> model_output;
        model_output.reserve(static_cast<size_t>(batch_ * assets_->config.io_channels * latent_tokens_));
        if (sampling.schedule_points <= 1 ||
            static_cast<int64_t>(sampling.schedule.size()) != batch_ * sampling.schedule_points) {
            throw std::runtime_error("Stable Audio RF DiT schedule shape mismatch");
        }
        std::vector<float> sigma_values(static_cast<size_t>(batch_));
        std::vector<float> dt_values(static_cast<size_t>(batch_));
        std::vector<float> timestep_features(static_cast<size_t>(graph_batch() * kTimestepFeaturesDim));
        for (int step = 0; step < static_cast<int>(sampling.schedule_points) - 1; ++step) {
            write_latent_input(latent, graph_x);
            for (int64_t b = 0; b < batch_; ++b) {
                const size_t base = static_cast<size_t>(b * sampling.schedule_points);
                const float sigma = sampling.schedule[base + static_cast<size_t>(step)];
                const float next_sigma = sampling.schedule[base + static_cast<size_t>(step + 1)];
                sigma_values[static_cast<size_t>(b)] = sigma;
                dt_values[static_cast<size_t>(b)] = next_sigma - sigma;
            }
            write_timestep_features(sigma_values, timestep_features);
            core::write_tensor_f32(x_, graph_x);
            core::write_tensor_f32(timestep_features_, timestep_features);
            core::write_tensor_i32(positions_tensor_, positions_);
            core::write_tensor_f32(padding_mask_, padding);
            core::write_tensor_f32(cross_, cross);
            core::write_tensor_f32(global_, global);
            core::write_tensor_f32(local_add_, local);
            if (use_cfg_) {
                core::write_tensor_f32(sigma_, sigma_values);
            }
            const auto compute_start = Clock::now();
            const ggml_status status = engine::core::compute_backend_graph(backend_, graph_, nullptr, "stable_audio.rf_dit");
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Stable Audio RF DiT graph compute failed");
            }
            if (step == 0) {
                engine::debug::timing_log_scalar(
                    "stable_audio.rf_dit.step0.graph.compute_ms",
                    engine::debug::elapsed_ms(compute_start, Clock::now()));
            }
            model_output = core::read_tensor_f32(output_);
            for (const float value : model_output) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error(
                        "Stable Audio RF DiT produced non-finite model output at step " + std::to_string(step));
                }
            }
            std::vector<float> step_noise;
            if (pingpong_sampler) {
                step_noise = engine::sampling::generate_torch_cuda_tensor_iterator_randn(
                    latent.size(),
                    seed,
                    rng_offset_blocks,
                    rng_policy,
                    engine::sampling::TorchRandnPrecision::Float32
                );
                rng_offset_blocks += engine::sampling::torch_cuda_tensor_iterator_offset_blocks(
                    static_cast<uint64_t>(latent.size()),
                    rng_policy);
            }
            for (size_t i = 0; i < latent.size(); ++i) {
                const int64_t b = static_cast<int64_t>(i) / (assets_->config.io_channels * latent_tokens_);
                if (pingpong_sampler) {
                    const float sigma = sigma_values[static_cast<size_t>(b)];
                    const float next_sigma = sigma + dt_values[static_cast<size_t>(b)];
                    const float denoised = latent[i] - sigma * model_output[i];
                    latent[i] = (1.0F - next_sigma) * denoised + next_sigma * step_noise[i];
                } else {
                    latent[i] += dt_values[static_cast<size_t>(b)] * model_output[i];
                }
                if (!std::isfinite(latent[i])) {
                    throw std::runtime_error(
                        "Stable Audio RF DiT produced non-finite latent at step " + std::to_string(step));
                }
            }
        }
        engine::debug::timing_log_scalar("stable_audio.rf_dit.total_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
        return latent;
    }

private:
    void build() {
        const auto & config = assets_->config;
        const int64_t batch2 = graph_batch();
        const int64_t total_tokens = config.num_memory_tokens + latent_tokens_;
        ggml_init_params params{512ull * 1024ull * 1024ull, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("Stable Audio RF DiT ggml context initialization failed");
        }
        auto input_ctx = ctx_for_inputs();
        x_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch2, latent_tokens_, config.io_channels}));
        timestep_features_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch2, kTimestepFeaturesDim}));
        cross_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch2, config.prompt_max_length + 1, config.cond_dim}));
        global_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch2, config.global_cond_dim}));
        local_add_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch2, latent_tokens_, config.local_add_cond_dim}));
        padding_mask_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch2, 1, total_tokens, 1}));
        latent_padding_mask_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_, latent_tokens_}));
        sigma_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_, 1, 1}));
        cfg_scale_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        apg_scale_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        positions_tensor_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({total_tokens}));
        ggml_set_input(x_.tensor);
        ggml_set_input(timestep_features_.tensor);
        ggml_set_input(cross_.tensor);
        ggml_set_input(global_.tensor);
        ggml_set_input(local_add_.tensor);
        ggml_set_input(padding_mask_.tensor);
        ggml_set_input(latent_padding_mask_.tensor);
        ggml_set_input(sigma_.tensor);
        ggml_set_input(cfg_scale_.tensor);
        ggml_set_input(apg_scale_.tensor);
        ggml_set_input(positions_tensor_.tensor);
        core::ModuleBuildContext build_ctx{ctx_.get(), "stable_audio.rf_dit", backend_type_};
        auto output = build_graph_output(build_ctx);
        output_ = output.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 1048576, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("Stable Audio RF DiT backend buffer allocation failed");
        }
        positions_.assign(static_cast<size_t>(total_tokens), 0);
        for (int64_t i = 0; i < total_tokens; ++i) {
            positions_[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        core::write_tensor_i32(positions_tensor_, positions_);
    }

    core::ModuleBuildContext ctx_for_inputs() {
        return core::ModuleBuildContext{ctx_.get(), "stable_audio.rf_dit.inputs", backend_type_};
    }

    void write_timestep_features(const std::vector<float> & sigmas, std::vector<float> & out) const {
        const int64_t batch2 = graph_batch();
        const int64_t half = kTimestepFeaturesDim / 2;
        if (static_cast<int64_t>(sigmas.size()) != batch_) {
            throw std::runtime_error("Stable Audio RF DiT sigma vector shape mismatch");
        }
        if (static_cast<int64_t>(out.size()) != batch2 * kTimestepFeaturesDim) {
            throw std::runtime_error("Stable Audio RF DiT timestep feature buffer shape mismatch");
        }
        for (int64_t b = 0; b < batch2; ++b) {
            const float sigma = sigmas[static_cast<size_t>(b % batch_)];
            float * row = out.data() + static_cast<std::ptrdiff_t>(b * kTimestepFeaturesDim);
            for (int64_t i = 0; i < half; ++i) {
                const float arg = sigma * timestep_freqs_[static_cast<size_t>(i)];
                row[i] = std::cos(arg);
                row[half + i] = std::sin(arg);
            }
        }
    }

    core::TensorValue build_graph_output(core::ModuleBuildContext & ctx) {
        const auto & config = assets_->config;
        auto hidden = modules::LinearModule({config.io_channels, config.io_channels, false, GGML_PREC_F32})
            .build(ctx, x_, weights_.preprocess_conv);
        hidden = modules::AddModule{}.build(ctx, hidden, x_);
        hidden = modules::LinearModule({config.io_channels, config.embed_dim, false, GGML_PREC_F32})
            .build(ctx, hidden, weights_.project_in);
        auto memory = core::reshape_tensor(ctx, weights_.memory_tokens, core::TensorShape::from_dims({1, config.num_memory_tokens, config.embed_dim}));
        memory = modules::RepeatModule({core::TensorShape::from_dims({graph_batch(), config.num_memory_tokens, config.embed_dim})}).build(ctx, memory);
        hidden = modules::ConcatModule({1}).build(ctx, memory, hidden);
        auto cross = modules::LinearModule({config.cond_dim, config.embed_dim, false, GGML_PREC_F32})
            .build(ctx, cross_, weights_.to_cond_embed_0);
        cross = modules::SiluModule{}.build(ctx, cross);
        cross = modules::LinearModule({config.embed_dim, config.embed_dim, false, GGML_PREC_F32})
            .build(ctx, cross, weights_.to_cond_embed_2);

        auto global = modules::LinearModule({config.global_cond_dim, config.embed_dim, false, GGML_PREC_F32})
            .build(ctx, global_, weights_.to_global_embed_0);
        global = modules::SiluModule{}.build(ctx, global);
        global = modules::LinearModule({config.embed_dim, config.embed_dim, false, GGML_PREC_F32})
            .build(ctx, global, weights_.to_global_embed_2);
        global = modules::AddModule{}.build(ctx, global, build_timestep_embedding(ctx, timestep_features_, weights_, config));
        global = modules::LinearModule({config.embed_dim, config.embed_dim, true, GGML_PREC_F32})
            .build(ctx, global, weights_.global_cond_embedder_0);
        global = modules::SiluModule{}.build(ctx, global);
        global = modules::LinearModule({config.embed_dim, 6 * config.embed_dim, true, GGML_PREC_F32})
            .build(ctx, global, weights_.global_cond_embedder_2);

        for (size_t layer_index = 0; layer_index < weights_.layers.size(); ++layer_index) {
            const auto & layer = weights_.layers[layer_index];
            hidden = build_rf_layer(
                ctx,
                hidden,
                cross,
                global,
                local_add_,
                positions_tensor_,
                padding_mask_,
                layer,
                config);
        }
        hidden = modules::SliceModule({1, config.num_memory_tokens, latent_tokens_}).build(ctx, hidden);
        hidden = modules::LinearModule({config.embed_dim, config.io_channels, false, GGML_PREC_F32})
            .build(ctx, hidden, weights_.project_out);
        auto post = modules::LinearModule({config.io_channels, config.io_channels, false, GGML_PREC_F32})
            .build(ctx, hidden, weights_.postprocess_conv);
        auto raw_output = modules::AddModule{}.build(ctx, hidden, post);
        if (!use_cfg_) {
            return modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, raw_output);
        }
        auto cfg_output = build_cfg_model_output(ctx, raw_output);
        return modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, cfg_output);
    }

    core::TensorValue build_cfg_model_output(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & raw_output) {
        const auto & config = assets_->config;
        auto x_cond = modules::SliceModule({0, 0, batch_}).build(ctx, x_);
        auto cond = modules::SliceModule({0, 0, batch_}).build(ctx, raw_output);
        auto uncond = modules::SliceModule({0, batch_, batch_}).build(ctx, raw_output);
        auto sigma = core::reshape_tensor(ctx, sigma_, core::TensorShape::from_dims({batch_, 1, 1}));
        sigma = repeat_like(ctx, sigma, cond);
        auto cond_x0 = sub_tensor(ctx, x_cond, modules::MulModule{}.build(ctx, cond, sigma));
        auto uncond_x0 = sub_tensor(ctx, x_cond, modules::MulModule{}.build(ctx, uncond, sigma));
        auto diff = sub_tensor(ctx, cond_x0, uncond_x0);

        auto mask = core::reshape_tensor(ctx, latent_padding_mask_, core::TensorShape::from_dims({batch_, latent_tokens_, 1}));
        mask = modules::RepeatModule({core::TensorShape::from_dims({batch_, latent_tokens_, config.io_channels})}).build(ctx, mask);
        auto cond_masked = modules::MulModule{}.build(ctx, cond_x0, mask);
        auto norm_sq = modules::MulModule{}.build(ctx, cond_masked, cond_masked);
        norm_sq = modules::ReduceSumModule({2}).build(ctx, norm_sq);
        norm_sq = modules::ReduceSumModule({1}).build(ctx, norm_sq);
        auto norm = core::wrap_tensor(
            ggml_sqrt(ctx.ggml, ggml_scale_bias(ctx.ggml, norm_sq.tensor, 1.0F, 1.0e-16F)),
            norm_sq.shape,
            GGML_TYPE_F32);
        auto v1_normalized = div_tensor(ctx, cond_masked, repeat_like(ctx, norm, cond_masked));
        auto dot = modules::MulModule{}.build(ctx, diff, v1_normalized);
        dot = modules::ReduceSumModule({2}).build(ctx, dot);
        dot = modules::ReduceSumModule({1}).build(ctx, dot);
        auto parallel = modules::MulModule{}.build(ctx, repeat_like(ctx, dot, diff), v1_normalized);
        auto orthogonal = modules::MulModule{}.build(ctx, sub_tensor(ctx, diff, parallel), mask);
        auto apg_value = core::reshape_tensor(ctx, apg_scale_, core::TensorShape::from_dims({1, 1, 1}));
        auto apg = repeat_like(ctx, apg_value, diff);
        auto one_minus_apg = scale_bias_tensor(ctx, apg, -1.0F, 1.0F);
        auto cfg_diff = modules::AddModule{}.build(
            ctx,
            modules::MulModule{}.build(ctx, apg, orthogonal),
            modules::MulModule{}.build(ctx, one_minus_apg, diff));
        auto cfg_value = core::reshape_tensor(ctx, cfg_scale_, core::TensorShape::from_dims({1, 1, 1}));
        auto cfg = scale_bias_tensor(ctx, repeat_like(ctx, cfg_value, diff), 1.0F, -1.0F);
        auto cfg_denoised = modules::AddModule{}.build(ctx, cond_x0, modules::MulModule{}.build(ctx, cfg, cfg_diff));
        return div_tensor(ctx, sub_tensor(ctx, x_cond, cfg_denoised), sigma);
    }

    void validate_inputs(
        const StableAudioSamplingState & sampling,
        const StableAudioConditioningInputs & conditioning) const {
        const auto & config = assets_->config;
        if (!matches(sampling)) {
            throw std::runtime_error("Stable Audio RF DiT graph shape mismatch");
        }
        if (sampling.channels != config.io_channels) {
            throw std::runtime_error("Stable Audio RF DiT channel count mismatch");
        }
        if (conditioning.positive.batch != batch_ ||
            conditioning.positive.cross_tokens != config.prompt_max_length + 1 ||
            conditioning.positive.cond_dim != config.cond_dim) {
            throw std::runtime_error("Stable Audio RF DiT positive conditioning shape mismatch");
        }
        if (conditioning.has_negative &&
            (conditioning.negative.batch != batch_ ||
             conditioning.negative.cross_tokens != config.prompt_max_length + 1 ||
             conditioning.negative.cond_dim != config.cond_dim)) {
            throw std::runtime_error("Stable Audio RF DiT negative conditioning shape mismatch");
        }
    }

    std::vector<float> make_cross_conditioning(const StableAudioConditioningInputs & conditioning) const {
        const auto & config = assets_->config;
        const int64_t tokens = config.prompt_max_length + 1;
        std::vector<float> out(static_cast<size_t>(graph_batch() * tokens * config.cond_dim), 0.0F);
        const int64_t row = tokens * config.cond_dim;
        for (int64_t b = 0; b < batch_; ++b) {
            std::copy_n(
                conditioning.positive.cross_attention.data() + static_cast<std::ptrdiff_t>(b * row),
                static_cast<size_t>(row),
                out.data() + static_cast<std::ptrdiff_t>(b * row));
            if (use_cfg_ && conditioning.has_negative) {
                const float * negative = conditioning.negative.cross_attention.data() + static_cast<std::ptrdiff_t>(b * row);
                float * dst = out.data() + static_cast<std::ptrdiff_t>((batch_ + b) * row);
                for (int64_t t = 0; t < tokens; ++t) {
                    const bool keep = conditioning.negative.cross_attention_mask[static_cast<size_t>(b * tokens + t)] != 0;
                    if (keep) {
                        std::copy_n(
                            negative + static_cast<std::ptrdiff_t>(t * config.cond_dim),
                            static_cast<size_t>(config.cond_dim),
                            dst + static_cast<std::ptrdiff_t>(t * config.cond_dim));
                    }
                }
            }
        }
        return out;
    }

    std::vector<float> make_global_conditioning(const StableAudioConditioningInputs & conditioning) const {
        const auto & config = assets_->config;
        std::vector<float> out(static_cast<size_t>(graph_batch() * config.global_cond_dim), 0.0F);
        for (int64_t b = 0; b < batch_; ++b) {
            const float * src = conditioning.positive.global.data() + static_cast<std::ptrdiff_t>(b * config.global_cond_dim);
            std::copy_n(src, static_cast<size_t>(config.global_cond_dim), out.data() + static_cast<std::ptrdiff_t>(b * config.global_cond_dim));
            if (use_cfg_) {
                std::copy_n(src, static_cast<size_t>(config.global_cond_dim), out.data() + static_cast<std::ptrdiff_t>((batch_ + b) * config.global_cond_dim));
            }
        }
        return out;
    }

    std::vector<float> make_local_conditioning(const StableAudioSamplingState & sampling) const {
        const auto & config = assets_->config;
        if (static_cast<int64_t>(sampling.local_add_conditioning.size()) !=
            batch_ * latent_tokens_ * config.local_add_cond_dim) {
            throw std::runtime_error("Stable Audio RF DiT local conditioning shape mismatch");
        }
        std::vector<float> out(static_cast<size_t>(graph_batch() * latent_tokens_ * config.local_add_cond_dim), 0.0F);
        const int64_t row = latent_tokens_ * config.local_add_cond_dim;
        for (int64_t b = 0; b < batch_; ++b) {
            const float * src = sampling.local_add_conditioning.data() + static_cast<std::ptrdiff_t>(b * row);
            std::copy_n(src, static_cast<size_t>(row), out.data() + static_cast<std::ptrdiff_t>(b * row));
            if (use_cfg_) {
                std::copy_n(src, static_cast<size_t>(row), out.data() + static_cast<std::ptrdiff_t>((batch_ + b) * row));
            }
        }
        return out;
    }

    std::vector<float> make_padding_mask(const StableAudioSamplingState & sampling) const {
        const auto & config = assets_->config;
        const int64_t total = config.num_memory_tokens + latent_tokens_;
        std::vector<float> out(static_cast<size_t>(graph_batch() * total), 0.0F);
        for (int64_t bb = 0; bb < graph_batch(); ++bb) {
            const int64_t b = bb % batch_;
            for (int64_t t = 0; t < config.num_memory_tokens; ++t) {
                out[static_cast<size_t>(bb * total + t)] = 1.0F;
            }
            for (int64_t t = 0; t < latent_tokens_; ++t) {
                out[static_cast<size_t>(bb * total + config.num_memory_tokens + t)] =
                    sampling.padding_mask[static_cast<size_t>(b * latent_tokens_ + t)];
            }
        }
        return out;
    }

    void write_latent_input(const std::vector<float> & latent, std::vector<float> & graph_x) const {
        const int64_t channels = assets_->config.io_channels;
        for (int64_t bb = 0; bb < graph_batch(); ++bb) {
            const int64_t b = bb % batch_;
            for (int64_t t = 0; t < latent_tokens_; ++t) {
                for (int64_t c = 0; c < channels; ++c) {
                    graph_x[static_cast<size_t>((bb * latent_tokens_ + t) * channels + c)] =
                        latent[static_cast<size_t>((b * channels + c) * latent_tokens_ + t)];
                }
            }
        }
    }

    int64_t graph_batch() const noexcept {
        return use_cfg_ ? 2 * batch_ : batch_;
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const StableAudioAssets> assets_;
    int64_t batch_ = 0;
    int64_t latent_tokens_ = 0;
    bool use_cfg_ = false;
    std::vector<float> timestep_freqs_;
    const StableAudioRfDitWeights & weights_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue x_;
    core::TensorValue timestep_features_;
    core::TensorValue cross_;
    core::TensorValue global_;
    core::TensorValue local_add_;
    core::TensorValue padding_mask_;
    core::TensorValue latent_padding_mask_;
    core::TensorValue sigma_;
    core::TensorValue cfg_scale_;
    core::TensorValue apg_scale_;
    core::TensorValue positions_tensor_;
    ggml_tensor * output_ = nullptr;
    std::vector<int32_t> positions_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

StableAudioRfDitRuntime::StableAudioRfDitRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const StableAudioAssets> assets,
    assets::TensorStorageType weight_storage_type)
    : execution_(&execution),
      assets_(std::move(assets)),
      weight_storage_type_(weight_storage_type) {
    if (execution_ == nullptr) {
        throw std::runtime_error("Stable Audio RF DiT runtime requires execution context");
    }
    if (assets_ == nullptr) {
        throw std::runtime_error("Stable Audio RF DiT runtime requires assets");
    }
}

StableAudioRfDitRuntime::~StableAudioRfDitRuntime() = default;

const StableAudioRfDitWeights & require_rf_weights(
    std::unique_ptr<StableAudioRfDitWeights> & weights,
    core::ExecutionContext & execution,
    const std::shared_ptr<const StableAudioAssets> & assets,
    assets::TensorStorageType weight_storage_type) {
    if (!weights) {
        weights = std::make_unique<StableAudioRfDitWeights>(load_stable_audio_rf_dit_weights(
            *assets,
            execution.backend(),
            execution.backend_type(),
            0,
            weight_storage_type));
    }
    return *weights;
}

void StableAudioRfDitRuntime::prepare(const StableAudioSamplingState & sampling, float cfg_scale) const {
    const bool use_cfg = cfg_scale != 1.0F;
    const auto & weights = require_rf_weights(weights_, *execution_, assets_, weight_storage_type_);
    if (!graph_ || !graph_->matches(sampling) || !graph_->matches_cfg(use_cfg)) {
        graph_ = std::make_unique<Graph>(*execution_, assets_, weights, sampling, use_cfg);
    }
}

std::vector<float> StableAudioRfDitRuntime::sample(
    const StableAudioSamplingState & sampling,
    const StableAudioConditioningInputs & conditioning,
    uint64_t seed,
    uint64_t & rng_offset_blocks,
    const engine::sampling::TorchCudaSamplingPolicy & rng_policy,
    std::string_view sampler_type,
    float cfg_scale,
    float apg_scale) const {
    prepare(sampling, cfg_scale);
    return graph_->sample(sampling, conditioning, seed, rng_offset_blocks, rng_policy, sampler_type, cfg_scale, apg_scale);
}

void StableAudioRfDitRuntime::release_runtime_graphs() const {
    graph_.reset();
}

}  // namespace engine::models::stable_audio
