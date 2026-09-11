#include "engine/models/dramabox/dit.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/packed_linear_weights.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::dramabox {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kDitWeightContextBytes = 8500ull * 1024ull * 1024ull;
constexpr size_t kDitGraphContextBytes = 768ull * 1024ull * 1024ull;
constexpr size_t kDitGraphNodeCapacity = 32768;
constexpr float kNormEps = 1.0e-6F;
constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr int64_t kTimestepFeatures = 256;
constexpr int64_t kStgSelfAttentionBlock = 29;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

modules::LinearWeights load_packed_self_qkv_gate_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t hidden,
    int64_t heads) {
    return modules::PackedLinearWeightsBuilder({
        hidden,
        {
            {prefix + ".to_q.weight", prefix + ".to_q.bias", hidden},
            {prefix + ".to_k.weight", prefix + ".to_k.bias", hidden},
            {prefix + ".to_v.weight", prefix + ".to_v.bias", hidden},
            {prefix + ".to_gate_logits.weight", prefix + ".to_gate_logits.bias", heads},
        },
        true,
    }).build(store, source, storage_type);
}

modules::LinearWeights load_packed_pair_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & first_prefix,
    const std::string & second_prefix,
    assets::TensorStorageType storage_type,
    int64_t first_out,
    int64_t second_out,
    int64_t in_features) {
    return modules::PackedLinearWeightsBuilder({
        in_features,
        {
            {first_prefix + ".weight", first_prefix + ".bias", first_out},
            {second_prefix + ".weight", second_prefix + ".bias", second_out},
        },
        true,
    }).build(store, source, storage_type);
}

DramaBoxAdaLayerNormWeights load_adaln(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t hidden,
    int64_t coefficient) {
    DramaBoxAdaLayerNormWeights weights;
    weights.timestep_linear_1 = modules::binding::linear_from_source(
        store,
        source,
        prefix + ".emb.timestep_embedder.linear_1",
        storage_type,
        hidden,
        kTimestepFeatures,
        true);
    weights.timestep_linear_2 = modules::binding::linear_from_source(
        store,
        source,
        prefix + ".emb.timestep_embedder.linear_2",
        storage_type,
        hidden,
        hidden,
        true);
    weights.output_linear = modules::binding::linear_from_source(
        store, source, prefix + ".linear", storage_type, coefficient * hidden, hidden, true);
    weights.output_coefficient = coefficient;
    return weights;
}

DramaBoxDitSelfAttentionWeights load_self_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t hidden,
    int64_t heads,
    DramaBoxPerfMode perf_mode) {
    DramaBoxDitSelfAttentionWeights weights;
    if (perf_mode == DramaBoxPerfMode::FlashAttention) {
        weights.q = modules::binding::linear_from_source(store, source, prefix + ".to_q", storage_type, hidden, hidden, true);
        weights.k = modules::binding::linear_from_source(store, source, prefix + ".to_k", storage_type, hidden, hidden, true);
        weights.v = modules::binding::linear_from_source(store, source, prefix + ".to_v", storage_type, hidden, hidden, true);
        weights.gate = modules::binding::linear_from_source(store, source, prefix + ".to_gate_logits", storage_type, heads, hidden, true);
    } else {
        weights.qkv_gate = load_packed_self_qkv_gate_linear(store, source, prefix, storage_type, hidden, heads);
    }
    weights.out = modules::binding::linear_from_source(store, source, prefix + ".to_out.0", storage_type, hidden, hidden, true);
    weights.q_norm = store.load_tensor(source, prefix + ".q_norm.weight", assets::TensorStorageType::Native, {hidden});
    weights.k_norm = store.load_tensor(source, prefix + ".k_norm.weight", assets::TensorStorageType::Native, {hidden});
    return weights;
}

DramaBoxDitCrossAttentionWeights load_cross_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t hidden,
    int64_t heads) {
    DramaBoxDitCrossAttentionWeights weights;
    weights.q_gate = load_packed_pair_linear(
        store,
        source,
        prefix + ".to_q",
        prefix + ".to_gate_logits",
        storage_type,
        hidden,
        heads,
        hidden);
    weights.kv = load_packed_pair_linear(
        store,
        source,
        prefix + ".to_k",
        prefix + ".to_v",
        storage_type,
        hidden,
        hidden,
        hidden);
    weights.out = modules::binding::linear_from_source(store, source, prefix + ".to_out.0", storage_type, hidden, hidden, true);
    weights.q_norm = store.load_tensor(source, prefix + ".q_norm.weight", assets::TensorStorageType::Native, {hidden});
    weights.k_norm = store.load_tensor(source, prefix + ".k_norm.weight", assets::TensorStorageType::Native, {hidden});
    return weights;
}

DramaBoxDitBlockWeights load_dit_transformer_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    int64_t index,
    const DramaBoxTransformerConfig & config,
    assets::TensorStorageType storage_type,
    DramaBoxPerfMode perf_mode) {
    const int64_t hidden = config.hidden_size;
    const std::string prefix = "model.diffusion_model.transformer_blocks." + std::to_string(index);
    DramaBoxDitBlockWeights weights;
    weights.audio_scale_shift_table =
        store.load_tensor(source, prefix + ".audio_scale_shift_table", assets::TensorStorageType::F32, {9, hidden});
    weights.audio_prompt_scale_shift_table =
        store.load_tensor(source, prefix + ".audio_prompt_scale_shift_table", assets::TensorStorageType::F32, {2, hidden});
    weights.self_attention = load_self_attention(
        store,
        source,
        prefix + ".audio_attn1",
        storage_type,
        hidden,
        config.num_attention_heads,
        perf_mode);
    weights.cross_attention = load_cross_attention(store, source, prefix + ".audio_attn2", storage_type, hidden, config.num_attention_heads);
    weights.ff_in = modules::binding::linear_from_source(
        store, source, prefix + ".audio_ff.net.0.proj", storage_type, hidden * 4, hidden, true);
    weights.ff_out = modules::binding::linear_from_source(
        store, source, prefix + ".audio_ff.net.2", storage_type, hidden, hidden * 4, true);
    return weights;
}

core::TensorValue attention_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t heads, int64_t dim) {
    auto reshaped = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, input),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
    return modules::TransposeModule({{0, 2, 1, 3}, reshaped.shape.rank}).build(ctx, reshaped);
}

core::TensorValue attention_core_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    const std::optional<core::TensorValue> & mask,
    int64_t heads,
    int64_t head_dim) {
    auto out = modules::ScaledDotProductAttentionModule({
        head_dim,
        modules::ScaledDotProductAttentionLowering::FlashPreserveViews,
        GGML_PREC_DEFAULT,
        modules::AttentionCausality::NonCausal,
    }).build(ctx, q_heads, k_heads, v_heads, mask);
    return core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, out),
        core::TensorShape::from_dims({q_heads.shape.dims[0], q_heads.shape.dims[2], heads * head_dim}));
}

core::TensorValue attention_core(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q,
    const core::TensorValue & k,
    const core::TensorValue & v,
    const std::optional<core::TensorValue> & mask,
    int64_t heads,
    int64_t head_dim) {
    return attention_core_heads(
        ctx,
        attention_heads(ctx, q, heads, head_dim),
        attention_heads(ctx, k, heads, head_dim),
        attention_heads(ctx, v, heads, head_dim),
        mask,
        heads,
        head_dim);
}

core::TensorValue audio_self_attention_core(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t heads,
    int64_t head_dim,
    int64_t ref_tokens) {
    if (ref_tokens <= 0) {
        return attention_core_heads(ctx, q_heads, k_heads, v_heads, std::nullopt, heads, head_dim);
    }
    const int64_t total_tokens = q_heads.shape.dims[2];
    const int64_t target_tokens = total_tokens - ref_tokens;
    if (target_tokens <= 0) {
        return attention_core_heads(ctx, q_heads, k_heads, v_heads, std::nullopt, heads, head_dim);
    }
    auto target = attention_core_heads(
        ctx,
        modules::SliceModule({2, 0, target_tokens}).build(ctx, q_heads),
        k_heads,
        v_heads,
        std::nullopt,
        heads,
        head_dim);
    auto ref = attention_core_heads(
        ctx,
        modules::SliceModule({2, target_tokens, ref_tokens}).build(ctx, q_heads),
        modules::SliceModule({2, target_tokens, ref_tokens}).build(ctx, k_heads),
        modules::SliceModule({2, target_tokens, ref_tokens}).build(ctx, v_heads),
        std::nullopt,
        heads,
        head_dim);
    return modules::ConcatModule({1}).build(ctx, target, ref);
}

core::TensorValue apply_gate_and_out(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const core::TensorValue & attn,
    const core::TensorValue & gates_in,
    const modules::LinearWeights & out_weights,
    int64_t heads,
    int64_t head_dim,
    int64_t hidden) {
    auto gates = modules::SigmoidModule{}.build(ctx, gates_in);
    gates = core::wrap_tensor(ggml_scale(ctx.ggml, gates.tensor, 2.0F), gates.shape, GGML_TYPE_F32);
    auto gate_view = core::reshape_tensor(ctx, gates, core::TensorShape::from_dims({x.shape.dims[0], x.shape.dims[1], heads, 1}));
    auto attn_heads = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, attn),
        core::TensorShape::from_dims({attn.shape.dims[0], attn.shape.dims[1], heads, head_dim}));
    auto out = core::wrap_tensor(ggml_mul(ctx.ggml, attn_heads.tensor, gate_view.tensor), attn_heads.shape, GGML_TYPE_F32);
    out = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, out), x.shape);
    return modules::LinearModule({
        hidden,
        hidden,
        true,
        ggml_is_quantized(out_weights.weight.tensor->type) ? GGML_PREC_DEFAULT : GGML_PREC_F32,
    }).build(ctx, out, out_weights);
}

struct AdaOutputs {
    core::TensorValue modulation;
    core::TensorValue embedding;
};

core::TensorValue self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const core::TensorValue & rope_cos,
    const core::TensorValue & rope_sin,
    int64_t ref_tokens,
    const DramaBoxDitSelfAttentionWeights & weights,
    const DramaBoxTransformerConfig & config,
    bool skip_last_perturbed_attention,
    DramaBoxPerfMode perf_mode) {
    const int64_t heads = config.num_attention_heads;
    const int64_t head_dim = config.attention_head_dim;
    const int64_t hidden = config.hidden_size;
    if (perf_mode == DramaBoxPerfMode::FlashAttention) {
        if (!weights.q.has_value() || !weights.k.has_value() || !weights.v.has_value() || !weights.gate.has_value()) {
            throw std::runtime_error("DramaBox DiT perf self-attention requires split Q/K/V/gate weights");
        }
        const int64_t total_tokens = x.shape.dims[1];
        if (total_tokens <= ref_tokens) {
            throw std::runtime_error("DramaBox DiT perf self-attention has no target tokens");
        }
        auto k = modules::LinearModule({
            hidden,
            hidden,
            true,
            ggml_is_quantized(weights.k->weight.tensor->type) ? GGML_PREC_DEFAULT : GGML_PREC_F32,
        }).build(ctx, x, *weights.k);
        auto v = modules::LinearModule({
            hidden,
            hidden,
            true,
            ggml_is_quantized(weights.v->weight.tensor->type) ? GGML_PREC_DEFAULT : GGML_PREC_F32,
        }).build(ctx, x, *weights.v);
        auto gates = modules::LinearModule({
            hidden,
            heads,
            true,
            ggml_is_quantized(weights.gate->weight.tensor->type) ? GGML_PREC_DEFAULT : GGML_PREC_F32,
        }).build(ctx, x, *weights.gate);
        k = modules::RMSNormModule({hidden, kNormEps, true, false}).build(ctx, k, {weights.k_norm, std::nullopt});
        auto k_heads = modules::SplitRoPEAttentionModule({heads, head_dim}).build(ctx, k, rope_cos, rope_sin);
        auto v_heads = attention_heads(ctx, v, heads, head_dim);
        core::TensorValue attn;
        if (skip_last_perturbed_attention && x.shape.dims[0] >= 2) {
            const int64_t prefix_batch = x.shape.dims[0] - 1;
            auto prefix_x = modules::SliceModule({0, 0, prefix_batch}).build(ctx, x);
            auto prefix_q = modules::LinearModule({
                hidden,
                hidden,
                true,
                ggml_is_quantized(weights.q->weight.tensor->type) ? GGML_PREC_DEFAULT : GGML_PREC_F32,
            }).build(ctx, prefix_x, *weights.q);
            prefix_q = modules::RMSNormModule({hidden, kNormEps, true, false}).build(ctx, prefix_q, {weights.q_norm, std::nullopt});
            auto prefix_q_heads = modules::SplitRoPEAttentionModule({heads, head_dim}).build(
                ctx,
                prefix_q,
                modules::SliceModule({0, 0, prefix_batch}).build(ctx, rope_cos),
                modules::SliceModule({0, 0, prefix_batch}).build(ctx, rope_sin));
            auto prefix_attn = audio_self_attention_core(
                ctx,
                prefix_q_heads,
                modules::SliceModule({0, 0, prefix_batch}).build(ctx, k_heads),
                modules::SliceModule({0, 0, prefix_batch}).build(ctx, v_heads),
                heads,
                head_dim,
                ref_tokens);
            attn = modules::ConcatModule({0}).build(ctx, prefix_attn, modules::SliceModule({0, prefix_batch, 1}).build(ctx, v));
        } else {
            auto q = modules::LinearModule({
                hidden,
                hidden,
                true,
                ggml_is_quantized(weights.q->weight.tensor->type) ? GGML_PREC_DEFAULT : GGML_PREC_F32,
            }).build(ctx, x, *weights.q);
            q = modules::RMSNormModule({hidden, kNormEps, true, false}).build(ctx, q, {weights.q_norm, std::nullopt});
            auto q_heads = modules::SplitRoPEAttentionModule({heads, head_dim}).build(ctx, q, rope_cos, rope_sin);
            attn = audio_self_attention_core(ctx, q_heads, k_heads, v_heads, heads, head_dim, ref_tokens);
        }
        return apply_gate_and_out(ctx, x, attn, gates, weights.out, heads, head_dim, hidden);
    }
    auto qkv_gate = modules::LinearModule({
        hidden,
        3 * hidden + heads,
        true,
        ggml_is_quantized(weights.qkv_gate.weight.tensor->type) ? GGML_PREC_DEFAULT : GGML_PREC_F32,
    }).build(ctx, x, weights.qkv_gate);
    auto q = modules::SliceModule({2, 0, hidden}).build(ctx, qkv_gate);
    auto k = modules::SliceModule({2, hidden, hidden}).build(ctx, qkv_gate);
    auto v = modules::SliceModule({2, 2 * hidden, hidden}).build(ctx, qkv_gate);
    auto gates = modules::SliceModule({2, 3 * hidden, heads}).build(ctx, qkv_gate);
    q = modules::RMSNormModule({hidden, kNormEps, true, false}).build(ctx, q, {weights.q_norm, std::nullopt});
    k = modules::RMSNormModule({hidden, kNormEps, true, false}).build(ctx, k, {weights.k_norm, std::nullopt});
    auto q_heads = modules::SplitRoPEAttentionModule({heads, head_dim}).build(ctx, q, rope_cos, rope_sin);
    auto k_heads = modules::SplitRoPEAttentionModule({heads, head_dim}).build(ctx, k, rope_cos, rope_sin);
    core::TensorValue attn;
    if (skip_last_perturbed_attention && x.shape.dims[0] >= 2) {
        const int64_t prefix_batch = x.shape.dims[0] - 1;
        auto prefix_q = modules::SliceModule({0, 0, prefix_batch}).build(ctx, q_heads);
        auto prefix_k = modules::SliceModule({0, 0, prefix_batch}).build(ctx, k_heads);
        auto prefix_v = modules::SliceModule({0, 0, prefix_batch}).build(ctx, v);
        auto prefix_v_heads = attention_heads(ctx, prefix_v, heads, head_dim);
        auto prefix_attn = audio_self_attention_core(ctx, prefix_q, prefix_k, prefix_v_heads, heads, head_dim, ref_tokens);
        attn = modules::ConcatModule({0}).build(ctx, prefix_attn, modules::SliceModule({0, prefix_batch, 1}).build(ctx, v));
    } else {
        auto v_heads = attention_heads(ctx, v, heads, head_dim);
        attn = audio_self_attention_core(ctx, q_heads, k_heads, v_heads, heads, head_dim, ref_tokens);
    }
    return apply_gate_and_out(ctx, x, attn, gates, weights.out, heads, head_dim, hidden);
}

core::TensorValue cross_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const core::TensorValue & context,
    const DramaBoxDitCrossAttentionWeights & weights,
    const DramaBoxTransformerConfig & config,
    bool share_last_context) {
    const int64_t heads = config.num_attention_heads;
    const int64_t head_dim = config.attention_head_dim;
    const int64_t hidden = config.hidden_size;
    auto q_gate = modules::LinearModule({
        hidden,
        hidden + heads,
        true,
        ggml_is_quantized(weights.q_gate.weight.tensor->type) ? GGML_PREC_DEFAULT : GGML_PREC_F32,
    }).build(ctx, x, weights.q_gate);
    auto q = modules::SliceModule({2, 0, hidden}).build(ctx, q_gate);
    auto gates = modules::SliceModule({2, hidden, heads}).build(ctx, q_gate);
    core::TensorValue k;
    core::TensorValue v;
    if (share_last_context && context.shape.dims[0] >= 2) {
        const int64_t prefix_batch = context.shape.dims[0] - 1;
        auto prefix_context = modules::SliceModule({0, 0, prefix_batch}).build(ctx, context);
        auto prefix_kv = modules::LinearModule({
            hidden,
            2 * hidden,
            true,
            ggml_is_quantized(weights.kv.weight.tensor->type) ? GGML_PREC_DEFAULT : GGML_PREC_F32,
        }).build(ctx, prefix_context, weights.kv);
        auto prefix_k = modules::SliceModule({2, 0, hidden}).build(ctx, prefix_kv);
        auto prefix_v = modules::SliceModule({2, hidden, hidden}).build(ctx, prefix_kv);
        auto cond_k = modules::SliceModule({0, 0, 1}).build(ctx, prefix_k);
        auto cond_v = modules::SliceModule({0, 0, 1}).build(ctx, prefix_v);
        k = modules::ConcatModule({0}).build(ctx, prefix_k, cond_k);
        v = modules::ConcatModule({0}).build(ctx, prefix_v, cond_v);
    } else {
        auto kv = modules::LinearModule({
            hidden,
            2 * hidden,
            true,
            ggml_is_quantized(weights.kv.weight.tensor->type) ? GGML_PREC_DEFAULT : GGML_PREC_F32,
        }).build(ctx, context, weights.kv);
        k = modules::SliceModule({2, 0, hidden}).build(ctx, kv);
        v = modules::SliceModule({2, hidden, hidden}).build(ctx, kv);
    }
    q = modules::RMSNormModule({hidden, kNormEps, true, false}).build(ctx, q, {weights.q_norm, std::nullopt});
    k = modules::RMSNormModule({hidden, kNormEps, true, false}).build(ctx, k, {weights.k_norm, std::nullopt});
    auto attn = attention_core(ctx, q, k, v, std::nullopt, heads, head_dim);
    return apply_gate_and_out(ctx, x, attn, gates, weights.out, heads, head_dim, hidden);
}

AdaOutputs adaln(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & timestep_features,
    const DramaBoxAdaLayerNormWeights & weights,
    int64_t hidden) {
    auto embedded = modules::LinearModule({
        kTimestepFeatures,
        hidden,
        true,
        ggml_is_quantized(weights.timestep_linear_1.weight.tensor->type) ? GGML_PREC_DEFAULT : GGML_PREC_F32,
    }).build(ctx, timestep_features, weights.timestep_linear_1);
    embedded = modules::SiluModule{}.build(ctx, embedded);
    embedded = modules::LinearModule({
        hidden,
        hidden,
        true,
        ggml_is_quantized(weights.timestep_linear_2.weight.tensor->type) ? GGML_PREC_DEFAULT : GGML_PREC_F32,
    }).build(ctx, embedded, weights.timestep_linear_2);
    auto output = modules::SiluModule{}.build(ctx, embedded);
    output = modules::LinearModule({
        hidden,
        weights.output_coefficient * hidden,
        true,
        ggml_is_quantized(weights.output_linear.weight.tensor->type) ? GGML_PREC_DEFAULT : GGML_PREC_F32,
    }).build(ctx, output, weights.output_linear);
    return {output, embedded};
}

core::TensorValue build_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & context,
    const core::TensorValue & timestep,
    const core::TensorValue & prompt_timestep,
    const core::TensorValue & rope_cos,
    const core::TensorValue & rope_sin,
    const DramaBoxDitBlockWeights & weights,
    const DramaBoxTransformerConfig & config,
    int64_t ref_tokens,
    int64_t block_index,
    bool share_last_context,
    DramaBoxPerfMode perf_mode) {
    const int64_t hidden = config.hidden_size;
    auto norm = modules::AdaptiveLayerNormModule({hidden, kNormEps, modules::AdaptiveLayerNormMode::RmsAffine, 0, 1})
        .build(ctx, input, timestep, {weights.audio_scale_shift_table});
    auto self = self_attention(
        ctx,
        norm,
        rope_cos,
        rope_sin,
        ref_tokens,
        weights.self_attention,
        config,
        block_index == kStgSelfAttentionBlock && share_last_context,
        perf_mode);
    auto self_modulated = modules::AdaptiveLayerNormModule({hidden, kNormEps, modules::AdaptiveLayerNormMode::Scale, -1, 2})
        .build(ctx, self, timestep, {weights.audio_scale_shift_table});
    auto x = core::wrap_tensor(ggml_add(ctx.ggml, input.tensor, self_modulated.tensor), input.shape, GGML_TYPE_F32);

    auto q_in = modules::AdaptiveLayerNormModule({hidden, kNormEps, modules::AdaptiveLayerNormMode::RmsAffine, 6, 7})
        .build(ctx, x, timestep, {weights.audio_scale_shift_table});
    auto kv = modules::AdaptiveLayerNormModule({hidden, kNormEps, modules::AdaptiveLayerNormMode::Affine, 0, 1})
        .build(ctx, context, prompt_timestep, {weights.audio_prompt_scale_shift_table});
    auto cross = cross_attention(ctx, q_in, kv, weights.cross_attention, config, share_last_context);
    auto cross_modulated = modules::AdaptiveLayerNormModule({hidden, kNormEps, modules::AdaptiveLayerNormMode::Scale, -1, 8})
        .build(ctx, cross, timestep, {weights.audio_scale_shift_table});
    x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, cross_modulated.tensor), x.shape, GGML_TYPE_F32);

    auto ff_in = modules::AdaptiveLayerNormModule({hidden, kNormEps, modules::AdaptiveLayerNormMode::RmsAffine, 3, 4})
        .build(ctx, x, timestep, {weights.audio_scale_shift_table});
    auto ff = modules::FeedForwardGeluModule({hidden, hidden * 4, true}).build(ctx, ff_in, {
        weights.ff_in.weight,
        weights.ff_in.bias,
        weights.ff_out.weight,
        weights.ff_out.bias,
    });
    auto ff_modulated = modules::AdaptiveLayerNormModule({hidden, kNormEps, modules::AdaptiveLayerNormMode::Scale, -1, 5})
        .build(ctx, ff, timestep, {weights.audio_scale_shift_table});
    x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, ff_modulated.tensor), x.shape, GGML_TYPE_F32);
    return x;
}

}  // namespace

std::vector<float> make_dramabox_timestep_features(const std::vector<float> & timesteps) {
    std::vector<float> out;
    fill_dramabox_timestep_features(timesteps, out);
    return out;
}

void fill_dramabox_timestep_features(
    const std::vector<float> & timesteps,
    std::vector<float> & out) {
    out.resize(timesteps.size() * kTimestepFeatures);
    const int64_t half = kTimestepFeatures / 2;
    for (size_t row = 0; row < timesteps.size(); ++row) {
        for (int64_t i = 0; i < half; ++i) {
            const double exponent = -std::log(10000.0) * static_cast<double>(i) / static_cast<double>(half);
            const double value = static_cast<double>(timesteps[row]) * std::exp(exponent);
            out[row * kTimestepFeatures + static_cast<size_t>(i)] = static_cast<float>(std::cos(value));
            out[row * kTimestepFeatures + static_cast<size_t>(half + i)] = static_cast<float>(std::sin(value));
        }
    }
}

void make_dramabox_audio_rope(
    const std::vector<float> & positions,
    int64_t batch,
    int64_t tokens,
    const DramaBoxConfig & config,
    std::vector<float> & cos,
    std::vector<float> & sin) {
    const int64_t heads = config.transformer.num_attention_heads;
    const int64_t head_dim = config.transformer.attention_head_dim;
    const int64_t hidden = heads * head_dim;
    const int64_t half = head_dim / 2;
    const int64_t total_freqs = hidden / 2;
    const int64_t max_pos = config.transformer.positional_embedding_max_pos.empty()
        ? 20
        : config.transformer.positional_embedding_max_pos.front();
    cos.assign(static_cast<size_t>(batch * heads * tokens * half), 0.0F);
    sin.assign(cos.size(), 0.0F);
    std::vector<double> rope_index(static_cast<size_t>(total_freqs), 0.0);
    for (int64_t i = 0; i < total_freqs; ++i) {
        double lin = 0.0;
        if (total_freqs > 1) {
            lin = static_cast<double>(i) / static_cast<double>(total_freqs - 1);
        }
        rope_index[static_cast<size_t>(i)] = std::pow(10000.0, lin) * kPi / 2.0;
    }
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t t = 0; t < tokens; ++t) {
            const float start = positions[static_cast<size_t>((b * tokens + t) * 2)];
            const float end = positions[static_cast<size_t>((b * tokens + t) * 2 + 1)];
            const double fractional = (static_cast<double>(start) + static_cast<double>(end)) * 0.5 / static_cast<double>(max_pos);
            for (int64_t i = 0; i < total_freqs; ++i) {
                const double freq = rope_index[static_cast<size_t>(i)] * (fractional * 2.0 - 1.0);
                const float c = static_cast<float>(std::cos(freq));
                const float s = static_cast<float>(std::sin(freq));
                const int64_t h = i / half;
                const int64_t d = i % half;
                const size_t out_index = static_cast<size_t>(((b * heads + h) * tokens + t) * half + d);
                cos[out_index] = c;
                sin[out_index] = s;
            }
        }
    }
}

void make_dramabox_audio_rope_repeated(
    const std::vector<float> & positions,
    int64_t repeat_count,
    int64_t tokens,
    const DramaBoxConfig & config,
    std::vector<float> & cos,
    std::vector<float> & sin) {
    if (repeat_count <= 0 || tokens <= 0 ||
        static_cast<int64_t>(positions.size()) != tokens * 2) {
        throw std::runtime_error("DramaBox repeated RoPE shape mismatch");
    }
    make_dramabox_audio_rope(positions, 1, tokens, config, cos, sin);
    const size_t row_values = cos.size();
    cos.resize(static_cast<size_t>(repeat_count) * row_values);
    sin.resize(cos.size());
    for (int64_t b = 1; b < repeat_count; ++b) {
        std::copy_n(cos.data(), row_values, cos.data() + static_cast<std::ptrdiff_t>(b) * static_cast<std::ptrdiff_t>(row_values));
        std::copy_n(sin.data(), row_values, sin.data() + static_cast<std::ptrdiff_t>(b) * static_cast<std::ptrdiff_t>(row_values));
    }
}

DramaBoxDitWeights load_dramabox_dit_weights(
    const DramaBoxAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type,
    DramaBoxPerfMode perf_mode) {
    const auto & config = assets.config.transformer;
    const auto & source = *assets.dit_weights;
    DramaBoxDitWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "dramabox.dit.weights",
        weight_context_bytes == 0 ? kDitWeightContextBytes : weight_context_bytes);
    weights.patchify_proj = modules::binding::linear_from_source(
        *weights.store,
        source,
        "model.diffusion_model.audio_patchify_proj",
        weight_storage_type,
        config.hidden_size,
        config.in_channels,
        true);
    weights.adaln = load_adaln(
        *weights.store,
        source,
        "model.diffusion_model.audio_adaln_single",
        weight_storage_type,
        config.hidden_size,
        9);
    weights.prompt_adaln = load_adaln(
        *weights.store,
        source,
        "model.diffusion_model.audio_prompt_adaln_single",
        weight_storage_type,
        config.hidden_size,
        2);
    weights.blocks.reserve(static_cast<size_t>(config.num_layers));
    for (int64_t i = 0; i < config.num_layers; ++i) {
        weights.blocks.push_back(load_dit_transformer_block(*weights.store, source, i, config, weight_storage_type, perf_mode));
    }
    weights.output_scale_shift_table =
        weights.store->load_tensor(source, "model.diffusion_model.audio_scale_shift_table", assets::TensorStorageType::F32, {2, config.hidden_size});
    weights.output_proj = modules::binding::linear_from_source(
        *weights.store,
        source,
        "model.diffusion_model.audio_proj_out",
        weight_storage_type,
        config.out_channels,
        config.hidden_size,
        true);
    weights.store->upload();
    return weights;
}

class DramaBoxDitRuntime::Graph {
public:
    Graph(
        core::ExecutionContext & execution,
        std::shared_ptr<const DramaBoxAssets> assets,
        const DramaBoxDitWeights & weights,
        int64_t batch,
        int64_t tokens,
        int64_t context_tokens,
        bool share_stg_prefix,
        int64_t ref_tokens,
        DramaBoxPerfMode perf_mode)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          assets_(std::move(assets)),
          batch_(batch),
          tokens_(tokens),
          context_tokens_(context_tokens),
          share_stg_prefix_(share_stg_prefix),
          ref_tokens_(ref_tokens),
          perf_mode_(perf_mode),
          weights_(weights) {
        if (backend_ == nullptr) {
            throw std::runtime_error("DramaBox DiT backend initialization failed");
        }
        build();
    }

    ~Graph() {
        if (backend_ != nullptr && graph_ != nullptr) {
            core::release_backend_graph_resources(backend_type_, backend_, graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(
        int64_t batch,
        int64_t tokens,
        int64_t context_tokens,
        bool share_stg_prefix,
        int64_t ref_tokens,
        DramaBoxPerfMode perf_mode) const noexcept {
        return batch == batch_ && tokens == tokens_ && context_tokens == context_tokens_ &&
            share_stg_prefix == share_stg_prefix_ && ref_tokens == ref_tokens_ && perf_mode == perf_mode_;
    }

    void prepare_static_inputs(
        const DramaBoxConditioningEncoding & conditioning,
        const std::vector<float> & rope_cos,
        const std::vector<float> & rope_sin,
        const std::vector<float> & timestep_mask) const {
        if (conditioning.batch != batch_ ||
            conditioning.tokens != context_tokens_ ||
            conditioning.hidden_size != assets_->config.transformer.hidden_size) {
            throw std::runtime_error("DramaBox DiT static conditioning shape mismatch");
        }
        if (static_cast<int64_t>(rope_cos.size()) != rope_cos_.shape.num_elements() ||
            static_cast<int64_t>(rope_sin.size()) != rope_sin_.shape.num_elements()) {
            throw std::runtime_error("DramaBox DiT static RoPE shape mismatch");
        }
        if (static_cast<int64_t>(timestep_mask.size()) != timestep_mask_.shape.num_elements()) {
            throw std::runtime_error("DramaBox DiT timestep mask shape mismatch");
        }
        for (const float value : timestep_mask) {
            if (value != 0.0F && value != 1.0F) {
                throw std::runtime_error("DramaBox DiT graph timestep mask supports only binary denoise masks");
            }
        }
        const std::vector<float> zero_timestep_features = make_dramabox_timestep_features({0.0F});
        const auto input_start = Clock::now();
        core::write_tensor_f32(context_, conditioning.features);
        core::write_tensor_f32(rope_cos_, rope_cos);
        core::write_tensor_f32(rope_sin_, rope_sin);
        core::write_tensor_f32(timestep_mask_, timestep_mask);
        core::write_tensor_f32(zero_timestep_features_, zero_timestep_features);
        static_inputs_ready_ = true;
        debug::timing_log_scalar("dramabox.dit.static_input_upload_ms", debug::elapsed_ms(input_start, Clock::now()));
    }

    std::vector<float> forward(const DramaBoxDitInputs & inputs) const {
        const bool share_stg_prefix = inputs.stg_enabled && inputs.batch >= 2;
        if (!matches(
                inputs.batch,
                inputs.tokens,
                context_tokens_,
                share_stg_prefix,
                inputs.ref_tokens,
                perf_mode_)) {
            throw std::runtime_error("DramaBox DiT input shape does not match prepared graph");
        }
        if (!static_inputs_ready_) {
            throw std::runtime_error("DramaBox DiT static inputs were not prepared");
        }
        if (inputs.latent == nullptr || inputs.sigma_features == nullptr) {
            throw std::runtime_error("DramaBox DiT inputs are not bound");
        }
        core::write_tensor_f32(latent_, *inputs.latent);
        core::write_tensor_f32(sigma_features_, *inputs.sigma_features);
        core::set_backend_threads(backend_, threads_);
        const auto compute_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(backend_, graph_, nullptr, "dramabox.dit");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DramaBox DiT graph compute failed");
        }
        ggml_backend_synchronize(backend_);
        debug::timing_log_scalar("dramabox.dit.graph.compute_ms", debug::elapsed_ms(compute_start, Clock::now()));
        auto out = core::read_tensor_f32(output_);
        return out;
    }

private:
    void build() {
        const auto build_start = Clock::now();
        const auto & config = assets_->config;
        ggml_init_params params{kDitGraphContextBytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("DramaBox DiT ggml context initialization failed");
        }
        const int64_t hidden = config.transformer.hidden_size;
        const int64_t heads = config.transformer.num_attention_heads;
        latent_ = core::wrap_tensor(
            ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, config.transformer.in_channels, tokens_, 1),
            core::TensorShape::from_dims({1, tokens_, config.transformer.in_channels}),
            GGML_TYPE_F32);
        sigma_features_ = core::wrap_tensor(
            ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, kTimestepFeatures, 1, 1),
            core::TensorShape::from_dims({1, 1, kTimestepFeatures}),
            GGML_TYPE_F32);
        zero_timestep_features_ = core::wrap_tensor(
            ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, kTimestepFeatures, 1, 1),
            core::TensorShape::from_dims({1, 1, kTimestepFeatures}),
            GGML_TYPE_F32);
        timestep_mask_ = core::wrap_tensor(
            ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, 1, tokens_, 1),
            core::TensorShape::from_dims({1, tokens_, 1}),
            GGML_TYPE_F32);
        context_ = core::wrap_tensor(
            ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, hidden, context_tokens_, batch_),
            core::TensorShape::from_dims({batch_, context_tokens_, hidden}),
            GGML_TYPE_F32);
        rope_cos_ = core::wrap_tensor(
            ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F32, hidden / (2 * heads), tokens_, heads, batch_),
            core::TensorShape::from_dims({batch_, heads, tokens_, hidden / (2 * heads)}),
            GGML_TYPE_F32);
        rope_sin_ = core::wrap_tensor(
            ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F32, hidden / (2 * heads), tokens_, heads, batch_),
            core::TensorShape::from_dims({batch_, heads, tokens_, hidden / (2 * heads)}),
            GGML_TYPE_F32);
        ggml_set_input(latent_.tensor);
        ggml_set_input(sigma_features_.tensor);
        ggml_set_input(zero_timestep_features_.tensor);
        ggml_set_input(timestep_mask_.tensor);
        ggml_set_input(context_.tensor);
        ggml_set_input(rope_cos_.tensor);
        ggml_set_input(rope_sin_.tensor);
        ggml_set_output(zero_timestep_features_.tensor);
        ggml_set_output(timestep_mask_.tensor);
        ggml_set_output(context_.tensor);
        ggml_set_output(rope_cos_.tensor);
        ggml_set_output(rope_sin_.tensor);
        core::ModuleBuildContext build_ctx{ctx_.get(), "dramabox.dit", backend_type_};
        auto x = modules::LinearModule({
            config.transformer.in_channels,
            hidden,
            true,
            ggml_is_quantized(weights_.patchify_proj.weight.tensor->type) ? GGML_PREC_DEFAULT : GGML_PREC_F32,
        }).build(build_ctx, latent_, weights_.patchify_proj);
        if (batch_ > 1) {
            x = modules::RepeatModule({core::TensorShape::from_dims({batch_, tokens_, hidden})}).build(build_ctx, x);
        }
        const auto token_timestep_shape = core::TensorShape::from_dims({1, tokens_, kTimestepFeatures});
        const auto sigma_token_features = modules::RepeatModule({token_timestep_shape}).build(build_ctx, sigma_features_);
        const auto zero_token_features = modules::RepeatModule({token_timestep_shape}).build(build_ctx, zero_timestep_features_);
        const auto timestep_mask_features = modules::RepeatModule({token_timestep_shape}).build(build_ctx, timestep_mask_);
        const auto inverse_mask = core::wrap_tensor(
            ggml_scale_bias(build_ctx.ggml, timestep_mask_features.tensor, -1.0F, 1.0F),
            timestep_mask_features.shape,
            GGML_TYPE_F32);
        const auto masked_sigma = core::wrap_tensor(
            ggml_mul(build_ctx.ggml, sigma_token_features.tensor, timestep_mask_features.tensor),
            token_timestep_shape,
            GGML_TYPE_F32);
        const auto masked_zero = core::wrap_tensor(
            ggml_mul(build_ctx.ggml, zero_token_features.tensor, inverse_mask.tensor),
            token_timestep_shape,
            GGML_TYPE_F32);
        auto timestep_features = core::wrap_tensor(
            ggml_add(build_ctx.ggml, masked_sigma.tensor, masked_zero.tensor),
            token_timestep_shape,
            GGML_TYPE_F32);
        auto timestep_outputs_one = adaln(build_ctx, timestep_features, weights_.adaln, hidden);
        auto prompt_timestep_outputs_one = adaln(build_ctx, sigma_features_, weights_.prompt_adaln, hidden);
        AdaOutputs timestep_outputs{
            timestep_outputs_one.modulation,
            timestep_outputs_one.embedding,
        };
        AdaOutputs prompt_timestep_outputs{
            prompt_timestep_outputs_one.modulation,
            prompt_timestep_outputs_one.embedding,
        };
        if (share_stg_prefix_) {
            const int64_t prefix_batch = batch_ - 1;
            auto prefix_x = modules::SliceModule({0, 0, prefix_batch}).build(build_ctx, x);
            auto prefix_context = modules::SliceModule({0, 0, prefix_batch}).build(build_ctx, context_);
            auto prefix_rope_cos = modules::SliceModule({0, 0, prefix_batch}).build(build_ctx, rope_cos_);
            auto prefix_rope_sin = modules::SliceModule({0, 0, prefix_batch}).build(build_ctx, rope_sin_);
            const int64_t layer_count = static_cast<int64_t>(weights_.blocks.size());
            for (int64_t i = 0; i < std::min<int64_t>(kStgSelfAttentionBlock, layer_count); ++i) {
                prefix_x = build_block(
                    build_ctx,
                    prefix_x,
                    prefix_context,
                    timestep_outputs.modulation,
                    prompt_timestep_outputs.modulation,
                    prefix_rope_cos,
                    prefix_rope_sin,
                    weights_.blocks[static_cast<size_t>(i)],
                    config.transformer,
                    ref_tokens_,
                    i,
                    false,
                    perf_mode_);
            }
            auto cond_prefix = modules::SliceModule({0, 0, 1}).build(build_ctx, prefix_x);
            x = modules::ConcatModule({0}).build(build_ctx, prefix_x, cond_prefix);
            for (int64_t i = kStgSelfAttentionBlock; i < layer_count; ++i) {
                x = build_block(
                    build_ctx,
                    x,
                    context_,
                    timestep_outputs.modulation,
                    prompt_timestep_outputs.modulation,
                    rope_cos_,
                    rope_sin_,
                    weights_.blocks[static_cast<size_t>(i)],
                    config.transformer,
                    ref_tokens_,
                    i,
                    true,
                    perf_mode_);
            }
        } else {
            for (int64_t i = 0; i < static_cast<int64_t>(weights_.blocks.size()); ++i) {
                x = build_block(
                    build_ctx,
                    x,
                    context_,
                    timestep_outputs.modulation,
                    prompt_timestep_outputs.modulation,
                    rope_cos_,
                    rope_sin_,
                    weights_.blocks[static_cast<size_t>(i)],
                    config.transformer,
                    ref_tokens_,
                    i,
                    false,
                    perf_mode_);
            }
        }
        auto output_shift_table = modules::SliceModule({0, 0, 1}).build(build_ctx, weights_.output_scale_shift_table);
        output_shift_table = core::reshape_tensor(build_ctx, output_shift_table, core::TensorShape::from_dims({1, 1, hidden}));
        auto output_scale_table = modules::SliceModule({0, 1, 1}).build(build_ctx, weights_.output_scale_shift_table);
        output_scale_table = core::reshape_tensor(build_ctx, output_scale_table, core::TensorShape::from_dims({1, 1, hidden}));
        auto shift = core::wrap_tensor(
            ggml_add(build_ctx.ggml, timestep_outputs.embedding.tensor, output_shift_table.tensor),
            timestep_outputs.embedding.shape,
            GGML_TYPE_F32);
        auto scale = core::wrap_tensor(
            ggml_add(build_ctx.ggml, timestep_outputs.embedding.tensor, output_scale_table.tensor),
            timestep_outputs.embedding.shape,
            GGML_TYPE_F32);
        x = modules::LayerNormModule({hidden, kNormEps, false, false}).build(build_ctx, x, {});
        auto one_plus_scale = core::wrap_tensor(
            ggml_scale_bias(build_ctx.ggml, scale.tensor, 1.0F, 1.0F),
            scale.shape,
            GGML_TYPE_F32);
        auto scaled = core::wrap_tensor(ggml_mul(build_ctx.ggml, x.tensor, one_plus_scale.tensor), x.shape, GGML_TYPE_F32);
        x = core::wrap_tensor(ggml_add(build_ctx.ggml, scaled.tensor, shift.tensor), scaled.shape, GGML_TYPE_F32);
        x = modules::LinearModule({
            hidden,
            config.transformer.out_channels,
            true,
            ggml_is_quantized(weights_.output_proj.weight.tensor->type) ? GGML_PREC_DEFAULT : GGML_PREC_F32,
        }).build(build_ctx, x, weights_.output_proj);
        output_ = core::ensure_backend_addressable_layout(build_ctx, x).tensor;
        ggml_set_output(output_);
        const auto expand_start = Clock::now();
        graph_ = ggml_new_graph_custom(ctx_.get(), kDitGraphNodeCapacity, false);
        ggml_build_forward_expand(graph_, output_);
        debug::timing_log_scalar("dramabox.dit.graph.expand_ms", debug::elapsed_ms(expand_start, Clock::now()));
        debug::trace_log_scalar("dramabox.dit.graph.nodes", static_cast<int64_t>(ggml_graph_n_nodes(graph_)));
        const auto alloc_start = Clock::now();
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("DramaBox DiT backend buffer allocation failed");
        }
        debug::timing_log_scalar("dramabox.dit.graph.alloc_ms", debug::elapsed_ms(alloc_start, Clock::now()));
        debug::timing_log_scalar("dramabox.dit.graph.build_ms", debug::elapsed_ms(build_start, Clock::now()));
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const DramaBoxAssets> assets_;
    int64_t batch_ = 0;
    int64_t tokens_ = 0;
    int64_t context_tokens_ = 0;
    bool share_stg_prefix_ = false;
    int64_t ref_tokens_ = 0;
    DramaBoxPerfMode perf_mode_ = DramaBoxPerfMode::Exact;
    const DramaBoxDitWeights & weights_;
    mutable bool static_inputs_ready_ = false;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue latent_;
    core::TensorValue sigma_features_;
    core::TensorValue zero_timestep_features_;
    core::TensorValue timestep_mask_;
    core::TensorValue context_;
    core::TensorValue rope_cos_;
    core::TensorValue rope_sin_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

DramaBoxDitRuntime::DramaBoxDitRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const DramaBoxAssets> assets,
    assets::TensorStorageType weight_storage_type,
    DramaBoxPerfMode perf_mode)
    : execution_(&execution),
      assets_(std::move(assets)),
      weight_storage_type_(weight_storage_type),
      perf_mode_(perf_mode) {
    if (execution_ == nullptr) {
        throw std::runtime_error("DramaBox DiT runtime requires execution context");
    }
    if (assets_ == nullptr) {
        throw std::runtime_error("DramaBox DiT runtime requires assets");
    }
}

DramaBoxDitRuntime::~DramaBoxDitRuntime() = default;

void DramaBoxDitRuntime::prepare(
    int64_t batch,
    int64_t tokens,
    int64_t context_tokens,
    bool stg_enabled,
    int64_t ref_tokens) const {
    if (!weights_) {
        weights_ = std::make_unique<DramaBoxDitWeights>(load_dramabox_dit_weights(
            *assets_,
            execution_->backend(),
            execution_->backend_type(),
            0,
            weight_storage_type_,
            perf_mode_));
        assets_->dit_weights->release_storage();
    }
    const bool share_stg_prefix = stg_enabled && batch >= 2;
    if (!graph_ || !graph_->matches(batch, tokens, context_tokens, share_stg_prefix, ref_tokens, perf_mode_)) {
        graph_.reset();
        graph_ = std::make_unique<Graph>(
            *execution_,
            assets_,
            *weights_,
            batch,
            tokens,
            context_tokens,
            share_stg_prefix,
            ref_tokens,
            perf_mode_);
    }
}

void DramaBoxDitRuntime::prepare_static_inputs(
    int64_t batch,
    int64_t tokens,
    bool stg_enabled,
    int64_t ref_tokens,
    const DramaBoxConditioningEncoding & conditioning,
    const std::vector<float> & rope_cos,
    const std::vector<float> & rope_sin,
    const std::vector<float> & timestep_mask) const {
    prepare(
        batch,
        tokens,
        conditioning.tokens,
        stg_enabled,
        ref_tokens);
    graph_->prepare_static_inputs(conditioning, rope_cos, rope_sin, timestep_mask);
}

std::vector<float> DramaBoxDitRuntime::forward(const DramaBoxDitInputs & inputs) const {
    if (!graph_) {
        throw std::runtime_error("DramaBox DiT graph was not prepared");
    }
    return graph_->forward(inputs);
}

void DramaBoxDitRuntime::release_runtime_state() const {
    graph_.reset();
    weights_.reset();
}

}  // namespace engine::models::dramabox
