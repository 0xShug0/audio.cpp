#include "engine/models/heartmula/mula.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/kv_cache.h"

#include "engine/framework/core/constant_tensor_cache.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::heartmula {
namespace {

namespace binding = modules::binding;

constexpr float kLlama3LowFreqFactor = 1.0F;
constexpr float kLlama3HighFreqFactor = 4.0F;
constexpr float kLlama3OldContextLength = 8192.0F;
constexpr float kPi = 3.14159265358979323846F;
constexpr int64_t kInitialBackboneGeneratedCacheFrames = 128;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

void require_assets(const HeartMuLaAssets & assets) {
    if (assets.mula_weights == nullptr) {
        throw std::runtime_error("HeartMuLa LM requires model weights");
    }
}

int64_t require_head_dim(const HeartMuLaTransformerConfig & config) {
    if (config.embed_dim <= 0 || config.intermediate_dim <= 0) {
        throw std::runtime_error("HeartMuLa transformer hidden sizes must be positive");
    }
    if (config.num_heads <= 0 || config.num_kv_heads <= 0 || config.head_dim <= 0) {
        throw std::runtime_error("HeartMuLa transformer attention dimensions must be positive");
    }
    if (config.num_heads % config.num_kv_heads != 0) {
        throw std::runtime_error("HeartMuLa transformer attention heads must be divisible by key/value heads");
    }
    if (config.num_heads * config.head_dim != config.embed_dim) {
        throw std::runtime_error("HeartMuLa transformer embed_dim must equal attention heads times head_dim");
    }
    return config.head_dim;
}

HeartMuLaTransformerLayerWeights load_transformer_layer_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const HeartMuLaTransformerConfig & config,
    const std::string & prefix,
    assets::TensorStorageType weight_storage_type) {
    const int64_t dim = require_head_dim(config);
    HeartMuLaTransformerLayerWeights weights;
    weights.sa_norm = source.require_f32_tensor(prefix + ".sa_norm.scale", {config.embed_dim});
    weights.q_proj = store.load_tensor(
        source,
        prefix + ".attn.q_proj.weight",
        weight_storage_type,
        {config.num_heads * dim, config.embed_dim});
    weights.k_proj = store.load_tensor(
        source,
        prefix + ".attn.k_proj.weight",
        weight_storage_type,
        {config.num_kv_heads * dim, config.embed_dim});
    weights.v_proj = store.load_tensor(
        source,
        prefix + ".attn.v_proj.weight",
        weight_storage_type,
        {config.num_kv_heads * dim, config.embed_dim});
    weights.output_proj = store.load_tensor(
        source,
        prefix + ".attn.output_proj.weight",
        weight_storage_type,
        {config.embed_dim, config.num_heads * dim});
    weights.mlp_norm = source.require_f32_tensor(prefix + ".mlp_norm.scale", {config.embed_dim});
    weights.w1 = store.load_tensor(
        source,
        prefix + ".mlp.w1.weight",
        weight_storage_type,
        {config.intermediate_dim, config.embed_dim});
    weights.w2 = store.load_tensor(
        source,
        prefix + ".mlp.w2.weight",
        weight_storage_type,
        {config.embed_dim, config.intermediate_dim});
    weights.w3 = store.load_tensor(
        source,
        prefix + ".mlp.w3.weight",
        weight_storage_type,
        {config.intermediate_dim, config.embed_dim});
    return weights;
}

HeartMuLaTransformerWeights load_transformer_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const HeartMuLaTransformerConfig & config,
    const std::string & prefix,
    assets::TensorStorageType weight_storage_type) {
    HeartMuLaTransformerWeights weights;
    weights.layers.reserve(static_cast<size_t>(config.num_layers));
    for (int64_t layer = 0; layer < config.num_layers; ++layer) {
        weights.layers.push_back(load_transformer_layer_weights(
            store,
            source,
            config,
            prefix + ".layers." + std::to_string(layer),
            weight_storage_type));
    }
    weights.norm = source.require_f32_tensor(prefix + ".norm.scale", {config.embed_dim});
    return weights;
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

core::TensorValue repeat_kv_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t repeats) {
    if (repeats == 1) {
        return input;
    }
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    const int64_t batch = contiguous.shape.dims[0];
    const int64_t kv_heads = contiguous.shape.dims[1];
    const int64_t steps = contiguous.shape.dims[2];
    const int64_t dim = contiguous.shape.dims[3];
    auto expanded = core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({batch, kv_heads, 1, steps * dim}));
    expanded = modules::RepeatModule({core::TensorShape::from_dims({batch, kv_heads, repeats, steps * dim})})
                   .build(ctx, expanded);
    expanded = core::ensure_backend_addressable_layout(ctx, expanded);
    return core::reshape_tensor(
        ctx,
        expanded,
        core::TensorShape::from_dims({batch, kv_heads * repeats, steps, dim}));
}

core::TensorValue attention_from_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const std::optional<core::TensorValue> & attention_mask) {
    const modules::MatMulModule matmul;
    auto scores = matmul.build(
        ctx,
        q_heads,
        modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    core::TensorValue attn;
    if (attention_mask.has_value()) {
        scores = core::ensure_backend_addressable_layout(ctx, scores);
        attn = core::wrap_tensor(
            ggml_soft_max_ext(
                ctx.ggml,
                scores.tensor,
                attention_mask->tensor,
                1.0F / std::sqrt(static_cast<float>(dim)),
                0.0F),
            scores.shape,
            GGML_TYPE_F32);
    } else {
        scores = core::wrap_tensor(
            ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(static_cast<float>(dim))),
            scores.shape,
            GGML_TYPE_F32);
        scores = core::wrap_tensor(ggml_diag_mask_inf(ctx.ggml, scores.tensor, 0), scores.shape, GGML_TYPE_F32);
        scores = core::ensure_backend_addressable_layout(ctx, scores);
        attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, scores.tensor), scores.shape, GGML_TYPE_F32);
    }
    return matmul.build(ctx, attn, v_heads);
}

core::TensorValue cache_view(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    int64_t start,
    int64_t steps,
    int64_t heads,
    int64_t dim) {
    if (start < 0 || steps <= 0 || start + steps > cache.shape.dims[1]) {
        throw std::runtime_error("HeartMuLa cache view range is invalid");
    }
    return core::wrap_tensor(
        ggml_view_4d(
            ctx.ggml,
            cache.tensor,
            dim,
            heads,
            steps,
            1,
            cache.tensor->nb[1],
            cache.tensor->nb[2],
            cache.tensor->nb[3],
            static_cast<size_t>(start) * cache.tensor->nb[2]),
        core::TensorShape::from_dims({1, steps, heads, dim}),
        GGML_TYPE_F32);
}

core::TensorValue cache_heads_view(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache) {
    core::validate_rank_between(cache, 4, 4, "cache");
    if (cache.type != GGML_TYPE_F32) {
        throw std::runtime_error("HeartMuLa cache heads view requires f32 cache");
    }
    const int64_t steps = cache.shape.dims[1];
    const int64_t heads = cache.shape.dims[2];
    const int64_t dim = cache.shape.dims[3];
    auto * view = ggml_view_4d(
        ctx.ggml,
        cache.tensor,
        dim,
        steps,
        heads,
        1,
        cache.tensor->nb[2],
        cache.tensor->nb[1],
        cache.tensor->nb[3],
        0);
    return core::wrap_tensor(
        view,
        core::TensorShape::from_dims({1, heads, steps, dim}),
        GGML_TYPE_F32);
}

core::TensorValue flash_attention_from_grouped_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const core::TensorValue & attention_mask) {
    if (!core::has_backend_addressable_layout(q_heads.tensor)) {
        throw std::runtime_error("HeartMuLa flash attention expects contiguous Q heads");
    }
    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q_heads.tensor,
        k_heads.tensor,
        v_heads.tensor,
        attention_mask.tensor,
        1.0F / std::sqrt(static_cast<float>(dim)),
        0.0F,
        0.0F);
    ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
    return core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({q_heads.shape.dims[0], q_heads.shape.dims[2], q_heads.shape.dims[1], dim}),
        GGML_TYPE_F32);
}

core::TensorValue apply_scaled_llama3_rope(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const HeartMuLaTransformerConfig & config,
    const core::TensorValue & rope_factors) {
    const int64_t dim = require_head_dim(config);
    return core::wrap_tensor(
        ggml_rope_ext(
            ctx.ggml,
            input.tensor,
            positions.tensor,
            rope_factors.tensor,
            static_cast<int>(dim),
            GGML_ROPE_TYPE_NORMAL,
            static_cast<int>(kLlama3OldContextLength),
            config.rope_base,
            1.0F,
            0.0F,
            1.0F,
            0.0F,
            0.0F),
        input.shape,
        input.type);
}

HeartMuLaTransformerLayerOutputs build_heartmula_transformer_layer_static_cache_tail(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const HeartMuLaTransformerLayerWeights & weights,
    const HeartMuLaTransformerConfig & config,
    const core::TensorValue & rope_factors,
    core::ConstantTensorCache & constants,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const core::TensorValue & attention_mask) {
    const int64_t dim = require_head_dim(config);
    const int64_t scratch_slot = cache_key.shape.dims[1] - 1;
    const modules::LinearModule q_proj(binding::linear_config(config.embed_dim, config.num_heads * dim, false));
    const modules::LinearModule k_proj(binding::linear_config(config.embed_dim, config.num_kv_heads * dim, false));
    const modules::LinearModule v_proj(binding::linear_config(config.embed_dim, config.num_kv_heads * dim, false));
    const modules::LinearModule output_proj(binding::linear_config(config.num_heads * dim, config.embed_dim, false));
    const modules::RMSNormModule hidden_norm({config.embed_dim, config.norm_eps, true, false});
    const modules::AddModule add;

    auto attn_in = hidden_norm.build(ctx, input, binding::norm_data(constants, weights.sa_norm));
    auto q = q_proj.build(ctx, attn_in, binding::linear_data(constants, weights.q_proj));
    auto k = k_proj.build(ctx, attn_in, binding::linear_data(constants, weights.k_proj));
    auto v = v_proj.build(ctx, attn_in, binding::linear_data(constants, weights.v_proj));
    q = apply_scaled_llama3_rope(ctx, reshape_heads(ctx, q, config.num_heads, dim), positions, config, rope_factors);
    k = apply_scaled_llama3_rope(ctx, reshape_heads(ctx, k, config.num_kv_heads, dim), positions, config, rope_factors);
    v = reshape_heads(ctx, v, config.num_kv_heads, dim);

    const int64_t batch = input.shape.dims[0];
    auto flat_k = core::reshape_tensor(ctx, k, core::TensorShape::from_dims({1, 1, batch * config.num_kv_heads, dim}));
    auto flat_v = core::reshape_tensor(ctx, v, core::TensorShape::from_dims({1, 1, batch * config.num_kv_heads, dim}));
    auto key_tail = cache_view(ctx, cache_key, scratch_slot, 1, batch * config.num_kv_heads, dim);
    auto value_tail = cache_view(ctx, cache_value, scratch_slot, 1, batch * config.num_kv_heads, dim);
    ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, flat_k.tensor, key_tail.tensor));
    ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, flat_v.tensor, value_tail.tensor));
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    q_heads = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, q_heads),
        core::TensorShape::from_dims({1, batch * config.num_heads, input.shape.dims[1], dim}));
    q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    auto k_heads = cache_heads_view(ctx, cache_key);
    auto v_heads = cache_heads_view(ctx, cache_value);
    auto context = flash_attention_from_grouped_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
    context = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, context),
        core::TensorShape::from_dims({batch, config.num_heads, input.shape.dims[1], dim}));
    context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.num_heads * dim}));
    auto x = add.build(ctx, input, output_proj.build(ctx, context, binding::linear_data(constants, weights.output_proj)));

    auto ff_in = hidden_norm.build(ctx, x, binding::norm_data(constants, weights.mlp_norm));
    auto gate = modules::LinearModule(binding::linear_config(config.embed_dim, config.intermediate_dim, false))
                    .build(ctx, ff_in, binding::linear_data(constants, weights.w1));
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule(binding::linear_config(config.embed_dim, config.intermediate_dim, false))
                  .build(ctx, ff_in, binding::linear_data(constants, weights.w3));
    auto gated = modules::MulModule{}.build(ctx, gate, up);
    auto ff = modules::LinearModule(binding::linear_config(config.intermediate_dim, config.embed_dim, false))
                  .build(ctx, gated, binding::linear_data(constants, weights.w2));
    return {add.build(ctx, x, ff), flat_k, flat_v};
}

core::TensorValue build_heartmula_transformer_layer_set_rows_tail(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const HeartMuLaTransformerLayerWeights & weights,
    const HeartMuLaTransformerConfig & config,
    const core::TensorValue & rope_factors,
    core::ConstantTensorCache & constants,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const core::TensorValue & cache_slot,
    const core::TensorValue & attention_mask) {
    const int64_t dim = require_head_dim(config);
    const int64_t kv_repeats = config.num_heads / config.num_kv_heads;
    const modules::LinearModule q_proj(binding::linear_config(config.embed_dim, config.num_heads * dim, false));
    const modules::LinearModule k_proj(binding::linear_config(config.embed_dim, config.num_kv_heads * dim, false));
    const modules::LinearModule v_proj(binding::linear_config(config.embed_dim, config.num_kv_heads * dim, false));
    const modules::LinearModule output_proj(binding::linear_config(config.num_heads * dim, config.embed_dim, false));
    const modules::RMSNormModule hidden_norm({config.embed_dim, config.norm_eps, true, false});
    const modules::AddModule add;

    auto attn_in = hidden_norm.build(ctx, input, binding::norm_data(constants, weights.sa_norm));
    auto q = q_proj.build(ctx, attn_in, binding::linear_data(constants, weights.q_proj));
    auto k = k_proj.build(ctx, attn_in, binding::linear_data(constants, weights.k_proj));
    auto v = v_proj.build(ctx, attn_in, binding::linear_data(constants, weights.v_proj));
    q = apply_scaled_llama3_rope(ctx, reshape_heads(ctx, q, config.num_heads, dim), positions, config, rope_factors);
    k = apply_scaled_llama3_rope(ctx, reshape_heads(ctx, k, config.num_kv_heads, dim), positions, config, rope_factors);
    v = reshape_heads(ctx, v, config.num_kv_heads, dim);

    const modules::FastKVSetRowsModule set_rows;
    const int64_t batch = input.shape.dims[0];
    auto flat_k = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, k),
        core::TensorShape::from_dims({1, 1, batch * config.num_kv_heads, dim}));
    auto flat_v = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, v),
        core::TensorShape::from_dims({1, 1, batch * config.num_kv_heads, dim}));
    auto updated_cache_key = set_rows.build(ctx, cache_key, flat_k, cache_slot);
    auto updated_cache_value = set_rows.build(ctx, cache_value, flat_v, cache_slot);
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    q_heads = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, q_heads),
        core::TensorShape::from_dims({1, batch * config.num_heads, input.shape.dims[1], dim}));
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, updated_cache_key.shape.rank}).build(ctx, updated_cache_key);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, updated_cache_value.shape.rank}).build(ctx, updated_cache_value);
    auto context = attention_from_heads(
        ctx,
        q_heads,
        repeat_kv_heads(ctx, k_heads, kv_repeats),
        repeat_kv_heads(ctx, v_heads, kv_repeats),
        dim,
        attention_mask);
    context = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, context),
        core::TensorShape::from_dims({batch, config.num_heads, input.shape.dims[1], dim}));
    context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.num_heads * dim}));
    auto x = add.build(ctx, input, output_proj.build(ctx, context, binding::linear_data(constants, weights.output_proj)));

    auto ff_in = hidden_norm.build(ctx, x, binding::norm_data(constants, weights.mlp_norm));
    auto gate = modules::LinearModule(binding::linear_config(config.embed_dim, config.intermediate_dim, false))
                    .build(ctx, ff_in, binding::linear_data(constants, weights.w1));
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule(binding::linear_config(config.embed_dim, config.intermediate_dim, false))
                  .build(ctx, ff_in, binding::linear_data(constants, weights.w3));
    auto gated = modules::MulModule{}.build(ctx, gate, up);
    auto ff = modules::LinearModule(binding::linear_config(config.intermediate_dim, config.embed_dim, false))
                  .build(ctx, gated, binding::linear_data(constants, weights.w2));
    return add.build(ctx, x, ff);
}

std::vector<float> flatten_prefill_cache_by_step(
    const std::vector<float> & values,
    int64_t batch,
    int64_t steps,
    int64_t heads,
    int64_t dim) {
    const size_t expected = static_cast<size_t>(batch * steps * heads * dim);
    if (values.size() != expected) {
        throw std::runtime_error("HeartMuLa backbone prefill cache tensor size mismatch");
    }
    std::vector<float> out(expected);
    for (int64_t step = 0; step < steps; ++step) {
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t head = 0; head < heads; ++head) {
                const size_t src = static_cast<size_t>(((b * steps + step) * heads + head) * dim);
                const size_t dst = static_cast<size_t>(((step * batch + b) * heads + head) * dim);
                std::copy(values.begin() + static_cast<ptrdiff_t>(src),
                          values.begin() + static_cast<ptrdiff_t>(src + dim),
                          out.begin() + static_cast<ptrdiff_t>(dst));
            }
        }
    }
    return out;
}

core::TensorValue codebook_audio_logits(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden,
    const core::TensorValue & audio_head,
    const core::TensorValue & codebook_index,
    const HeartMuLaConfig & config) {
    const auto flat_hidden = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, hidden),
        core::TensorShape::from_dims({hidden.shape.dims[0], hidden.shape.dims[2]}));
    const auto flat_audio_head = core::reshape_tensor(
        ctx,
        audio_head,
        core::TensorShape::from_dims({
            config.audio_num_codebooks - 1,
            config.decoder.embed_dim * config.audio_vocab_size,
        }));
    auto selected_head = core::wrap_tensor(
        ggml_get_rows(ctx.ggml, flat_audio_head.tensor, codebook_index.tensor),
        core::TensorShape::from_dims({1, config.decoder.embed_dim * config.audio_vocab_size}),
        audio_head.type);
    selected_head = core::reshape_tensor(
        ctx,
        selected_head,
        core::TensorShape::from_dims({config.decoder.embed_dim, config.audio_vocab_size}));
    return modules::MatMulModule{}.build(ctx, flat_hidden, selected_head);
}

std::vector<float> llama3_scaled_rope_factors(const HeartMuLaTransformerConfig & config) {
    const int64_t head_dim = require_head_dim(config);
    if (head_dim % 2 != 0) {
        throw std::runtime_error("HeartMuLa Llama3 scaled RoPE requires even head_dim");
    }
    std::vector<float> factors;
    factors.reserve(static_cast<size_t>(head_dim / 2));
    const float low_freq_wavelen = kLlama3OldContextLength / kLlama3LowFreqFactor;
    const float high_freq_wavelen = kLlama3OldContextLength / kLlama3HighFreqFactor;
    for (int64_t index = 0; index < head_dim; index += 2) {
        const float freq = std::pow(config.rope_base, -static_cast<float>(index) / static_cast<float>(head_dim));
        const float wavelen = 2.0F * kPi / freq;
        float scaled_freq = freq;
        if (wavelen > low_freq_wavelen) {
            scaled_freq = freq / config.rope_scale_factor;
        } else if (wavelen >= high_freq_wavelen) {
            const float smooth = (kLlama3OldContextLength / wavelen - kLlama3LowFreqFactor) /
                (kLlama3HighFreqFactor - kLlama3LowFreqFactor);
            scaled_freq = (1.0F - smooth) * freq / config.rope_scale_factor + smooth * freq;
        }
        factors.push_back(freq / scaled_freq);
    }
    return factors;
}

}  // namespace

HeartMuLaTransformerLayerOutputs build_heartmula_transformer_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const HeartMuLaTransformerLayerWeights & weights,
    const HeartMuLaTransformerConfig & config,
    const core::TensorValue & rope_factors,
    core::ConstantTensorCache & constants,
    const std::optional<core::TensorValue> & prefix_key,
    const std::optional<core::TensorValue> & prefix_value,
    const std::optional<core::TensorValue> & attention_mask) {
    const int64_t dim = require_head_dim(config);
    const int64_t kv_repeats = config.num_heads / config.num_kv_heads;
    const modules::LinearModule q_proj(binding::linear_config(config.embed_dim, config.num_heads * dim, false));
    const modules::LinearModule k_proj(binding::linear_config(config.embed_dim, config.num_kv_heads * dim, false));
    const modules::LinearModule v_proj(binding::linear_config(config.embed_dim, config.num_kv_heads * dim, false));
    const modules::LinearModule output_proj(binding::linear_config(config.num_heads * dim, config.embed_dim, false));
    const modules::RMSNormModule hidden_norm({config.embed_dim, config.norm_eps, true, false});
    const modules::AddModule add;

    auto attn_in = hidden_norm.build(ctx, input, binding::norm_data(constants, weights.sa_norm));
    auto q = q_proj.build(ctx, attn_in, binding::linear_data(constants, weights.q_proj));
    auto k = k_proj.build(ctx, attn_in, binding::linear_data(constants, weights.k_proj));
    auto v = v_proj.build(ctx, attn_in, binding::linear_data(constants, weights.v_proj));
    q = apply_scaled_llama3_rope(ctx, reshape_heads(ctx, q, config.num_heads, dim), positions, config, rope_factors);
    k = apply_scaled_llama3_rope(ctx, reshape_heads(ctx, k, config.num_kv_heads, dim), positions, config, rope_factors);
    v = reshape_heads(ctx, v, config.num_kv_heads, dim);
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto all_k = prefix_key.has_value() ? modules::ConcatModule({1}).build(ctx, *prefix_key, k) : k;
    auto all_v = prefix_value.has_value() ? modules::ConcatModule({1}).build(ctx, *prefix_value, v) : v;
    auto k_heads = repeat_kv_heads(ctx, modules::TransposeModule({{0, 2, 1, 3}, all_k.shape.rank}).build(ctx, all_k), kv_repeats);
    auto v_heads = repeat_kv_heads(ctx, modules::TransposeModule({{0, 2, 1, 3}, all_v.shape.rank}).build(ctx, all_v), kv_repeats);
    auto context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
    context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.num_heads * dim}));
    auto x = add.build(ctx, input, output_proj.build(ctx, context, binding::linear_data(constants, weights.output_proj)));

    auto ff_in = hidden_norm.build(ctx, x, binding::norm_data(constants, weights.mlp_norm));
    auto gate = modules::LinearModule(binding::linear_config(config.embed_dim, config.intermediate_dim, false))
                    .build(ctx, ff_in, binding::linear_data(constants, weights.w1));
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule(binding::linear_config(config.embed_dim, config.intermediate_dim, false))
                  .build(ctx, ff_in, binding::linear_data(constants, weights.w3));
    auto gated = modules::MulModule{}.build(ctx, gate, up);
    auto ff = modules::LinearModule(binding::linear_config(config.intermediate_dim, config.embed_dim, false))
                  .build(ctx, gated, binding::linear_data(constants, weights.w2));
    return {add.build(ctx, x, ff), k, v};
}

class HeartMuLaBackbonePrefillGraph {
public:
    HeartMuLaBackbonePrefillGraph(
        const HeartMuLaWeightsRuntime & runtime,
        int64_t batch_size,
        int64_t steps)
        : runtime_(&runtime),
          batch_size_(batch_size),
          steps_(steps) {
        if (batch_size_ <= 0 || steps_ <= 0) {
            throw std::runtime_error("HeartMuLa backbone prefill graph shape is invalid");
        }
        const auto & config = runtime_->assets().mula_config.backbone;
        const auto & weights = runtime_->weights();
        const int64_t head_dim = require_head_dim(config);
        ggml_init_params params{runtime_->backbone_prefill_graph_arena_bytes(), nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize HeartMuLa backbone prefill graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "heartmula.backbone.prefill"};
        auto x = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({batch_size_, steps_, config.embed_dim}));
        input_ = x.tensor;
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, steps_);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({steps_}), GGML_TYPE_I32);

        auto & constants = runtime_->backbone_constants();
        constants.begin_graph();
        for (const auto & layer : weights.backbone.layers) {
            auto layer_out = build_heartmula_transformer_layer(
                ctx,
                x,
                positions,
                layer,
                config,
                weights.backbone_rope_factors,
                constants);
            x = layer_out.output;
            auto * key_readback = ggml_cpy(
                ctx_.get(),
                layer_out.key.tensor,
                ggml_dup_tensor(ctx_.get(), layer_out.key.tensor));
            auto * value_readback = ggml_cpy(
                ctx_.get(),
                layer_out.value.tensor,
                ggml_dup_tensor(ctx_.get(), layer_out.value.tensor));
            keys_.push_back(key_readback);
            values_.push_back(value_readback);
            readback_outputs_.push_back(key_readback);
            readback_outputs_.push_back(value_readback);
        }
        x = modules::SliceModule({1, steps_ - 1, 1}).build(ctx, x);
        x = modules::RMSNormModule({config.embed_dim, config.norm_eps, true, false})
                .build(ctx, x, binding::norm_data(constants, weights.backbone.norm));
        hidden_output_ = ggml_cpy(ctx_.get(), x.tensor, ggml_dup_tensor(ctx_.get(), x.tensor));
        auto logits = modules::LinearModule(
                          binding::linear_config(config.embed_dim, runtime_->assets().mula_config.audio_vocab_size, false))
                          .build(ctx, x, binding::linear_data(constants, weights.codebook0_head));
        logits_output_ = ggml_cpy(ctx_.get(), logits.tensor, ggml_dup_tensor(ctx_.get(), logits.tensor));
        readback_outputs_.push_back(hidden_output_);
        readback_outputs_.push_back(logits_output_);
        for (auto * output : readback_outputs_) {
            ggml_set_output(output);
        }
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        for (auto * output : readback_outputs_) {
            ggml_build_forward_expand(graph_, output);
        }
        constants.finish_graph();
        constants.ensure_uploaded();
        allocate_workspace();
        layer_step_elems_ = batch_size_ * config.num_kv_heads * head_dim;
    }

    ~HeartMuLaBackbonePrefillGraph() {
        release_workspace();
    }

    void release_workspace() {
        if (gallocr_ != nullptr) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
    }

    bool matches(const HeartMuLaWeightsRuntime & runtime, int64_t batch_size, int64_t steps) const {
        return runtime_ == &runtime && batch_size_ == batch_size && steps_ == steps;
    }

    HeartMuLaBackbonePrefillOutput run(const std::vector<float> & embeddings) {
        allocate_workspace();
        const auto & config = runtime_->assets().mula_config.backbone;
        const int64_t head_dim = require_head_dim(config);
        if (static_cast<int64_t>(embeddings.size()) != batch_size_ * steps_ * config.embed_dim) {
            throw std::runtime_error("HeartMuLa backbone prefill embedding payload size mismatch");
        }
        ggml_backend_tensor_set(input_, embeddings.data(), 0, embeddings.size() * sizeof(float));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("HeartMuLa backbone prefill graph compute failed");
        }
        HeartMuLaBackbonePrefillOutput out;
        out.result.logits.vocab_size = runtime_->assets().mula_config.audio_vocab_size;
        out.result.logits.values.resize(static_cast<size_t>(batch_size_ * out.result.logits.vocab_size));
        ggml_backend_tensor_get(
            logits_output_,
            out.result.logits.values.data(),
            0,
            out.result.logits.values.size() * sizeof(float));
        out.result.last_hidden.dims = config.embed_dim;
        out.result.last_hidden.values.resize(static_cast<size_t>(batch_size_ * config.embed_dim));
        ggml_backend_tensor_get(
            hidden_output_,
            out.result.last_hidden.values.data(),
            0,
            out.result.last_hidden.values.size() * sizeof(float));
        out.state.current_end = steps_;
        out.state.layers.resize(keys_.size());
        const size_t layer_values = static_cast<size_t>(steps_ * layer_step_elems_);
        for (size_t layer = 0; layer < keys_.size(); ++layer) {
            auto & state_layer = out.state.layers[layer];
            state_layer.valid_steps = steps_;
            std::vector<float> key_values(layer_values);
            std::vector<float> value_values(layer_values);
            ggml_backend_tensor_get(keys_[layer], key_values.data(), 0, key_values.size() * sizeof(float));
            ggml_backend_tensor_get(values_[layer], value_values.data(), 0, value_values.size() * sizeof(float));
            state_layer.key = flatten_prefill_cache_by_step(
                key_values,
                batch_size_,
                steps_,
                config.num_kv_heads,
                head_dim);
            state_layer.value = flatten_prefill_cache_by_step(
                value_values,
                batch_size_,
                steps_,
                config.num_kv_heads,
                head_dim);
        }
        return out;
    }

private:
    void allocate_workspace() {
        if (gallocr_ != nullptr) {
            return;
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
                gallocr_ = nullptr;
            }
            throw std::runtime_error("failed to allocate HeartMuLa backbone prefill graph");
        }
        std::vector<int32_t> positions_values(static_cast<size_t>(steps_), 0);
        for (int64_t i = 0; i < steps_; ++i) {
            positions_values[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        ggml_backend_tensor_set(positions_, positions_values.data(), 0, positions_values.size() * sizeof(int32_t));
    }

    const HeartMuLaWeightsRuntime * runtime_ = nullptr;
    int64_t batch_size_ = 0;
    int64_t steps_ = 0;
    int64_t layer_step_elems_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * hidden_output_ = nullptr;
    ggml_tensor * logits_output_ = nullptr;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    std::vector<ggml_tensor *> readback_outputs_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

class HeartMuLaBackboneCachedStepGraph {
public:
    HeartMuLaBackboneCachedStepGraph(
        const HeartMuLaWeightsRuntime & runtime,
        int64_t batch_size,
        int64_t cache_steps)
        : runtime_(&runtime),
          batch_size_(batch_size),
          cache_steps_(cache_steps) {
        if (batch_size_ <= 0 || cache_steps_ <= 0) {
            throw std::runtime_error("HeartMuLa backbone cached step graph shape is invalid");
        }
        const auto & config = runtime_->assets().mula_config.backbone;
        const auto & weights = runtime_->weights();
        const int64_t head_dim = require_head_dim(config);
        ggml_init_params params{runtime_->backbone_step_graph_arena_bytes(), nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize HeartMuLa backbone cached step graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "heartmula.backbone.cached_step"};
        auto x = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({batch_size_, 1, config.embed_dim}));
        input_ = x.tensor;
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, cache_steps_ + 1, 1, 1, 1);
        auto attention_mask = core::wrap_tensor(
            attention_mask_,
            core::TensorShape::from_dims({1, 1, 1, cache_steps_ + 1}),
            GGML_TYPE_F16);

        std::vector<core::TensorValue> cache_keys;
        std::vector<core::TensorValue> cache_values;
        cache_keys.reserve(weights.backbone.layers.size());
        cache_values.reserve(weights.backbone.layers.size());
        auto & constants = runtime_->backbone_constants();
        constants.begin_graph();
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        for (const auto & layer : weights.backbone.layers) {
            cache_keys.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_ + 1, batch_size_ * config.num_kv_heads, head_dim})));
            cache_values.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_ + 1, batch_size_ * config.num_kv_heads, head_dim})));
            auto layer_out = build_heartmula_transformer_layer_static_cache_tail(
                ctx,
                graph_,
                x,
                positions,
                layer,
                config,
                weights.backbone_rope_factors,
                constants,
                cache_keys.back(),
                cache_values.back(),
                attention_mask);
            x = layer_out.output;
            key_sources_.push_back(ggml_view_1d(ctx_.get(), layer_out.key.tensor, batch_size_ * config.num_kv_heads * head_dim, 0));
            value_sources_.push_back(ggml_view_1d(ctx_.get(), layer_out.value.tensor, batch_size_ * config.num_kv_heads * head_dim, 0));
        }
        step_cache_ = runtime::TransformerKVCache(
            cache_steps_ + 1,
            batch_size_ * config.num_kv_heads * head_dim,
            std::move(cache_keys),
            std::move(cache_values));
        build_transfer_views(batch_size_ * config.num_kv_heads * head_dim);
        x = modules::RMSNormModule({config.embed_dim, config.norm_eps, true, false})
                .build(ctx, x, binding::norm_data(constants, weights.backbone.norm));
        hidden_output_ = x.tensor;
        auto logits = modules::LinearModule(
                          binding::linear_config(config.embed_dim, runtime_->assets().mula_config.audio_vocab_size, false))
                          .build(ctx, x, binding::linear_data(constants, weights.codebook0_head));
        logits_output_ = logits.tensor;
        ggml_set_output(logits_output_);
        ggml_build_forward_expand(graph_, logits_output_);
        constants.finish_graph();
        constants.ensure_uploaded();
        allocate_workspace();
        attention_mask_buffer_.assign(
            static_cast<size_t>(cache_steps_ + 1),
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
    }

    ~HeartMuLaBackboneCachedStepGraph() {
        release_workspace();
    }

    void release_workspace() {
        if (buffer_ != nullptr) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
            ggml_backend_buffer_free(buffer_);
            buffer_ = nullptr;
        }
    }

    bool can_run(const HeartMuLaWeightsRuntime & runtime, int64_t batch_size, int64_t required_capacity) const {
        return runtime_ == &runtime && batch_size_ == batch_size && cache_steps_ >= required_capacity;
    }

    int64_t current_end() const noexcept {
        return step_cache_.current_end();
    }

    int64_t cache_steps() const noexcept {
        return cache_steps_;
    }

    void import_state(const runtime::TransformerKVState & state) {
        allocate_workspace();
        step_cache_.import_state(state);
    }

    runtime::TransformerKVState export_state() const {
        return step_cache_.export_state();
    }

    HeartMuLaBackboneResult run_step(const std::vector<float> & embedding) {
        allocate_workspace();
        const auto & config = runtime_->assets().mula_config.backbone;
        if (static_cast<int64_t>(embedding.size()) != batch_size_ * config.embed_dim) {
            throw std::runtime_error("HeartMuLa backbone cached step embedding payload size mismatch");
        }
        if (step_cache_.valid_steps() >= cache_steps_) {
            throw std::runtime_error("HeartMuLa backbone cached step exceeds cache capacity");
        }
        ggml_backend_tensor_set(input_, embedding.data(), 0, embedding.size() * sizeof(float));
        const int32_t position = static_cast<int32_t>(step_cache_.current_end());
        ggml_backend_tensor_set(positions_, &position, 0, sizeof(int32_t));
        const int64_t cache_slot = step_cache_.valid_steps();
        std::fill(
            attention_mask_buffer_.begin(),
            attention_mask_buffer_.end(),
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        for (int64_t i = 0; i < step_cache_.valid_steps(); ++i) {
            attention_mask_buffer_[static_cast<size_t>(i)] = ggml_fp32_to_fp16(0.0F);
        }
        attention_mask_buffer_[static_cast<size_t>(cache_steps_)] = ggml_fp32_to_fp16(0.0F);
        ggml_backend_tensor_set(
            attention_mask_,
            attention_mask_buffer_.data(),
            0,
            attention_mask_buffer_.size() * sizeof(ggml_fp16_t));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("HeartMuLa backbone cached step graph compute failed");
        }
        const size_t dst_slot = static_cast<size_t>(cache_slot);
        for (size_t layer = 0; layer < key_sources_.size(); ++layer) {
            ggml_backend_tensor_copy(key_sources_[layer], key_destinations_[dst_slot][layer]);
            ggml_backend_tensor_copy(value_sources_[layer], value_destinations_[dst_slot][layer]);
        }
        HeartMuLaBackboneResult out;
        out.logits.vocab_size = runtime_->assets().mula_config.audio_vocab_size;
        out.logits.values.resize(static_cast<size_t>(batch_size_ * out.logits.vocab_size));
        ggml_backend_tensor_get(logits_output_, out.logits.values.data(), 0, out.logits.values.size() * sizeof(float));
        out.last_hidden.dims = config.embed_dim;
        out.last_hidden.values.resize(static_cast<size_t>(batch_size_ * config.embed_dim));
        ggml_backend_tensor_get(hidden_output_, out.last_hidden.values.data(), 0, out.last_hidden.values.size() * sizeof(float));
        step_cache_.advance_after_direct_append(1);
        return out;
    }

private:
    void build_transfer_views(int64_t step_elems) {
        key_destinations_.assign(static_cast<size_t>(cache_steps_), {});
        value_destinations_.assign(static_cast<size_t>(cache_steps_), {});
        for (int64_t slot = 0; slot < cache_steps_; ++slot) {
            const size_t byte_offset = static_cast<size_t>(slot * step_elems) * sizeof(float);
            auto & key_slot = key_destinations_[static_cast<size_t>(slot)];
            auto & value_slot = value_destinations_[static_cast<size_t>(slot)];
            key_slot.reserve(key_sources_.size());
            value_slot.reserve(value_sources_.size());
            for (size_t layer = 0; layer < key_sources_.size(); ++layer) {
                key_slot.push_back(ggml_view_1d(ctx_.get(), step_cache_.key_tensor(layer).tensor, step_elems, byte_offset));
                value_slot.push_back(ggml_view_1d(ctx_.get(), step_cache_.value_tensor(layer).tensor, step_elems, byte_offset));
            }
        }
    }

    void allocate_workspace() {
        if (buffer_ != nullptr) {
            return;
        }
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
            throw std::runtime_error("failed to allocate HeartMuLa backbone cached step graph");
        }
    }

    const HeartMuLaWeightsRuntime * runtime_ = nullptr;
    int64_t batch_size_ = 0;
    int64_t cache_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * hidden_output_ = nullptr;
    ggml_tensor * logits_output_ = nullptr;
    std::vector<ggml_tensor *> key_sources_;
    std::vector<ggml_tensor *> value_sources_;
    std::vector<std::vector<ggml_tensor *>> key_destinations_;
    std::vector<std::vector<ggml_tensor *>> value_destinations_;
    std::vector<ggml_fp16_t> attention_mask_buffer_;
    runtime::TransformerKVCache step_cache_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

class HeartMuLaDecoderPrefillGraph {
public:
    HeartMuLaDecoderPrefillGraph(
        const HeartMuLaWeightsRuntime & runtime,
        int64_t batch_size,
        int64_t steps)
        : runtime_(&runtime),
          batch_size_(batch_size),
          steps_(steps) {
        if (batch_size_ <= 0 || steps_ <= 0) {
            throw std::runtime_error("HeartMuLa decoder prefill graph shape is invalid");
        }
        const auto & config = runtime_->assets().mula_config.decoder;
        const auto & weights = runtime_->weights();
        const int64_t head_dim = require_head_dim(config);
        ggml_init_params params{runtime_->decoder_prefill_graph_arena_bytes(), nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize HeartMuLa decoder prefill graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "heartmula.decoder.prefill"};
        auto x = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({batch_size_, steps_, runtime_->assets().mula_config.backbone.embed_dim}));
        input_ = x.tensor;
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, steps_);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({steps_}), GGML_TYPE_I32);
        codebook_index_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto codebook_index = core::wrap_tensor(codebook_index_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);

        auto & constants = runtime_->decoder_constants();
        constants.begin_graph();
        x = modules::LinearModule(
                binding::linear_config(runtime_->assets().mula_config.backbone.embed_dim, config.embed_dim, false))
                .build(ctx, x, binding::linear_data(constants, weights.projection));
        for (const auto & layer : weights.decoder.layers) {
            auto layer_out = build_heartmula_transformer_layer(
                ctx,
                x,
                positions,
                layer,
                config,
                weights.decoder_rope_factors,
                constants);
            x = layer_out.output;
            auto * key_readback = ggml_cpy(
                ctx_.get(),
                layer_out.key.tensor,
                ggml_dup_tensor(ctx_.get(), layer_out.key.tensor));
            auto * value_readback = ggml_cpy(
                ctx_.get(),
                layer_out.value.tensor,
                ggml_dup_tensor(ctx_.get(), layer_out.value.tensor));
            keys_.push_back(key_readback);
            values_.push_back(value_readback);
            readback_outputs_.push_back(key_readback);
            readback_outputs_.push_back(value_readback);
        }
        x = modules::SliceModule({1, steps_ - 1, 1}).build(ctx, x);
        x = modules::RMSNormModule({config.embed_dim, config.norm_eps, true, false})
                .build(ctx, x, binding::norm_data(constants, weights.decoder.norm));
        hidden_output_ = ggml_cpy(ctx_.get(), x.tensor, ggml_dup_tensor(ctx_.get(), x.tensor));
        auto logits = codebook_audio_logits(ctx, x, weights.audio_head, codebook_index, runtime_->assets().mula_config);
        logits_output_ = ggml_cpy(ctx_.get(), logits.tensor, ggml_dup_tensor(ctx_.get(), logits.tensor));
        readback_outputs_.push_back(hidden_output_);
        readback_outputs_.push_back(logits_output_);
        for (auto * output : readback_outputs_) {
            ggml_set_output(output);
        }
        graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
        for (auto * output : readback_outputs_) {
            ggml_build_forward_expand(graph_, output);
        }
        constants.finish_graph();
        constants.ensure_uploaded();
        allocate_workspace();
        layer_step_elems_ = batch_size_ * config.num_kv_heads * head_dim;
    }

    ~HeartMuLaDecoderPrefillGraph() {
        release_workspace();
    }

    void release_workspace() {
        if (gallocr_ != nullptr) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
    }

    bool matches(const HeartMuLaWeightsRuntime & runtime, int64_t batch_size, int64_t steps) const {
        return runtime_ == &runtime && batch_size_ == batch_size && steps_ == steps;
    }

    HeartMuLaDecoderPrefillOutput run(const std::vector<float> & embeddings, int64_t codebook_index) {
        allocate_workspace();
        const auto & full_config = runtime_->assets().mula_config;
        const auto & config = full_config.decoder;
        const int64_t head_dim = require_head_dim(config);
        if (static_cast<int64_t>(embeddings.size()) != batch_size_ * steps_ * full_config.backbone.embed_dim) {
            throw std::runtime_error("HeartMuLa decoder prefill embedding payload size mismatch");
        }
        if (codebook_index < 0 || codebook_index >= full_config.audio_num_codebooks - 1) {
            throw std::runtime_error("HeartMuLa decoder prefill codebook index is out of range");
        }
        ggml_backend_tensor_set(input_, embeddings.data(), 0, embeddings.size() * sizeof(float));
        const int32_t codebook = static_cast<int32_t>(codebook_index);
        ggml_backend_tensor_set(codebook_index_, &codebook, 0, sizeof(int32_t));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("HeartMuLa decoder prefill graph compute failed");
        }
        HeartMuLaDecoderPrefillOutput out;
        out.result.logits.vocab_size = full_config.audio_vocab_size;
        out.result.logits.values.resize(static_cast<size_t>(batch_size_ * full_config.audio_vocab_size));
        ggml_backend_tensor_get(
            logits_output_,
            out.result.logits.values.data(),
            0,
            out.result.logits.values.size() * sizeof(float));
        out.result.last_hidden.dims = config.embed_dim;
        out.result.last_hidden.values.resize(static_cast<size_t>(batch_size_ * config.embed_dim));
        ggml_backend_tensor_get(
            hidden_output_,
            out.result.last_hidden.values.data(),
            0,
            out.result.last_hidden.values.size() * sizeof(float));
        out.state.current_end = steps_;
        out.state.layers.resize(keys_.size());
        const size_t layer_values = static_cast<size_t>(steps_ * layer_step_elems_);
        for (size_t layer = 0; layer < keys_.size(); ++layer) {
            auto & state_layer = out.state.layers[layer];
            state_layer.valid_steps = steps_;
            std::vector<float> key_values(layer_values);
            std::vector<float> value_values(layer_values);
            ggml_backend_tensor_get(keys_[layer], key_values.data(), 0, key_values.size() * sizeof(float));
            ggml_backend_tensor_get(values_[layer], value_values.data(), 0, value_values.size() * sizeof(float));
            state_layer.key = flatten_prefill_cache_by_step(
                key_values,
                batch_size_,
                steps_,
                config.num_kv_heads,
                head_dim);
            state_layer.value = flatten_prefill_cache_by_step(
                value_values,
                batch_size_,
                steps_,
                config.num_kv_heads,
                head_dim);
        }
        return out;
    }

private:
    void allocate_workspace() {
        if (gallocr_ != nullptr) {
            return;
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
                gallocr_ = nullptr;
            }
            throw std::runtime_error("failed to allocate HeartMuLa decoder prefill graph");
        }
        std::vector<int32_t> positions_values(static_cast<size_t>(steps_), 0);
        for (int64_t i = 0; i < steps_; ++i) {
            positions_values[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        ggml_backend_tensor_set(positions_, positions_values.data(), 0, positions_values.size() * sizeof(int32_t));
    }

    const HeartMuLaWeightsRuntime * runtime_ = nullptr;
    int64_t batch_size_ = 0;
    int64_t steps_ = 0;
    int64_t layer_step_elems_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * codebook_index_ = nullptr;
    ggml_tensor * hidden_output_ = nullptr;
    ggml_tensor * logits_output_ = nullptr;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    std::vector<ggml_tensor *> readback_outputs_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

class HeartMuLaDecoderCachedStepGraph {
public:
    HeartMuLaDecoderCachedStepGraph(
        const HeartMuLaWeightsRuntime & runtime,
        int64_t batch_size,
        int64_t cache_steps)
        : runtime_(&runtime),
          batch_size_(batch_size),
          cache_steps_(cache_steps) {
        if (batch_size_ <= 0 || cache_steps_ <= 0) {
            throw std::runtime_error("HeartMuLa decoder cached step graph shape is invalid");
        }
        const auto & config = runtime_->assets().mula_config.decoder;
        const auto & weights = runtime_->weights();
        const int64_t head_dim = require_head_dim(config);
        ggml_init_params params{runtime_->decoder_step_graph_arena_bytes(), nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize HeartMuLa decoder cached step graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "heartmula.decoder.cached_step"};
        auto x = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({batch_size_, 1, runtime_->assets().mula_config.backbone.embed_dim}));
        input_ = x.tensor;
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        cache_slot_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto cache_slot = core::wrap_tensor(cache_slot_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        codebook_index_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto codebook_index = core::wrap_tensor(codebook_index_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, cache_steps_, 1, 1, 1);
        auto attention_mask = core::wrap_tensor(
            attention_mask_,
            core::TensorShape::from_dims({1, 1, 1, cache_steps_}),
            GGML_TYPE_F16);

        std::vector<core::TensorValue> cache_keys;
        std::vector<core::TensorValue> cache_values;
        cache_keys.reserve(weights.decoder.layers.size());
        cache_values.reserve(weights.decoder.layers.size());
        auto & constants = runtime_->decoder_constants();
        constants.begin_graph();
        x = modules::LinearModule(
                binding::linear_config(runtime_->assets().mula_config.backbone.embed_dim, config.embed_dim, false))
                .build(ctx, x, binding::linear_data(constants, weights.projection));
        for (const auto & layer : weights.decoder.layers) {
            cache_keys.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_, batch_size_ * config.num_kv_heads, head_dim})));
            cache_values.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_, batch_size_ * config.num_kv_heads, head_dim})));
            x = build_heartmula_transformer_layer_set_rows_tail(
                ctx,
                x,
                positions,
                layer,
                config,
                weights.decoder_rope_factors,
                constants,
                cache_keys.back(),
                cache_values.back(),
                cache_slot,
                attention_mask);
        }
        step_cache_ = runtime::TransformerKVCache(
            cache_steps_,
            batch_size_ * config.num_kv_heads * head_dim,
            std::move(cache_keys),
            std::move(cache_values));
        x = modules::RMSNormModule({config.embed_dim, config.norm_eps, true, false})
                .build(ctx, x, binding::norm_data(constants, weights.decoder.norm));
        hidden_output_ = x.tensor;
        auto logits = codebook_audio_logits(ctx, x, weights.audio_head, codebook_index, runtime_->assets().mula_config);
        logits_output_ = logits.tensor;
        ggml_set_output(logits_output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
        ggml_build_forward_expand(graph_, logits_output_);
        constants.finish_graph();
        constants.ensure_uploaded();
        allocate_workspace();
        attention_mask_buffer_.assign(
            static_cast<size_t>(cache_steps_),
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
    }

    ~HeartMuLaDecoderCachedStepGraph() {
        release_workspace();
    }

    void release_workspace() {
        if (buffer_ != nullptr) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
            ggml_backend_buffer_free(buffer_);
            buffer_ = nullptr;
        }
    }

    bool can_run(const HeartMuLaWeightsRuntime & runtime, int64_t batch_size, int64_t required_capacity) const {
        return runtime_ == &runtime && batch_size_ == batch_size && cache_steps_ >= required_capacity;
    }

    int64_t current_end() const noexcept {
        return step_cache_.current_end();
    }

    void import_state(const runtime::TransformerKVState & state) {
        allocate_workspace();
        step_cache_.import_state(state);
    }

    runtime::TransformerKVState export_state() const {
        return step_cache_.export_state();
    }

    HeartMuLaDecoderResult run_step(const std::vector<float> & embedding, int64_t codebook_index) {
        allocate_workspace();
        const auto & full_config = runtime_->assets().mula_config;
        const auto & config = full_config.decoder;
        if (static_cast<int64_t>(embedding.size()) != batch_size_ * full_config.backbone.embed_dim) {
            throw std::runtime_error("HeartMuLa decoder cached step embedding payload size mismatch");
        }
        if (codebook_index < 0 || codebook_index >= full_config.audio_num_codebooks - 1) {
            throw std::runtime_error("HeartMuLa decoder cached step codebook index is out of range");
        }
        if (step_cache_.valid_steps() >= cache_steps_) {
            throw std::runtime_error("HeartMuLa decoder cached step exceeds cache capacity");
        }
        ggml_backend_tensor_set(input_, embedding.data(), 0, embedding.size() * sizeof(float));
        const int32_t position = static_cast<int32_t>(step_cache_.current_end());
        ggml_backend_tensor_set(positions_, &position, 0, sizeof(int32_t));
        const int32_t cache_slot = static_cast<int32_t>(step_cache_.valid_steps());
        ggml_backend_tensor_set(cache_slot_, &cache_slot, 0, sizeof(int32_t));
        const int32_t codebook = static_cast<int32_t>(codebook_index);
        ggml_backend_tensor_set(codebook_index_, &codebook, 0, sizeof(int32_t));
        std::fill(
            attention_mask_buffer_.begin(),
            attention_mask_buffer_.end(),
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        for (int64_t i = 0; i < step_cache_.valid_steps(); ++i) {
            attention_mask_buffer_[static_cast<size_t>(i)] = ggml_fp32_to_fp16(0.0F);
        }
        attention_mask_buffer_[static_cast<size_t>(cache_slot)] = ggml_fp32_to_fp16(0.0F);
        ggml_backend_tensor_set(
            attention_mask_,
            attention_mask_buffer_.data(),
            0,
            attention_mask_buffer_.size() * sizeof(ggml_fp16_t));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("HeartMuLa decoder cached step graph compute failed");
        }
        HeartMuLaDecoderResult out;
        out.logits.vocab_size = full_config.audio_vocab_size;
        out.logits.values.resize(static_cast<size_t>(batch_size_ * full_config.audio_vocab_size));
        ggml_backend_tensor_get(logits_output_, out.logits.values.data(), 0, out.logits.values.size() * sizeof(float));
        out.last_hidden.dims = config.embed_dim;
        out.last_hidden.values.resize(static_cast<size_t>(batch_size_ * config.embed_dim));
        ggml_backend_tensor_get(hidden_output_, out.last_hidden.values.data(), 0, out.last_hidden.values.size() * sizeof(float));
        step_cache_.advance_after_direct_append(1);
        return out;
    }

private:
    void allocate_workspace() {
        if (buffer_ != nullptr) {
            return;
        }
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
            throw std::runtime_error("failed to allocate HeartMuLa decoder cached step graph");
        }
    }

    const HeartMuLaWeightsRuntime * runtime_ = nullptr;
    int64_t batch_size_ = 0;
    int64_t cache_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * cache_slot_ = nullptr;
    ggml_tensor * codebook_index_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * hidden_output_ = nullptr;
    ggml_tensor * logits_output_ = nullptr;
    std::vector<ggml_fp16_t> attention_mask_buffer_;
    runtime::TransformerKVCache step_cache_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

class HeartMuLaFrameEmbeddingGraph {
public:
    HeartMuLaFrameEmbeddingGraph(
        const HeartMuLaWeightsRuntime & runtime,
        int64_t batch_size,
        int64_t steps,
        bool apply_muq)
        : runtime_(&runtime),
          batch_size_(batch_size),
          steps_(steps),
          apply_muq_(apply_muq) {
        if (batch_size_ <= 0 || steps_ <= 0) {
            throw std::runtime_error("HeartMuLa frame embedding graph shape is invalid");
        }
        const auto & config = runtime_->assets().mula_config;
        ggml_init_params params{runtime_->frame_embedding_graph_arena_bytes(), nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize HeartMuLa frame embedding graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "heartmula.frame_embedding"};
        audio_token_ids_ = ggml_new_tensor_3d(
            ctx_.get(),
            GGML_TYPE_I32,
            config.audio_num_codebooks,
            steps_,
            batch_size_);
        text_token_ids_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_I32, steps_, batch_size_);
        audio_mask_ = ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, config.audio_num_codebooks, steps_, batch_size_);
        text_cond_mask_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_F32, steps_, batch_size_);
        text_uncond_mask_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_F32, steps_, batch_size_);
        auto audio_ids = core::wrap_tensor(
            audio_token_ids_,
            core::TensorShape::from_dims({batch_size_, steps_, config.audio_num_codebooks}),
            GGML_TYPE_I32);
        auto text_ids = core::wrap_tensor(
            text_token_ids_,
            core::TensorShape::from_dims({batch_size_, steps_}),
            GGML_TYPE_I32);
        auto audio_mask = core::wrap_tensor(
            audio_mask_,
            core::TensorShape::from_dims({batch_size_, steps_, config.audio_num_codebooks}),
            GGML_TYPE_F32);
        auto text_cond_mask = core::wrap_tensor(
            text_cond_mask_,
            core::TensorShape::from_dims({batch_size_, steps_}),
            GGML_TYPE_F32);
        auto text_uncond_mask = core::wrap_tensor(
            text_uncond_mask_,
            core::TensorShape::from_dims({batch_size_, steps_}),
            GGML_TYPE_F32);

        auto & constants = runtime_->embedding_constants();
        constants.begin_graph();
        auto audio_emb = modules::EmbeddingModule({
                config.audio_vocab_size * config.audio_num_codebooks,
                config.backbone.embed_dim,
            }).build(ctx, audio_ids, runtime_->weights().audio_embeddings);
        auto audio_mask_4d = core::reshape_tensor(
            ctx,
            audio_mask,
            core::TensorShape::from_dims({batch_size_, steps_, config.audio_num_codebooks, 1}));
        audio_mask_4d = modules::RepeatModule({
                core::TensorShape::from_dims({batch_size_, steps_, config.audio_num_codebooks, config.backbone.embed_dim}),
            }).build(ctx, audio_mask_4d);
        audio_emb = modules::MulModule{}.build(ctx, audio_emb, audio_mask_4d);
        auto audio_sum = modules::ReduceSumModule({2}).build(ctx, audio_emb);
        audio_sum = core::reshape_tensor(
            ctx,
            audio_sum,
            core::TensorShape::from_dims({batch_size_, steps_, config.backbone.embed_dim}));

        auto text_emb = modules::EmbeddingModule({config.text_vocab_size, config.backbone.embed_dim})
                            .build(ctx, text_ids, runtime_->weights().text_embeddings);
        zero_text_ids_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_I32, steps_, batch_size_);
        auto zero_text_ids = core::wrap_tensor(
            zero_text_ids_,
            core::TensorShape::from_dims({batch_size_, steps_}),
            GGML_TYPE_I32);
        auto uncond_text = modules::EmbeddingModule({1, config.backbone.embed_dim})
                               .build(ctx, zero_text_ids, runtime_->weights().unconditional_text_embedding);
        auto text_cond_mask_3d = core::reshape_tensor(
            ctx,
            text_cond_mask,
            core::TensorShape::from_dims({batch_size_, steps_, 1}));
        text_cond_mask_3d = modules::RepeatModule({
                core::TensorShape::from_dims({batch_size_, steps_, config.backbone.embed_dim}),
            }).build(ctx, text_cond_mask_3d);
        auto text_uncond_mask_3d = core::reshape_tensor(
            ctx,
            text_uncond_mask,
            core::TensorShape::from_dims({batch_size_, steps_, 1}));
        text_uncond_mask_3d = modules::RepeatModule({
                core::TensorShape::from_dims({batch_size_, steps_, config.backbone.embed_dim}),
            }).build(ctx, text_uncond_mask_3d);
        auto text = modules::AddModule{}.build(
            ctx,
            modules::MulModule{}.build(ctx, text_emb, text_cond_mask_3d),
            modules::MulModule{}.build(ctx, uncond_text, text_uncond_mask_3d));
        auto merged = modules::AddModule{}.build(ctx, audio_sum, text);

        if (apply_muq_) {
            muq_embed_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_F32, config.muq_dim, batch_size_);
            muq_cond_mask_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_F32, batch_size_);
            muq_uncond_mask_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_F32, batch_size_);
            muq_row_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
            auto muq_embed = core::wrap_tensor(
                muq_embed_,
                core::TensorShape::from_dims({batch_size_, config.muq_dim}),
                GGML_TYPE_F32);
            auto muq = modules::LinearModule(binding::linear_config(config.muq_dim, config.backbone.embed_dim, true))
                           .build(
                               ctx,
                               muq_embed,
                               binding::linear_data(
                                   constants,
                                   runtime_->weights().muq_linear_weight,
                                   runtime_->weights().muq_linear_bias));
            zero_muq_ids_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, batch_size_);
            auto zero_muq_ids = core::wrap_tensor(
                zero_muq_ids_,
                core::TensorShape::from_dims({batch_size_}),
                GGML_TYPE_I32);
            auto muq_uncond = modules::EmbeddingModule({1, config.backbone.embed_dim})
                                  .build(ctx, zero_muq_ids, runtime_->weights().unconditional_text_embedding);
            auto muq_cond_mask = core::wrap_tensor(
                muq_cond_mask_,
                core::TensorShape::from_dims({batch_size_}),
                GGML_TYPE_F32);
            auto muq_uncond_mask = core::wrap_tensor(
                muq_uncond_mask_,
                core::TensorShape::from_dims({batch_size_}),
                GGML_TYPE_F32);
            muq_cond_mask = core::reshape_tensor(ctx, muq_cond_mask, core::TensorShape::from_dims({batch_size_, 1}));
            muq_uncond_mask = core::reshape_tensor(ctx, muq_uncond_mask, core::TensorShape::from_dims({batch_size_, 1}));
            muq_cond_mask = modules::RepeatModule({
                    core::TensorShape::from_dims({batch_size_, config.backbone.embed_dim}),
                }).build(ctx, muq_cond_mask);
            muq_uncond_mask = modules::RepeatModule({
                    core::TensorShape::from_dims({batch_size_, config.backbone.embed_dim}),
                }).build(ctx, muq_uncond_mask);
            muq = modules::AddModule{}.build(
                ctx,
                modules::MulModule{}.build(ctx, muq, muq_cond_mask),
                modules::MulModule{}.build(ctx, muq_uncond, muq_uncond_mask));
            auto step_major = modules::TransposeModule({{1, 0, 2, 3}, 3}).build(ctx, merged);
            step_major = core::ensure_backend_addressable_layout(ctx, step_major);
            auto flat_steps = core::reshape_tensor(
                ctx,
                step_major,
                core::TensorShape::from_dims({steps_, batch_size_ * config.backbone.embed_dim}));
            auto flat_muq = core::reshape_tensor(
                ctx,
                core::ensure_backend_addressable_layout(ctx, muq),
                core::TensorShape::from_dims({1, batch_size_ * config.backbone.embed_dim}));
            auto row_index = core::wrap_tensor(muq_row_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
            auto flat_updated = core::wrap_tensor(
                ggml_set_rows(ctx.ggml, flat_steps.tensor, flat_muq.tensor, row_index.tensor),
                flat_steps.shape,
                GGML_TYPE_F32);
            auto updated = core::reshape_tensor(
                ctx,
                flat_updated,
                core::TensorShape::from_dims({steps_, batch_size_, config.backbone.embed_dim}));
            merged = modules::TransposeModule({{1, 0, 2, 3}, 3}).build(ctx, updated);
        }

        output_ = core::ensure_backend_addressable_layout(ctx, merged).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
        ggml_build_forward_expand(graph_, output_);
        constants.finish_graph();
        constants.ensure_uploaded();
        zero_i32_buffer_.assign(static_cast<size_t>(std::max(batch_size_, batch_size_ * steps_)), 0);
        allocate_workspace();
    }

    ~HeartMuLaFrameEmbeddingGraph() {
        release_workspace();
    }

    void release_workspace() {
        if (buffer_ != nullptr) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
            ggml_backend_buffer_free(buffer_);
            buffer_ = nullptr;
        }
    }

    bool matches(const HeartMuLaWeightsRuntime & runtime, int64_t batch_size, int64_t steps, bool apply_muq) const {
        return runtime_ == &runtime && batch_size_ == batch_size && steps_ == steps && apply_muq_ == apply_muq;
    }

    HeartMuLaMergedEmbeddings run(const HeartMuLaFrameEmbeddingInputs & inputs) {
        allocate_workspace();
        const auto & config = runtime_->assets().mula_config;
        const size_t audio_count = static_cast<size_t>(batch_size_ * steps_ * config.audio_num_codebooks);
        const size_t text_count = static_cast<size_t>(batch_size_ * steps_);
        if (inputs.batch_size != batch_size_ || inputs.steps != steps_) {
            throw std::runtime_error("HeartMuLa frame embedding input shape mismatch");
        }
        if (inputs.audio_token_ids.size() != audio_count || inputs.audio_mask.size() != audio_count ||
            inputs.text_token_ids.size() != text_count || inputs.text_cond_mask.size() != text_count ||
            inputs.text_uncond_mask.size() != text_count) {
            throw std::runtime_error("HeartMuLa frame embedding input payload size mismatch");
        }
        ggml_backend_tensor_set(audio_token_ids_, inputs.audio_token_ids.data(), 0, audio_count * sizeof(int32_t));
        ggml_backend_tensor_set(text_token_ids_, inputs.text_token_ids.data(), 0, text_count * sizeof(int32_t));
        ggml_backend_tensor_set(audio_mask_, inputs.audio_mask.data(), 0, audio_count * sizeof(float));
        ggml_backend_tensor_set(text_cond_mask_, inputs.text_cond_mask.data(), 0, text_count * sizeof(float));
        ggml_backend_tensor_set(text_uncond_mask_, inputs.text_uncond_mask.data(), 0, text_count * sizeof(float));
        if (apply_muq_) {
            if (!inputs.apply_muq || inputs.muq_row < 0 || inputs.muq_row >= steps_) {
                throw std::runtime_error("HeartMuLa frame embedding MuQ row is invalid");
            }
            if (inputs.muq_embed.size() != static_cast<size_t>(batch_size_ * config.muq_dim) ||
                inputs.muq_cond_mask.size() != static_cast<size_t>(batch_size_) ||
                inputs.muq_uncond_mask.size() != static_cast<size_t>(batch_size_)) {
                throw std::runtime_error("HeartMuLa frame embedding MuQ payload size mismatch");
            }
            const int32_t row = static_cast<int32_t>(inputs.muq_row);
            ggml_backend_tensor_set(muq_embed_, inputs.muq_embed.data(), 0, inputs.muq_embed.size() * sizeof(float));
            ggml_backend_tensor_set(muq_cond_mask_, inputs.muq_cond_mask.data(), 0, inputs.muq_cond_mask.size() * sizeof(float));
            ggml_backend_tensor_set(muq_uncond_mask_, inputs.muq_uncond_mask.data(), 0, inputs.muq_uncond_mask.size() * sizeof(float));
            ggml_backend_tensor_set(muq_row_, &row, 0, sizeof(int32_t));
        } else if (inputs.apply_muq) {
            throw std::runtime_error("HeartMuLa frame embedding graph was built without MuQ support");
        }
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("HeartMuLa frame embedding graph compute failed");
        }
        HeartMuLaMergedEmbeddings out;
        out.batch_size = batch_size_;
        out.steps = steps_;
        out.dims = config.backbone.embed_dim;
        out.values.resize(static_cast<size_t>(batch_size_ * steps_ * config.backbone.embed_dim));
        ggml_backend_tensor_get(output_, out.values.data(), 0, out.values.size() * sizeof(float));
        return out;
    }

private:
    void allocate_workspace() {
        if (buffer_ != nullptr) {
            return;
        }
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
            throw std::runtime_error("failed to allocate HeartMuLa frame embedding graph");
        }
        ggml_backend_tensor_set(
            zero_text_ids_,
            zero_i32_buffer_.data(),
            0,
            static_cast<size_t>(batch_size_ * steps_) * sizeof(int32_t));
        if (apply_muq_) {
            ggml_backend_tensor_set(
                zero_muq_ids_,
                zero_i32_buffer_.data(),
                0,
                static_cast<size_t>(batch_size_) * sizeof(int32_t));
        }
    }

    const HeartMuLaWeightsRuntime * runtime_ = nullptr;
    int64_t batch_size_ = 0;
    int64_t steps_ = 0;
    bool apply_muq_ = false;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * audio_token_ids_ = nullptr;
    ggml_tensor * text_token_ids_ = nullptr;
    ggml_tensor * zero_text_ids_ = nullptr;
    ggml_tensor * zero_muq_ids_ = nullptr;
    ggml_tensor * audio_mask_ = nullptr;
    ggml_tensor * text_cond_mask_ = nullptr;
    ggml_tensor * text_uncond_mask_ = nullptr;
    ggml_tensor * muq_embed_ = nullptr;
    ggml_tensor * muq_cond_mask_ = nullptr;
    ggml_tensor * muq_uncond_mask_ = nullptr;
    ggml_tensor * muq_row_ = nullptr;
    ggml_tensor * output_ = nullptr;
    std::vector<int32_t> zero_i32_buffer_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

HeartMuLaWeights load_heartmula_weights(
    const HeartMuLaAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type) {
    require_assets(assets);
    const auto & source = *assets.mula_weights;
    const auto & config = assets.mula_config;
    const auto & backbone = config.backbone;
    const auto & decoder = config.decoder;
    HeartMuLaWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "heartmula.mula.weights",
        weight_context_bytes);
    weights.text_embeddings = weights.store->load_tensor(
        source,
        "text_embeddings.weight",
        weight_storage_type,
        {config.text_vocab_size, backbone.embed_dim});
    weights.audio_embeddings = weights.store->load_tensor(
        source,
        "audio_embeddings.weight",
        weight_storage_type,
        {config.audio_vocab_size * config.audio_num_codebooks, backbone.embed_dim});
    weights.unconditional_text_embedding = weights.store->load_tensor(
        source,
        "unconditional_text_embedding.weight",
        weight_storage_type,
        {1, backbone.embed_dim});
    weights.projection = weights.store->load_tensor(
        source,
        "projection.weight",
        weight_storage_type,
        {decoder.embed_dim, backbone.embed_dim});
    weights.codebook0_head = weights.store->load_tensor(
        source,
        "codebook0_head.weight",
        weight_storage_type,
        {config.audio_vocab_size, backbone.embed_dim});
    weights.audio_head = weights.store->load_tensor(
        source,
        "audio_head",
        weight_storage_type,
        {config.audio_num_codebooks - 1, decoder.embed_dim, config.audio_vocab_size});
    weights.muq_linear_weight = weights.store->load_tensor(
        source,
        "muq_linear.weight",
        weight_storage_type,
        {backbone.embed_dim, config.muq_dim});
    weights.muq_linear_bias = weights.store->load_tensor(
        source,
        "muq_linear.bias",
        assets::TensorStorageType::F32,
        {backbone.embed_dim});
    weights.backbone_rope_factors = weights.store->make_f32(
        core::TensorShape::from_dims({backbone.head_dim / 2}),
        llama3_scaled_rope_factors(backbone));
    weights.decoder_rope_factors = weights.store->make_f32(
        core::TensorShape::from_dims({decoder.head_dim / 2}),
        llama3_scaled_rope_factors(decoder));
    weights.backbone = load_transformer_weights(
        *weights.store,
        source,
        backbone,
        "backbone",
        weight_storage_type);
    weights.decoder = load_transformer_weights(
        *weights.store,
        source,
        decoder,
        "decoder",
        weight_storage_type);
    weights.store->upload();
    return weights;
}

HeartMuLaBackboneCachedState::HeartMuLaBackboneCachedState() = default;
HeartMuLaBackboneCachedState::~HeartMuLaBackboneCachedState() = default;
HeartMuLaBackboneCachedState::HeartMuLaBackboneCachedState(HeartMuLaBackboneCachedState &&) noexcept = default;
HeartMuLaBackboneCachedState & HeartMuLaBackboneCachedState::operator=(
    HeartMuLaBackboneCachedState &&) noexcept = default;

HeartMuLaDecoderCachedState::HeartMuLaDecoderCachedState() = default;
HeartMuLaDecoderCachedState::~HeartMuLaDecoderCachedState() = default;
HeartMuLaDecoderCachedState::HeartMuLaDecoderCachedState(HeartMuLaDecoderCachedState &&) noexcept = default;
HeartMuLaDecoderCachedState & HeartMuLaDecoderCachedState::operator=(
    HeartMuLaDecoderCachedState &&) noexcept = default;

HeartMuLaWeightsRuntime::HeartMuLaWeightsRuntime(
    std::shared_ptr<const HeartMuLaAssets> assets,
    core::BackendType backend_type,
    int device,
    int threads,
    size_t weight_context_bytes,
    size_t constant_context_bytes,
    size_t backbone_prefill_graph_arena_bytes,
    size_t backbone_step_graph_arena_bytes,
    size_t decoder_prefill_graph_arena_bytes,
    size_t decoder_step_graph_arena_bytes,
    size_t frame_embedding_graph_arena_bytes,
    assets::TensorStorageType weight_storage_type)
    : assets_(std::move(assets)),
      backend_type_(backend_type),
      device_(device),
      threads_(threads),
      backbone_prefill_graph_arena_bytes_(backbone_prefill_graph_arena_bytes),
      backbone_step_graph_arena_bytes_(backbone_step_graph_arena_bytes),
      decoder_prefill_graph_arena_bytes_(decoder_prefill_graph_arena_bytes),
      decoder_step_graph_arena_bytes_(decoder_step_graph_arena_bytes),
      frame_embedding_graph_arena_bytes_(frame_embedding_graph_arena_bytes) {
    if (assets_ == nullptr) {
        throw std::runtime_error("HeartMuLa weights runtime requires assets");
    }
    if (threads_ <= 0) {
        throw std::runtime_error("HeartMuLa weights runtime requires positive thread count");
    }
    backend_ = core::init_backend({backend_type, device, threads_});
    weights_ = std::make_shared<HeartMuLaWeights>(
        load_heartmula_weights(*assets_, backend_, backend_type, weight_context_bytes, weight_storage_type));
    assets_->mula_weights->release_storage();
    backbone_constants_ = std::make_unique<core::ConstantTensorCache>(
        backend_,
        threads_,
        "heartmula.mula.backbone.constants",
        constant_context_bytes);
    decoder_constants_ = std::make_unique<core::ConstantTensorCache>(
        backend_,
        threads_,
        "heartmula.mula.decoder.constants",
        constant_context_bytes);
    embedding_constants_ = std::make_unique<core::ConstantTensorCache>(
        backend_,
        threads_,
        "heartmula.mula.embedding.constants",
        constant_context_bytes);
}

HeartMuLaWeightsRuntime::~HeartMuLaWeightsRuntime() {
    step_frame_embedding_graph_.reset();
    prompt_frame_embedding_graph_.reset();
    decoder_cached_step_graph_.reset();
    backbone_prefill_graph_.reset();
    backbone_cached_step_graph_.reset();
    decoder_prefill_graph_.reset();
    embedding_constants_.reset();
    decoder_constants_.reset();
    backbone_constants_.reset();
    weights_.reset();
    if (backend_ != nullptr) {
        ggml_backend_free(backend_);
    }
}

const HeartMuLaAssets & HeartMuLaWeightsRuntime::assets() const noexcept {
    return *assets_;
}

const HeartMuLaWeights & HeartMuLaWeightsRuntime::weights() const noexcept {
    return *weights_;
}

ggml_backend_t HeartMuLaWeightsRuntime::backend() const noexcept {
    return backend_;
}

core::BackendType HeartMuLaWeightsRuntime::backend_type() const noexcept {
    return backend_type_;
}

int HeartMuLaWeightsRuntime::device() const noexcept {
    return device_;
}

core::ConstantTensorCache & HeartMuLaWeightsRuntime::backbone_constants() const noexcept {
    return *backbone_constants_;
}

core::ConstantTensorCache & HeartMuLaWeightsRuntime::decoder_constants() const noexcept {
    return *decoder_constants_;
}

core::ConstantTensorCache & HeartMuLaWeightsRuntime::embedding_constants() const noexcept {
    return *embedding_constants_;
}

int HeartMuLaWeightsRuntime::threads() const noexcept {
    return threads_;
}

size_t HeartMuLaWeightsRuntime::backbone_prefill_graph_arena_bytes() const noexcept {
    return backbone_prefill_graph_arena_bytes_;
}

size_t HeartMuLaWeightsRuntime::backbone_step_graph_arena_bytes() const noexcept {
    return backbone_step_graph_arena_bytes_;
}

size_t HeartMuLaWeightsRuntime::decoder_prefill_graph_arena_bytes() const noexcept {
    return decoder_prefill_graph_arena_bytes_;
}

size_t HeartMuLaWeightsRuntime::decoder_step_graph_arena_bytes() const noexcept {
    return decoder_step_graph_arena_bytes_;
}

size_t HeartMuLaWeightsRuntime::frame_embedding_graph_arena_bytes() const noexcept {
    return frame_embedding_graph_arena_bytes_;
}

HeartMuLaBackbonePrefillOutput HeartMuLaWeightsRuntime::backbone_prefill_embeddings(
    const std::vector<float> & embeddings,
    int64_t batch_size,
    int64_t steps) const {
    const auto & config = assets_->mula_config.backbone;
    if (batch_size <= 0 || steps <= 0) {
        throw std::runtime_error("HeartMuLa backbone prefill requires positive batch and step counts");
    }
    if (steps > config.max_seq_len) {
        throw std::runtime_error("HeartMuLa backbone prefill exceeds model context length");
    }
    if (static_cast<int64_t>(embeddings.size()) != batch_size * steps * config.embed_dim) {
        throw std::runtime_error("HeartMuLa backbone prefill embedding payload size mismatch");
    }
    if (backbone_prefill_graph_ == nullptr ||
        !backbone_prefill_graph_->matches(*this, batch_size, steps)) {
        backbone_prefill_graph_ = std::make_unique<HeartMuLaBackbonePrefillGraph>(*this, batch_size, steps);
    }
    return backbone_prefill_graph_->run(embeddings);
}

void HeartMuLaWeightsRuntime::reset_backbone_cached_state(
    HeartMuLaBackboneCachedState & state,
    runtime::TransformerKVState prefill_state) const {
    state.prefill_steps_ = prefill_state.current_end;
    state.pending_state_ = std::move(prefill_state);
    state.graph_has_state_ = false;
}

HeartMuLaBackboneResult HeartMuLaWeightsRuntime::backbone_cached_step(
    const std::vector<float> & embedding,
    int64_t batch_size,
    HeartMuLaBackboneCachedState & state,
    int64_t cache_capacity) const {
    const auto & config = assets_->mula_config.backbone;
    if (batch_size <= 0) {
        throw std::runtime_error("HeartMuLa backbone cached step requires positive batch size");
    }
    if (cache_capacity <= 0) {
        throw std::runtime_error("HeartMuLa backbone cached step requires positive cache capacity");
    }
    if (cache_capacity > config.max_seq_len) {
        throw std::runtime_error("HeartMuLa backbone cached step exceeds model context length");
    }
    if (static_cast<int64_t>(embedding.size()) != batch_size * config.embed_dim) {
        throw std::runtime_error("HeartMuLa backbone cached step embedding payload size mismatch");
    }
    const int64_t current_end = state.graph_has_state_ && backbone_cached_step_graph_ != nullptr
        ? backbone_cached_step_graph_->current_end()
        : state.pending_state_.current_end;
    const int64_t required_capacity = current_end + 1;
    if (required_capacity > config.max_seq_len) {
        throw std::runtime_error("HeartMuLa backbone cached step capacity exceeds model context length");
    }
    const int64_t prompt_steps = state.prefill_steps_;
    const int64_t old_generated_capacity = backbone_cached_step_graph_ == nullptr
        ? 0
        : std::max<int64_t>(1, backbone_cached_step_graph_->cache_steps() - prompt_steps);
    const int64_t required_generated_capacity = std::max<int64_t>(1, required_capacity - prompt_steps);
    const int64_t max_generated_capacity = std::max<int64_t>(1, cache_capacity - prompt_steps);
    const int64_t next_generated_capacity = std::min<int64_t>(
        max_generated_capacity,
        std::max<int64_t>(
            required_generated_capacity,
            old_generated_capacity > 0 ? old_generated_capacity * 2 : kInitialBackboneGeneratedCacheFrames));
    const int64_t graph_capacity = prompt_steps + next_generated_capacity;
    if (backbone_cached_step_graph_ != nullptr && state.graph_has_state_ &&
        !backbone_cached_step_graph_->can_run(*this, batch_size, required_capacity)) {
        state.pending_state_ = backbone_cached_step_graph_->export_state();
        state.graph_has_state_ = false;
    }
    if (backbone_cached_step_graph_ == nullptr ||
        !backbone_cached_step_graph_->can_run(*this, batch_size, required_capacity)) {
        backbone_cached_step_graph_.reset();
        backbone_cached_step_graph_ = std::make_unique<HeartMuLaBackboneCachedStepGraph>(
            *this,
            batch_size,
            graph_capacity);
    }
    if (!state.graph_has_state_) {
        backbone_cached_step_graph_->import_state(state.pending_state_);
        state.pending_state_ = {};
        state.graph_has_state_ = true;
    }
    return backbone_cached_step_graph_->run_step(embedding);
}

HeartMuLaDecoderPrefillOutput HeartMuLaWeightsRuntime::decoder_prefill_embeddings(
    const std::vector<float> & embeddings,
    int64_t batch_size,
    int64_t steps,
    int64_t codebook_index) const {
    const auto & full_config = assets_->mula_config;
    if (batch_size <= 0 || steps <= 0) {
        throw std::runtime_error("HeartMuLa decoder prefill requires positive batch and step counts");
    }
    if (steps > full_config.decoder.max_seq_len) {
        throw std::runtime_error("HeartMuLa decoder prefill exceeds model context length");
    }
    if (codebook_index < 0 || codebook_index >= full_config.audio_num_codebooks - 1) {
        throw std::runtime_error("HeartMuLa decoder prefill codebook index is out of range");
    }
    if (static_cast<int64_t>(embeddings.size()) != batch_size * steps * full_config.backbone.embed_dim) {
        throw std::runtime_error("HeartMuLa decoder prefill embedding payload size mismatch");
    }
    if (decoder_prefill_graph_ == nullptr ||
        !decoder_prefill_graph_->matches(*this, batch_size, steps)) {
        decoder_prefill_graph_ = std::make_unique<HeartMuLaDecoderPrefillGraph>(*this, batch_size, steps);
    }
    return decoder_prefill_graph_->run(embeddings, codebook_index);
}

void HeartMuLaWeightsRuntime::reset_decoder_cached_state(
    HeartMuLaDecoderCachedState & state,
    runtime::TransformerKVState prefill_state) const {
    state.pending_state_ = std::move(prefill_state);
    state.graph_has_state_ = false;
}

HeartMuLaDecoderResult HeartMuLaWeightsRuntime::decoder_cached_step(
    const std::vector<float> & embedding,
    int64_t batch_size,
    int64_t codebook_index,
    HeartMuLaDecoderCachedState & state,
    int64_t cache_capacity) const {
    const auto & full_config = assets_->mula_config;
    if (batch_size <= 0) {
        throw std::runtime_error("HeartMuLa decoder cached step requires positive batch size");
    }
    if (cache_capacity <= 0) {
        throw std::runtime_error("HeartMuLa decoder cached step requires positive cache capacity");
    }
    if (cache_capacity > full_config.decoder.max_seq_len) {
        throw std::runtime_error("HeartMuLa decoder cached step exceeds model context length");
    }
    if (codebook_index < 0 || codebook_index >= full_config.audio_num_codebooks - 1) {
        throw std::runtime_error("HeartMuLa decoder cached step codebook index is out of range");
    }
    if (static_cast<int64_t>(embedding.size()) != batch_size * full_config.backbone.embed_dim) {
        throw std::runtime_error("HeartMuLa decoder cached step embedding payload size mismatch");
    }
    const int64_t current_end = state.graph_has_state_ && decoder_cached_step_graph_ != nullptr
        ? decoder_cached_step_graph_->current_end()
        : state.pending_state_.current_end;
    const int64_t required_capacity = std::max<int64_t>(cache_capacity, current_end + 1);
    if (required_capacity > full_config.decoder.max_seq_len) {
        throw std::runtime_error("HeartMuLa decoder cached step capacity exceeds model context length");
    }
    if (decoder_cached_step_graph_ != nullptr && state.graph_has_state_ &&
        !decoder_cached_step_graph_->can_run(*this, batch_size, required_capacity)) {
        state.pending_state_ = decoder_cached_step_graph_->export_state();
        state.graph_has_state_ = false;
    }
    if (decoder_cached_step_graph_ == nullptr ||
        !decoder_cached_step_graph_->can_run(*this, batch_size, required_capacity)) {
        decoder_cached_step_graph_.reset();
        decoder_cached_step_graph_ = std::make_unique<HeartMuLaDecoderCachedStepGraph>(
            *this,
            batch_size,
            required_capacity);
    }
    if (!state.graph_has_state_) {
        decoder_cached_step_graph_->import_state(state.pending_state_);
        state.pending_state_ = {};
        state.graph_has_state_ = true;
    }
    return decoder_cached_step_graph_->run_step(embedding, codebook_index);
}

HeartMuLaMergedEmbeddings HeartMuLaWeightsRuntime::merge_frame_embeddings(
    const HeartMuLaFrameEmbeddingInputs & inputs) const {
    const auto & config = assets_->mula_config;
    if (inputs.batch_size <= 0 || inputs.steps <= 0) {
        throw std::runtime_error("HeartMuLa frame embedding requires positive batch and step counts");
    }
    if (inputs.steps > config.backbone.max_seq_len) {
        throw std::runtime_error("HeartMuLa frame embedding exceeds backbone context length");
    }
    const size_t audio_count = static_cast<size_t>(inputs.batch_size * inputs.steps * config.audio_num_codebooks);
    const size_t text_count = static_cast<size_t>(inputs.batch_size * inputs.steps);
    if (inputs.audio_token_ids.size() != audio_count || inputs.audio_mask.size() != audio_count ||
        inputs.text_token_ids.size() != text_count || inputs.text_cond_mask.size() != text_count ||
        inputs.text_uncond_mask.size() != text_count) {
        throw std::runtime_error("HeartMuLa frame embedding payload size mismatch");
    }
    if (inputs.apply_muq) {
        if (inputs.muq_embed.size() != static_cast<size_t>(inputs.batch_size * config.muq_dim) ||
            inputs.muq_cond_mask.size() != static_cast<size_t>(inputs.batch_size) ||
            inputs.muq_uncond_mask.size() != static_cast<size_t>(inputs.batch_size)) {
            throw std::runtime_error("HeartMuLa frame embedding MuQ payload size mismatch");
        }
    }
    auto & graph = inputs.apply_muq ? prompt_frame_embedding_graph_ : step_frame_embedding_graph_;
    if (graph == nullptr ||
        !graph->matches(*this, inputs.batch_size, inputs.steps, inputs.apply_muq)) {
        graph = std::make_unique<HeartMuLaFrameEmbeddingGraph>(
            *this,
            inputs.batch_size,
            inputs.steps,
            inputs.apply_muq);
    }
    return graph->run(inputs);
}

void HeartMuLaWeightsRuntime::release_graph_workspaces() const {
    if (backbone_prefill_graph_ != nullptr) {
        backbone_prefill_graph_->release_workspace();
    }
    if (backbone_cached_step_graph_ != nullptr) {
        backbone_cached_step_graph_->release_workspace();
    }
    if (decoder_prefill_graph_ != nullptr) {
        decoder_prefill_graph_->release_workspace();
    }
    if (decoder_cached_step_graph_ != nullptr) {
        decoder_cached_step_graph_->release_workspace();
    }
    if (prompt_frame_embedding_graph_ != nullptr) {
        prompt_frame_embedding_graph_->release_workspace();
    }
    if (step_frame_embedding_graph_ != nullptr) {
        step_frame_embedding_graph_->release_workspace();
    }
}

void HeartMuLaWeightsRuntime::clear_graph_cache() const {
    backbone_prefill_graph_.reset();
    backbone_cached_step_graph_.reset();
    decoder_prefill_graph_.reset();
    decoder_cached_step_graph_.reset();
    prompt_frame_embedding_graph_.reset();
    step_frame_embedding_graph_.reset();
}

}  // namespace engine::models::heartmula
