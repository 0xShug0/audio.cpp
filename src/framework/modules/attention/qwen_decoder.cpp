#include "engine/framework/modules/attention/qwen_decoder.h"

#include "attention_internal.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <cmath>
#include <stdexcept>

namespace engine::modules {
namespace {

using namespace attention::internal;

inline const core::ModulePortSpec kQwenDecoderInputs[] = {
    {"input", core::PortKind::Activation, false},
};

inline const core::ModulePortSpec kQwenDecoderOutputs[] = {
    {"output", core::PortKind::Activation, false},
};

inline const core::ModuleSchema kQwenDecoderLayerSchema = {
    "QwenDecoderLayer",
    "nn.block",
    kQwenDecoderInputs,
    1,
    kQwenDecoderOutputs,
    1,
    "Qwen-style decoder block with grouped-query attention, RoPE, q/k RMSNorm, and SwiGLU MLP.",
};

int64_t require_head_dim(const QwenDecoderLayerConfig & config) {
    if (config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 || config.head_dim <= 0) {
        throw std::runtime_error("QwenDecoderLayerConfig attention dimensions must be positive");
    }
    if (config.hidden_size <= 0 || config.intermediate_size <= 0) {
        throw std::runtime_error("QwenDecoderLayerConfig hidden sizes must be positive");
    }
    if (config.num_attention_heads % config.num_key_value_heads != 0) {
        throw std::runtime_error("QwenDecoderLayerConfig.num_attention_heads must be divisible by num_key_value_heads");
    }
    return config.head_dim;
}

core::TensorValue reshape_qwen_heads(
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
    auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    const int64_t batch = contiguous.shape.dims[0];
    const int64_t kv_heads = contiguous.shape.dims[1];
    const int64_t steps = contiguous.shape.dims[2];
    const int64_t dim = contiguous.shape.dims[3];
    auto expanded = core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({batch, kv_heads, 1, steps * dim}));
    expanded = RepeatModule({core::TensorShape::from_dims({batch, kv_heads, repeats, steps * dim})})
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
    const MatMulModule matmul;
    auto scores = matmul.build(ctx, q_heads, TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
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

core::TensorValue attention_from_grouped_query_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    int64_t attention_heads,
    int64_t key_value_heads,
    const core::TensorValue & attention_mask) {
    const int64_t repeats = attention_heads / key_value_heads;
    std::vector<core::TensorValue> head_outputs;
    head_outputs.reserve(static_cast<size_t>(attention_heads));
    for (int64_t head = 0; head < attention_heads; ++head) {
        const int64_t key_value_head = head / repeats;
        auto q_head = SliceModule({1, head, 1}).build(ctx, q_heads);
        auto k_head = SliceModule({1, key_value_head, 1}).build(ctx, k_heads);
        auto v_head = SliceModule({1, key_value_head, 1}).build(ctx, v_heads);
        head_outputs.push_back(attention_from_heads(ctx, q_head, k_head, v_head, dim, attention_mask));
    }
    return concat_all(ctx, head_outputs, 1);
}

core::TensorValue flash_attention_from_grouped_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const core::TensorValue & attention_mask,
    ggml_prec precision) {
    if (!core::has_backend_addressable_layout(q_heads.tensor) ||
        !core::has_backend_addressable_layout(k_heads.tensor) ||
        !core::has_backend_addressable_layout(v_heads.tensor)) {
        throw std::runtime_error("Qwen decoder flash attention expects contiguous Q/K/V heads");
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
    ggml_flash_attn_ext_set_prec(flash, precision);
    return core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({q_heads.shape.dims[0], q_heads.shape.dims[2], q_heads.shape.dims[1], dim}),
        GGML_TYPE_F32);
}

core::TensorValue flash_attention_from_grouped_heads_view_kv(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const core::TensorValue & attention_mask,
    ggml_prec precision) {
    const auto q_contiguous = core::ensure_backend_addressable_layout(ctx, q_heads);
    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q_contiguous.tensor,
        k_heads.tensor,
        v_heads.tensor,
        attention_mask.tensor,
        1.0F / std::sqrt(static_cast<float>(dim)),
        0.0F,
        0.0F);
    ggml_flash_attn_ext_set_prec(flash, precision);
    return core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({q_contiguous.shape.dims[0], q_contiguous.shape.dims[2], q_contiguous.shape.dims[1], dim}),
        GGML_TYPE_F32);
}

core::TensorValue cache_view(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    int64_t start,
    int64_t steps,
    int64_t heads,
    int64_t dim) {
    if (start < 0 || steps <= 0 || start + steps > cache.shape.dims[1]) {
        throw std::runtime_error("Qwen decoder cache view range is invalid");
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

LinearWeights require_linear(const LinearWeights & weights, bool use_bias, const char * name) {
    if (use_bias && !weights.bias.has_value()) {
        throw std::runtime_error(std::string(name) + " bias is required");
    }
    return weights;
}

}  // namespace

QwenDecoderLayerConfig qwen_decoder_layer_config_from_stack(const QwenDecoderStackConfig & config) {
    QwenDecoderLayerConfig out;
    out.hidden_size = config.hidden_size;
    out.num_attention_heads = config.num_attention_heads;
    out.num_key_value_heads = config.num_key_value_heads;
    out.head_dim = config.head_dim;
    out.intermediate_size = config.intermediate_size;
    out.rms_norm_eps = config.rms_norm_eps;
    out.rope_theta = config.rope_theta;
    out.attention_precision = config.attention_precision;
    out.projection_precision = config.projection_precision;
    out.use_qk_norm = config.use_qk_norm;
    out.runtime = config.runtime;
    return out;
}

core::TensorValue build_mlp(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const QwenDecoderLayerConfig & config,
    const QwenMLPWeights & weights) {
    auto gate = LinearModule(
                    {
                        config.hidden_size,
                        config.intermediate_size,
                        weights.gate_proj.bias.has_value(),
                        config.projection_precision,
                    })
                    .build(ctx, input, require_linear(weights.gate_proj, false, "QwenMLPWeights.gate_proj"));
    gate = SiluModule{}.build(ctx, gate);
    auto up = LinearModule(
                  {
                      config.hidden_size,
                      config.intermediate_size,
                      weights.up_proj.bias.has_value(),
                      config.projection_precision,
                  })
                  .build(ctx, input, require_linear(weights.up_proj, false, "QwenMLPWeights.up_proj"));
    auto gated = MulModule{}.build(ctx, gate, up);
    return LinearModule(
               {
                   config.intermediate_size,
                   config.hidden_size,
                   weights.down_proj.bias.has_value(),
                   config.projection_precision,
               })
        .build(ctx, gated, require_linear(weights.down_proj, false, "QwenMLPWeights.down_proj"));
}

QwenDecoderLayerModule::QwenDecoderLayerModule(QwenDecoderLayerConfig config) : config_(config) {
    require_head_dim(config_);
}

const QwenDecoderLayerConfig & QwenDecoderLayerModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & QwenDecoderLayerModule::schema() const noexcept {
    return static_schema();
}

QwenDecoderLayerOutputs QwenDecoderLayerModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const QwenDecoderLayerWeights & weights,
    const std::optional<core::TensorValue> & prefix_key,
    const std::optional<core::TensorValue> & prefix_value,
    const std::optional<core::TensorValue> & attention_mask) const {
    validate_sequence_input(input, config_.hidden_size, "input");
    if (prefix_key.has_value() != prefix_value.has_value()) {
        throw std::runtime_error("Qwen decoder layer requires both prefix_key and prefix_value or neither");
    }

    const int64_t dim = require_head_dim(config_);
    const int64_t kv_repeats = config_.num_attention_heads / config_.num_key_value_heads;

    auto x_norm = RMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false})
                      .build(ctx, input, weights.input_norm);
    auto q = LinearModule(
                 {
                     config_.hidden_size,
                     config_.num_attention_heads * dim,
                     weights.self_attention.q_bias.has_value(),
                     config_.projection_precision,
                 })
                 .build(
                     ctx,
                     x_norm,
                     {weights.self_attention.q_weight, weights.self_attention.q_bias});
    auto k = LinearModule(
                 {
                     config_.hidden_size,
                     config_.num_key_value_heads * dim,
                     weights.self_attention.k_bias.has_value(),
                     config_.projection_precision,
                 })
                 .build(
                     ctx,
                     x_norm,
                     {weights.self_attention.k_weight, weights.self_attention.k_bias});
    auto v = LinearModule(
                 {
                     config_.hidden_size,
                     config_.num_key_value_heads * dim,
                     weights.self_attention.v_bias.has_value(),
                     config_.projection_precision,
                 })
                 .build(
                     ctx,
                     x_norm,
                     {weights.self_attention.v_weight, weights.self_attention.v_bias});

    q = reshape_qwen_heads(ctx, q, config_.num_attention_heads, dim);
    k = reshape_qwen_heads(ctx, k, config_.num_key_value_heads, dim);
    if (config_.use_qk_norm) {
        q = RMSNormModule({dim, config_.rms_norm_eps, true, false}).build(ctx, q, weights.q_norm);
        k = RMSNormModule({dim, config_.rms_norm_eps, true, false}).build(ctx, k, weights.k_norm);
    }
    v = reshape_qwen_heads(ctx, v, config_.num_key_value_heads, dim);

    q = RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config_.rope_theta}).build(ctx, q, positions);
    k = RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config_.rope_theta}).build(ctx, k, positions);
    k = core::ensure_backend_addressable_layout(ctx, k);
    v = core::ensure_backend_addressable_layout(ctx, v);

    auto q_heads = TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto all_k = prefix_key.has_value() ? ConcatModule({1}).build(ctx, *prefix_key, k) : k;
    auto all_v = prefix_value.has_value() ? ConcatModule({1}).build(ctx, *prefix_value, v) : v;
    core::TensorValue context;
    if (!prefix_key.has_value() && attention_mask.has_value() &&
        config_.runtime.attention.prefill_mode == QwenDecoderAttentionMode::FlashGroupedViewKV) {
        q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
        auto k_heads = TransposeModule({{0, 2, 1, 3}, all_k.shape.rank}).build(ctx, all_k);
        auto v_heads = TransposeModule({{0, 2, 1, 3}, all_v.shape.rank}).build(ctx, all_v);
        context = flash_attention_from_grouped_heads_view_kv(
            ctx,
            q_heads,
            k_heads,
            v_heads,
            dim,
            *attention_mask,
            config_.attention_precision);
    } else if (!prefix_key.has_value() && attention_mask.has_value() &&
               config_.runtime.attention.prefill_mode == QwenDecoderAttentionMode::FlashGrouped) {
        q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
        auto k_heads = TransposeModule({{0, 2, 1, 3}, all_k.shape.rank}).build(ctx, all_k);
        auto v_heads = TransposeModule({{0, 2, 1, 3}, all_v.shape.rank}).build(ctx, all_v);
        k_heads = core::wrap_tensor(ggml_cont(ctx.ggml, k_heads.tensor), k_heads.shape, k_heads.type);
        v_heads = core::wrap_tensor(ggml_cont(ctx.ggml, v_heads.tensor), v_heads.shape, v_heads.type);
        context = flash_attention_from_grouped_heads(
            ctx,
            q_heads,
            k_heads,
            v_heads,
            dim,
            *attention_mask,
            config_.attention_precision);
    } else {
        auto k_heads = repeat_kv_heads(
            ctx,
            TransposeModule({{0, 2, 1, 3}, all_k.shape.rank}).build(ctx, all_k),
            kv_repeats);
        auto v_heads = repeat_kv_heads(
            ctx,
            TransposeModule({{0, 2, 1, 3}, all_v.shape.rank}).build(ctx, all_v),
            kv_repeats);
        context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
        context = TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    }
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config_.num_attention_heads * dim}));

    auto attn_out = LinearModule(
                        {
                            config_.num_attention_heads * dim,
                            config_.hidden_size,
                            weights.self_attention.out_bias.has_value(),
                            config_.projection_precision,
                        })
                        .build(
                            ctx,
                            context,
                            {weights.self_attention.out_weight, weights.self_attention.out_bias});
    auto x = AddModule{}.build(ctx, input, attn_out);

    auto ff_in = RMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false})
                     .build(ctx, x, weights.post_norm);
    auto ff = build_mlp(ctx, ff_in, config_, weights.mlp);
    return {AddModule{}.build(ctx, x, ff), k, v};
}

QwenDecoderLayerOutputs QwenDecoderLayerModule::build_with_static_cache_tail(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const QwenDecoderLayerWeights & weights,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const std::optional<core::TensorValue> & cache_slot,
    const core::TensorValue & attention_mask) const {
    validate_sequence_input(input, config_.hidden_size, "input");
    const int64_t dim = require_head_dim(config_);
    const int64_t kv_repeats = config_.num_attention_heads / config_.num_key_value_heads;

    auto x_norm = RMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false})
                      .build(ctx, input, weights.input_norm);
    auto q = LinearModule(
                 {
                     config_.hidden_size,
                     config_.num_attention_heads * dim,
                     weights.self_attention.q_bias.has_value(),
                     config_.projection_precision,
                 })
                 .build(
                     ctx,
                     x_norm,
                     {weights.self_attention.q_weight, weights.self_attention.q_bias});
    auto k = LinearModule(
                 {
                     config_.hidden_size,
                     config_.num_key_value_heads * dim,
                     weights.self_attention.k_bias.has_value(),
                     config_.projection_precision,
                 })
                 .build(
                     ctx,
                     x_norm,
                     {weights.self_attention.k_weight, weights.self_attention.k_bias});
    auto v = LinearModule(
                 {
                     config_.hidden_size,
                     config_.num_key_value_heads * dim,
                     weights.self_attention.v_bias.has_value(),
                     config_.projection_precision,
                 })
                 .build(
                     ctx,
                     x_norm,
                     {weights.self_attention.v_weight, weights.self_attention.v_bias});

    q = reshape_qwen_heads(ctx, q, config_.num_attention_heads, dim);
    k = reshape_qwen_heads(ctx, k, config_.num_key_value_heads, dim);
    if (config_.use_qk_norm) {
        q = RMSNormModule({dim, config_.rms_norm_eps, true, false}).build(ctx, q, weights.q_norm);
        k = RMSNormModule({dim, config_.rms_norm_eps, true, false}).build(ctx, k, weights.k_norm);
    }
    v = reshape_qwen_heads(ctx, v, config_.num_key_value_heads, dim);

    q = RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config_.rope_theta}).build(ctx, q, positions);
    k = RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config_.rope_theta}).build(ctx, k, positions);
    k = core::ensure_backend_addressable_layout(ctx, k);
    v = core::ensure_backend_addressable_layout(ctx, v);

    core::TensorValue attention_key_cache = cache_key;
    core::TensorValue attention_value_cache = cache_value;
    core::TensorValue stored_key = k;
    core::TensorValue stored_value = v;
    if (config_.runtime.static_cache.update_mode == QwenDecoderStaticCacheUpdateMode::DirectSetRows) {
        if (!cache_slot.has_value()) {
            throw std::runtime_error("Qwen decoder direct static-cache update requires cache_slot");
        }
        const FastKVSetRowsModule set_rows;
        attention_key_cache = set_rows.build(ctx, cache_key, k, *cache_slot);
        attention_value_cache = set_rows.build(ctx, cache_value, v, *cache_slot);
    } else {
        const int64_t scratch_slot = cache_key.shape.dims[1] - 1;
        stored_key = cache_view(ctx, cache_key, scratch_slot, 1, config_.num_key_value_heads, dim);
        stored_value = cache_view(ctx, cache_value, scratch_slot, 1, config_.num_key_value_heads, dim);
        ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, k.tensor, stored_key.tensor));
        ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, v.tensor, stored_value.tensor));
    }

    auto q_heads = TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    auto k_heads = TransposeModule({{0, 2, 1, 3}, attention_key_cache.shape.rank}).build(ctx, attention_key_cache);
    auto v_heads = TransposeModule({{0, 2, 1, 3}, attention_value_cache.shape.rank}).build(ctx, attention_value_cache);
    core::TensorValue context;
    const bool use_grouped_query =
        config_.runtime.attention.static_mode == QwenDecoderAttentionMode::ManualRepeatThenGroupedQuery &&
        config_.runtime.attention.grouped_query_min_steps > 0 &&
        cache_key.shape.dims[1] >= config_.runtime.attention.grouped_query_min_steps &&
        kv_repeats > 1;
    if (use_grouped_query) {
        k_heads = core::ensure_backend_addressable_layout(ctx, k_heads);
        v_heads = core::ensure_backend_addressable_layout(ctx, v_heads);
        context = attention_from_grouped_query_heads(
            ctx,
            q_heads,
            k_heads,
            v_heads,
            dim,
            config_.num_attention_heads,
            config_.num_key_value_heads,
            attention_mask);
    } else if (config_.runtime.attention.static_mode == QwenDecoderAttentionMode::ManualRepeat ||
               config_.runtime.attention.static_mode == QwenDecoderAttentionMode::ManualRepeatThenGroupedQuery) {
        k_heads = repeat_kv_heads(ctx, k_heads, kv_repeats);
        v_heads = repeat_kv_heads(ctx, v_heads, kv_repeats);
        k_heads = core::wrap_tensor(ggml_cont(ctx.ggml, k_heads.tensor), k_heads.shape, k_heads.type);
        v_heads = core::wrap_tensor(ggml_cont(ctx.ggml, v_heads.tensor), v_heads.shape, v_heads.type);
        context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
    } else if (config_.runtime.attention.static_mode == QwenDecoderAttentionMode::FlashGroupedViewKV) {
        context = flash_attention_from_grouped_heads_view_kv(
            ctx,
            q_heads,
            k_heads,
            v_heads,
            dim,
            attention_mask,
            config_.attention_precision);
    } else {
        k_heads = core::wrap_tensor(ggml_cont(ctx.ggml, k_heads.tensor), k_heads.shape, k_heads.type);
        v_heads = core::wrap_tensor(ggml_cont(ctx.ggml, v_heads.tensor), v_heads.shape, v_heads.type);
        context = flash_attention_from_grouped_heads(
            ctx,
            q_heads,
            k_heads,
            v_heads,
            dim,
            attention_mask,
            config_.attention_precision);
    }
    if (config_.runtime.static_cache.transpose_context) {
        context = TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    }
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({1, 1, config_.num_attention_heads * dim}));

    auto attn_out = LinearModule(
                        {
                            config_.num_attention_heads * dim,
                            config_.hidden_size,
                            weights.self_attention.out_bias.has_value(),
                            config_.projection_precision,
                        })
                        .build(
                            ctx,
                            context,
                            {weights.self_attention.out_weight, weights.self_attention.out_bias});
    auto x = AddModule{}.build(ctx, input, attn_out);

    auto ff_in = RMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false})
                     .build(ctx, x, weights.post_norm);
    auto ff = build_mlp(ctx, ff_in, config_, weights.mlp);
    return {AddModule{}.build(ctx, x, ff), stored_key, stored_value};
}

const core::ModuleSchema & QwenDecoderLayerModule::static_schema() noexcept {
    return kQwenDecoderLayerSchema;
}

QwenDecoderStackModule::QwenDecoderStackModule(QwenDecoderStackConfig config) : config_(config) {
    if (config_.layers <= 0) {
        throw std::runtime_error("QwenDecoderStackConfig.layers must be positive");
    }
    require_head_dim(qwen_decoder_layer_config_from_stack(config_));
}

const QwenDecoderStackConfig & QwenDecoderStackModule::config() const noexcept {
    return config_;
}

QwenDecoderStackOutputs QwenDecoderStackModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const QwenDecoderStackWeights & weights,
    const std::optional<QwenDecoderStackState> & prefix_state,
    const std::optional<core::TensorValue> & attention_mask) const {
    if (static_cast<int64_t>(weights.layers.size()) != config_.layers) {
        throw std::runtime_error("QwenDecoderStackWeights layer count does not match config.layers");
    }
    if (prefix_state.has_value() && static_cast<int64_t>(prefix_state->layers.size()) != config_.layers) {
        throw std::runtime_error("QwenDecoderStackState layer count does not match config.layers");
    }

    auto output = input;
    QwenDecoderStackState state;
    state.layers.reserve(weights.layers.size());
    const QwenDecoderLayerModule layer_module(qwen_decoder_layer_config_from_stack(config_));
    for (size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
        const auto * layer_prefix = prefix_state.has_value() ? &prefix_state->layers[layer_index] : nullptr;
        auto layer = layer_module.build(
            ctx,
            output,
            positions,
            weights.layers[layer_index],
            layer_prefix != nullptr ? layer_prefix->key : std::nullopt,
            layer_prefix != nullptr ? layer_prefix->value : std::nullopt,
            attention_mask);
        output = layer.output;
        state.layers.push_back({layer.key, layer.value});
    }
    return {output, std::move(state)};
}

}  // namespace engine::modules
