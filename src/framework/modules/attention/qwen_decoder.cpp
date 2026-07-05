#include "engine/framework/modules/attention/qwen_decoder.h"

#include "attention_internal.h"

#include "engine/framework/modules/activation_modules.h"
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
    std::vector<core::TensorValue> heads;
    heads.reserve(static_cast<size_t>(input.shape.dims[1] * repeats));
    for (int64_t head = 0; head < input.shape.dims[1]; ++head) {
        auto one = SliceModule({1, head, 1}).build(ctx, input);
        for (int64_t rep = 0; rep < repeats; ++rep) {
            heads.push_back(one);
        }
    }
    return concat_all(ctx, heads, 1);
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

QwenDecoderLayerConfig make_layer_config(const QwenDecoderStackConfig & config) {
    return {
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.intermediate_size,
        config.rms_norm_eps,
        config.rope_theta,
        config.attention_precision,
        config.projection_precision,
    };
}

core::TensorValue build_mlp(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const QwenDecoderLayerConfig & config,
    const QwenMLPWeights & weights) {
    auto gate = LinearModule(
                    {config.hidden_size, config.intermediate_size, false, config.projection_precision})
                    .build(ctx, input, require_linear(weights.gate_proj, false, "QwenMLPWeights.gate_proj"));
    gate = SiluModule{}.build(ctx, gate);
    auto up = LinearModule(
                  {config.hidden_size, config.intermediate_size, false, config.projection_precision})
                  .build(ctx, input, require_linear(weights.up_proj, false, "QwenMLPWeights.up_proj"));
    auto gated = MulModule{}.build(ctx, gate, up);
    return LinearModule(
               {config.intermediate_size, config.hidden_size, false, config.projection_precision})
        .build(ctx, gated, require_linear(weights.down_proj, false, "QwenMLPWeights.down_proj"));
}

}  // namespace

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
                 {config_.hidden_size, config_.num_attention_heads * dim, false, config_.projection_precision})
                 .build(
                     ctx,
                     x_norm,
                     {weights.self_attention.q_weight, std::nullopt});
    auto k = LinearModule(
                 {config_.hidden_size, config_.num_key_value_heads * dim, false, config_.projection_precision})
                 .build(
                     ctx,
                     x_norm,
                     {weights.self_attention.k_weight, std::nullopt});
    auto v = LinearModule(
                 {config_.hidden_size, config_.num_key_value_heads * dim, false, config_.projection_precision})
                 .build(
                     ctx,
                     x_norm,
                     {weights.self_attention.v_weight, std::nullopt});

    q = RMSNormModule({dim, config_.rms_norm_eps, true, false})
            .build(ctx, reshape_qwen_heads(ctx, q, config_.num_attention_heads, dim), weights.q_norm);
    k = RMSNormModule({dim, config_.rms_norm_eps, true, false})
            .build(ctx, reshape_qwen_heads(ctx, k, config_.num_key_value_heads, dim), weights.k_norm);
    v = reshape_qwen_heads(ctx, v, config_.num_key_value_heads, dim);

    q = RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config_.rope_theta}).build(ctx, q, positions);
    k = RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config_.rope_theta}).build(ctx, k, positions);

    auto q_heads = TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto all_k = prefix_key.has_value() ? ConcatModule({1}).build(ctx, *prefix_key, k) : k;
    auto all_v = prefix_value.has_value() ? ConcatModule({1}).build(ctx, *prefix_value, v) : v;
    auto k_heads = repeat_kv_heads(
        ctx,
        TransposeModule({{0, 2, 1, 3}, all_k.shape.rank}).build(ctx, all_k),
        kv_repeats);
    auto v_heads = repeat_kv_heads(
        ctx,
        TransposeModule({{0, 2, 1, 3}, all_v.shape.rank}).build(ctx, all_v),
        kv_repeats);

    auto context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
    context = TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config_.num_attention_heads * dim}));

    auto attn_out = LinearModule(
                        {config_.num_attention_heads * dim, config_.hidden_size, false, config_.projection_precision})
                        .build(
                            ctx,
                            context,
                            {weights.self_attention.out_weight, std::nullopt});
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
    const core::TensorValue & attention_mask) const {
    validate_sequence_input(input, config_.hidden_size, "input");
    const int64_t dim = require_head_dim(config_);
    const int64_t scratch_slot = cache_key.shape.dims[1] - 1;

    auto x_norm = RMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false})
                      .build(ctx, input, weights.input_norm);
    auto q = LinearModule(
                 {config_.hidden_size, config_.num_attention_heads * dim, false, config_.projection_precision})
                 .build(
                     ctx,
                     x_norm,
                     {weights.self_attention.q_weight, std::nullopt});
    auto k = LinearModule(
                 {config_.hidden_size, config_.num_key_value_heads * dim, false, config_.projection_precision})
                 .build(
                     ctx,
                     x_norm,
                     {weights.self_attention.k_weight, std::nullopt});
    auto v = LinearModule(
                 {config_.hidden_size, config_.num_key_value_heads * dim, false, config_.projection_precision})
                 .build(
                     ctx,
                     x_norm,
                     {weights.self_attention.v_weight, std::nullopt});

    q = RMSNormModule({dim, config_.rms_norm_eps, true, false})
            .build(ctx, reshape_qwen_heads(ctx, q, config_.num_attention_heads, dim), weights.q_norm);
    k = RMSNormModule({dim, config_.rms_norm_eps, true, false})
            .build(ctx, reshape_qwen_heads(ctx, k, config_.num_key_value_heads, dim), weights.k_norm);
    v = reshape_qwen_heads(ctx, v, config_.num_key_value_heads, dim);

    q = RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config_.rope_theta}).build(ctx, q, positions);
    k = RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config_.rope_theta}).build(ctx, k, positions);

    auto key_tail = cache_view(ctx, cache_key, scratch_slot, 1, config_.num_key_value_heads, dim);
    auto value_tail = cache_view(ctx, cache_value, scratch_slot, 1, config_.num_key_value_heads, dim);
    ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, k.tensor, key_tail.tensor));
    ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, v.tensor, value_tail.tensor));

    auto q_heads = TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    auto k_heads = TransposeModule({{0, 2, 1, 3}, cache_key.shape.rank}).build(ctx, cache_key);
    auto v_heads = TransposeModule({{0, 2, 1, 3}, cache_value.shape.rank}).build(ctx, cache_value);
    k_heads = core::wrap_tensor(ggml_cont(ctx.ggml, k_heads.tensor), k_heads.shape, k_heads.type);
    v_heads = core::wrap_tensor(ggml_cont(ctx.ggml, v_heads.tensor), v_heads.shape, v_heads.type);

    auto context = flash_attention_from_grouped_heads(
        ctx,
        q_heads,
        k_heads,
        v_heads,
        dim,
        attention_mask,
        config_.attention_precision);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({1, 1, config_.num_attention_heads * dim}));

    auto attn_out = LinearModule(
                        {config_.num_attention_heads * dim, config_.hidden_size, false, config_.projection_precision})
                        .build(
                            ctx,
                            context,
                            {weights.self_attention.out_weight, std::nullopt});
    auto x = AddModule{}.build(ctx, input, attn_out);

    auto ff_in = RMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false})
                     .build(ctx, x, weights.post_norm);
    auto ff = build_mlp(ctx, ff_in, config_, weights.mlp);
    return {AddModule{}.build(ctx, x, ff), k, v};
}

const core::ModuleSchema & QwenDecoderLayerModule::static_schema() noexcept {
    return kQwenDecoderLayerSchema;
}

QwenDecoderStackModule::QwenDecoderStackModule(QwenDecoderStackConfig config) : config_(config) {
    if (config_.layers <= 0) {
        throw std::runtime_error("QwenDecoderStackConfig.layers must be positive");
    }
    require_head_dim(make_layer_config(config_));
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
    const QwenDecoderLayerModule layer_module(make_layer_config(config_));
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
