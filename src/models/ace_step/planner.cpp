#include "engine/models/ace_step/planner.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/transformers/qwen_decoder.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/models/ace_step/prompt_builder.h"

#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace engine::models::ace_step {
namespace {

namespace modules = engine::modules;

using modules::QwenDecoderLayerWeights;
using modules::QwenDecoderStackWeights;

using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct PlannerWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    QwenDecoderStackWeights layers;
    core::TensorValue norm;
    core::TensorValue lm_head;
};

int64_t count_valid_tokens(const AceStepTokenizedText & tokens) {
    int64_t count = 0;
    for (const int32_t value : tokens.attention_mask) {
        if (value == 0) {
            break;
        }
        ++count;
    }
    if (count <= 0 || count > static_cast<int64_t>(tokens.input_ids.size())) {
        throw std::runtime_error("ACE-Step planner prompt tokenization is invalid");
    }
    return count;
}

AceStepTokenizedText trim_to_valid_tokens(const AceStepTokenizedText & tokens) {
    const int64_t valid = count_valid_tokens(tokens);
    AceStepTokenizedText trimmed;
    trimmed.text = tokens.text;
    trimmed.input_ids.assign(tokens.input_ids.begin(), tokens.input_ids.begin() + valid);
    trimmed.attention_mask.assign(tokens.attention_mask.begin(), tokens.attention_mask.begin() + valid);
    return trimmed;
}

std::pair<AceStepTokenizedText, AceStepTokenizedText> left_pad_pair_to_shared_length(
    const AceStepTokenizedText & first,
    const AceStepTokenizedText & second,
    int32_t pad_token_id) {
    const int64_t first_valid = static_cast<int64_t>(first.input_ids.size());
    const int64_t second_valid = static_cast<int64_t>(second.input_ids.size());
    const int64_t shared = std::max(first_valid, second_valid);
    const auto left_pad = [&](const AceStepTokenizedText & source, int64_t source_valid) {
        AceStepTokenizedText out;
        out.text = source.text;
        out.input_ids.assign(static_cast<size_t>(shared), pad_token_id);
        out.attention_mask.assign(static_cast<size_t>(shared), 0);
        const int64_t offset = shared - source_valid;
        std::copy(source.input_ids.begin(), source.input_ids.end(), out.input_ids.begin() + offset);
        std::fill(out.attention_mask.begin() + offset, out.attention_mask.end(), 1);
        return out;
    };
    return {left_pad(first, first_valid), left_pad(second, second_valid)};
}

std::vector<ggml_fp16_t> build_prefill_attention_mask_values(const std::vector<int32_t> & attention_mask) {
    const int64_t tokens = static_cast<int64_t>(attention_mask.size());
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    const ggml_fp16_t visible = ggml_fp32_to_fp16(0.0F);
    std::vector<ggml_fp16_t> values(static_cast<size_t>(tokens * tokens), visible);
    for (int64_t q = 0; q < tokens; ++q) {
        if (attention_mask[static_cast<size_t>(q)] == 0) {
            for (int64_t k = 0; k < tokens; ++k) {
                values[static_cast<size_t>(q * tokens + k)] = neg_inf;
            }
            values[static_cast<size_t>(q * tokens + q)] = visible;
            continue;
        }
        for (int64_t k = 0; k < tokens; ++k) {
            if (k > q || attention_mask[static_cast<size_t>(k)] == 0) {
                values[static_cast<size_t>(q * tokens + k)] = neg_inf;
            }
        }
    }
    return values;
}

std::vector<ggml_fp16_t> build_cfg_prefill_attention_mask_values(
    const std::vector<int32_t> & conditional_attention_mask,
    const std::vector<int32_t> & unconditional_attention_mask) {
    if (conditional_attention_mask.size() != unconditional_attention_mask.size()) {
        throw std::runtime_error("ACE-Step planner CFG prefill attention-mask size mismatch");
    }
    const auto conditional_values = build_prefill_attention_mask_values(conditional_attention_mask);
    const auto unconditional_values = build_prefill_attention_mask_values(unconditional_attention_mask);
    std::vector<ggml_fp16_t> values;
    values.reserve(conditional_values.size() + unconditional_values.size());
    values.insert(values.end(), conditional_values.begin(), conditional_values.end());
    values.insert(values.end(), unconditional_values.begin(), unconditional_values.end());
    return values;
}

int64_t planner_attention_head_dim(const AceStepPlannerConfig & config) {
    if (config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 || config.head_dim <= 0) {
        throw std::runtime_error("ACE-Step planner attention configuration is invalid");
    }
    return config.head_dim;
}

bool planner_prefill_uses_host_backend(core::BackendType backend_type) {
    switch (backend_type) {
    case core::BackendType::Vulkan:
        // Vulkan keeps the validated host-prefill path for now; it can try the Metal
        // decode-graph prompt prefill path below once parity and performance are checked.
        return true;
    case core::BackendType::Metal:
    case core::BackendType::Cpu:
    case core::BackendType::Cuda:
    case core::BackendType::Hip:
    case core::BackendType::BestAvailable:
        return false;
    }
    return false;
}

core::TensorValue reshape_planner_heads(
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

core::TensorValue sdpa_from_planner_grouped_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const core::TensorValue & attention_mask) {
    if (q_heads.shape.rank != 4 || k_heads.shape.rank != 4 || v_heads.shape.rank != 4) {
        throw std::runtime_error("ACE-Step planner SDPA expects rank-4 Q/K/V tensors");
    }
    if (q_heads.shape.dims[0] != k_heads.shape.dims[0] ||
        q_heads.shape.dims[0] != v_heads.shape.dims[0] ||
        q_heads.shape.dims[3] != k_heads.shape.dims[3] ||
        q_heads.shape.dims[3] != v_heads.shape.dims[3]) {
        throw std::runtime_error("ACE-Step planner SDPA Q/K/V tensor shapes are incompatible");
    }
    const int64_t query_heads = q_heads.shape.dims[1];
    const int64_t key_value_heads = k_heads.shape.dims[1];
    if (query_heads % key_value_heads != 0) {
        throw std::runtime_error("ACE-Step planner SDPA requires attention heads divisible by KV heads");
    }
    const auto contiguous_k = core::ensure_backend_addressable_layout(ctx, k_heads);
    const auto contiguous_v = core::ensure_backend_addressable_layout(ctx, v_heads);
    if (!core::has_backend_addressable_layout(q_heads.tensor) ||
        !core::has_backend_addressable_layout(contiguous_k.tensor) ||
        !core::has_backend_addressable_layout(contiguous_v.tensor)) {
        throw std::runtime_error("ACE-Step planner flash attention expects contiguous Q/K/V heads");
    }
    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q_heads.tensor,
        contiguous_k.tensor,
        contiguous_v.tensor,
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

core::TensorValue ensure_planner_contiguous(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value) {
    return core::ensure_backend_addressable_layout(ctx, value);
}

core::TensorValue zero_masked_query_rows(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & query_mask) {
    core::validate_shape(
        query_mask,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], 1, 1}),
        "query_mask");
    const auto mask = core::wrap_tensor(ggml_repeat(ctx.ggml, query_mask.tensor, input.tensor), input.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_mul(ctx.ggml, input.tensor, mask.tensor), input.shape, GGML_TYPE_F32);
}

ggml_type planner_phase2_activation_type(core::BackendType backend_type) {
    (void) backend_type;
    return GGML_TYPE_F32;
}

core::TensorValue cast_planner_activation(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    ggml_type target_type) {
    (void) ctx;
    (void) target_type;
    return value;
}

core::TensorValue planner_position_slice(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & positions,
    int64_t batch_index,
    int64_t seq_len) {
    if (positions.shape.rank == 1) {
        if (positions.shape.dims[0] == seq_len && batch_index == 0) {
            return positions;
        }
        if (seq_len == 1 && batch_index >= 0 && batch_index < positions.shape.dims[0]) {
            auto scalar = modules::SliceModule({0, batch_index, 1}).build(ctx, positions);
            return core::reshape_tensor(
                ctx,
                core::ensure_backend_addressable_layout(ctx, scalar),
                core::TensorShape::from_dims({1}));
        }
        throw std::runtime_error("ACE-Step planner rank-1 positions do not match requested batch/sequence layout");
    }
    if (positions.shape.rank == 2) {
        auto row = modules::SliceModule({0, batch_index, 1}).build(ctx, positions);
        return core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, row), core::TensorShape::from_dims({seq_len}));
    }
    throw std::runtime_error("ACE-Step planner batched positions must have rank 1 or 2");
}

core::TensorValue apply_planner_rope_batched(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    int64_t dim,
    float rope_theta) {
    if (input.shape.rank != 4) {
        throw std::runtime_error("ACE-Step planner RoPE expects a rank-4 tensor");
    }
    const int64_t batch = input.shape.dims[0];
    core::TensorValue output = {};
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        auto one = modules::SliceModule({0, batch_index, 1}).build(ctx, input);
        auto one_positions = planner_position_slice(ctx, positions, batch_index, input.shape.dims[1]);
        one = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, rope_theta}).build(ctx, one, one_positions);
        output = output.valid() ? modules::ConcatModule({0}).build(ctx, output, one) : one;
    }
    return output;
}

core::TensorValue planner_cache_view(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    int64_t batch_start,
    int64_t batch_count,
    int64_t step_start,
    int64_t step_count,
    int64_t heads,
    int64_t dim) {
    if (batch_start < 0 || batch_count <= 0 || batch_start + batch_count > cache.shape.dims[0]) {
        throw std::runtime_error("ACE-Step planner cache batch range is invalid");
    }
    if (step_start < 0 || step_count <= 0 || step_start + step_count > cache.shape.dims[1]) {
        throw std::runtime_error("ACE-Step planner cache step range is invalid");
    }
    const size_t byte_offset = static_cast<size_t>(batch_start) * cache.tensor->nb[3] +
        static_cast<size_t>(step_start) * cache.tensor->nb[2];
    return core::wrap_tensor(
        ggml_view_4d(
            ctx.ggml,
            cache.tensor,
            dim,
            heads,
            step_count,
            batch_count,
            cache.tensor->nb[1],
            cache.tensor->nb[2],
            cache.tensor->nb[3],
            byte_offset),
        core::TensorShape::from_dims({batch_count, step_count, heads, dim}),
        GGML_TYPE_F32);
}

modules::QwenDecoderLayerOutputs planner_decoder_layer_batched(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const QwenDecoderLayerWeights & weights,
    const AceStepPlannerConfig & config,
    const core::TensorValue & attention_mask,
    const core::TensorValue & query_mask,
    ggml_type activation_type) {
    const int64_t dim = planner_attention_head_dim(config);

    auto x_norm = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, weights.input_norm.bias.has_value()})
                      .build(ctx, input, weights.input_norm);
    auto q = modules::LinearModule({config.hidden_size, config.num_attention_heads * dim, false})
                 .build(ctx, x_norm, {weights.self_attention.q_weight, std::nullopt});
    auto k = modules::LinearModule({config.hidden_size, config.num_key_value_heads * dim, false})
                 .build(ctx, x_norm, {weights.self_attention.k_weight, std::nullopt});
    auto v = modules::LinearModule({config.hidden_size, config.num_key_value_heads * dim, false})
                 .build(ctx, x_norm, {weights.self_attention.v_weight, std::nullopt});
    q = cast_planner_activation(ctx, q, activation_type);
    k = cast_planner_activation(ctx, k, activation_type);
    v = cast_planner_activation(ctx, v, activation_type);

    q = modules::RMSNormModule({dim, config.rms_norm_eps, true, weights.q_norm.bias.has_value()})
            .build(ctx, reshape_planner_heads(ctx, q, config.num_attention_heads, dim), weights.q_norm);
    k = modules::RMSNormModule({dim, config.rms_norm_eps, true, weights.k_norm.bias.has_value()})
            .build(ctx, reshape_planner_heads(ctx, k, config.num_key_value_heads, dim), weights.k_norm);
    v = reshape_planner_heads(ctx, v, config.num_key_value_heads, dim);
    v = cast_planner_activation(ctx, v, activation_type);

    q = apply_planner_rope_batched(ctx, q, positions, dim, config.rope_theta);
    k = apply_planner_rope_batched(ctx, k, positions, dim, config.rope_theta);
    q = cast_planner_activation(ctx, q, activation_type);
    k = cast_planner_activation(ctx, k, activation_type);

    std::array<int, core::kMaxTensorRank> head_axes = {0, 2, 1, 3};
    auto q_heads = modules::TransposeModule({head_axes, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({head_axes, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({head_axes, v.shape.rank}).build(ctx, v);
    q_heads = ensure_planner_contiguous(ctx, q_heads);
    k_heads = ensure_planner_contiguous(ctx, k_heads);
    v_heads = ensure_planner_contiguous(ctx, v_heads);

    auto context = sdpa_from_planner_grouped_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
    context = zero_masked_query_rows(ctx, context, query_mask);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.num_attention_heads * dim}));
    context = cast_planner_activation(ctx, context, activation_type);

    auto attn_out = modules::LinearModule({config.hidden_size, config.hidden_size, false})
                        .build(ctx, context, {weights.self_attention.out_weight, std::nullopt});
    attn_out = cast_planner_activation(ctx, attn_out, activation_type);
    auto x = modules::AddModule{}.build(ctx, input, attn_out);
    x = cast_planner_activation(ctx, x, activation_type);

    auto ff_in = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, weights.post_norm.bias.has_value()})
                     .build(ctx, x, weights.post_norm);
    auto gate = modules::LinearModule({config.hidden_size, config.intermediate_size, false})
                    .build(ctx, ff_in, {weights.mlp.gate_proj.weight, std::nullopt});
    gate = modules::SiluModule{}.build(ctx, gate);
    gate = cast_planner_activation(ctx, gate, activation_type);
    auto up = modules::LinearModule({config.hidden_size, config.intermediate_size, false})
                  .build(ctx, ff_in, {weights.mlp.up_proj.weight, std::nullopt});
    up = cast_planner_activation(ctx, up, activation_type);
    auto gated = modules::MulModule{}.build(ctx, gate, up);
    gated = cast_planner_activation(ctx, gated, activation_type);
    auto ff = modules::LinearModule({config.intermediate_size, config.hidden_size, false})
                  .build(ctx, gated, {weights.mlp.down_proj.weight, std::nullopt});
    ff = cast_planner_activation(ctx, ff, activation_type);
    auto output = modules::AddModule{}.build(ctx, x, ff);
    output = cast_planner_activation(ctx, output, activation_type);
    return {output, k, v};
}

core::TensorValue planner_set_compact_kv_row(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    const core::TensorValue & row,
    const core::TensorValue & cache_slot) {
    core::validate_rank_between(cache, 4, 4, "cache");
    core::validate_shape(
        row,
        core::TensorShape::from_dims({cache.shape.dims[0], 1, cache.shape.dims[2], cache.shape.dims[3]}),
        "row");
    const int64_t batch = cache.shape.dims[0];
    core::validate_shape(cache_slot, core::TensorShape::from_dims({batch}), "cache_slot");
    if (batch == 1) {
        const modules::FastKVSetRowsModule set_rows;
        return set_rows.build(ctx, cache, row, cache_slot);
    }

    const int64_t steps = cache.shape.dims[1];
    const int64_t row_elems = cache.shape.dims[2] * cache.shape.dims[3];
    auto flat_cache = core::reshape_tensor(
        ctx,
        cache,
        core::TensorShape::from_dims({batch * steps, row_elems}));
    auto flat_row = core::reshape_tensor(
        ctx,
        ensure_planner_contiguous(ctx, row),
        core::TensorShape::from_dims({batch, row_elems}));
    ggml_tensor * updated = ggml_set_rows(ctx.ggml, flat_cache.tensor, flat_row.tensor, cache_slot.tensor);
    auto flat_updated = core::wrap_tensor(updated, flat_cache.shape, cache.type);
    return core::reshape_tensor(ctx, flat_updated, cache.shape);
}

modules::QwenDecoderLayerOutputs planner_decoder_layer_with_static_cache_tail_batched(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const QwenDecoderLayerWeights & weights,
    const AceStepPlannerConfig & config,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const core::TensorValue & attention_mask,
    ggml_type activation_type) {
    const int64_t dim = planner_attention_head_dim(config);
    const int64_t scratch_slot = cache_key.shape.dims[1] - 1;

    auto x_norm = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, weights.input_norm.bias.has_value()})
                      .build(ctx, input, weights.input_norm);
    auto q = modules::LinearModule({config.hidden_size, config.num_attention_heads * dim, false})
                 .build(ctx, x_norm, {weights.self_attention.q_weight, std::nullopt});
    auto k = modules::LinearModule({config.hidden_size, config.num_key_value_heads * dim, false})
                 .build(ctx, x_norm, {weights.self_attention.k_weight, std::nullopt});
    auto v = modules::LinearModule({config.hidden_size, config.num_key_value_heads * dim, false})
                 .build(ctx, x_norm, {weights.self_attention.v_weight, std::nullopt});
    q = cast_planner_activation(ctx, q, activation_type);
    k = cast_planner_activation(ctx, k, activation_type);
    v = cast_planner_activation(ctx, v, activation_type);

    q = modules::RMSNormModule({dim, config.rms_norm_eps, true, weights.q_norm.bias.has_value()})
            .build(ctx, reshape_planner_heads(ctx, q, config.num_attention_heads, dim), weights.q_norm);
    k = modules::RMSNormModule({dim, config.rms_norm_eps, true, weights.k_norm.bias.has_value()})
            .build(ctx, reshape_planner_heads(ctx, k, config.num_key_value_heads, dim), weights.k_norm);
    v = reshape_planner_heads(ctx, v, config.num_key_value_heads, dim);
    v = cast_planner_activation(ctx, v, activation_type);

    q = apply_planner_rope_batched(ctx, q, positions, dim, config.rope_theta);
    k = apply_planner_rope_batched(ctx, k, positions, dim, config.rope_theta);
    q = cast_planner_activation(ctx, q, activation_type);
    k = cast_planner_activation(ctx, k, activation_type);

    auto key_tail = planner_cache_view(ctx, cache_key, 0, cache_key.shape.dims[0], scratch_slot, 1, config.num_key_value_heads, dim);
    auto value_tail =
        planner_cache_view(ctx, cache_value, 0, cache_value.shape.dims[0], scratch_slot, 1, config.num_key_value_heads, dim);
    ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, k.tensor, key_tail.tensor));
    ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, v.tensor, value_tail.tensor));

    std::array<int, core::kMaxTensorRank> head_axes = {0, 2, 1, 3};
    auto q_heads = modules::TransposeModule({head_axes, q.shape.rank}).build(ctx, q);
    q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    auto k_heads = modules::TransposeModule({head_axes, cache_key.shape.rank}).build(ctx, cache_key);
    auto v_heads = modules::TransposeModule({head_axes, cache_value.shape.rank}).build(ctx, cache_value);
    k_heads = core::wrap_tensor(ggml_cont(ctx.ggml, k_heads.tensor), k_heads.shape, k_heads.type);
    v_heads = core::wrap_tensor(ggml_cont(ctx.ggml, v_heads.tensor), v_heads.shape, v_heads.type);
    auto context = sdpa_from_planner_grouped_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.num_attention_heads * dim}));
    context = cast_planner_activation(ctx, context, activation_type);

    auto attn_out = modules::LinearModule({config.hidden_size, config.hidden_size, false})
                        .build(ctx, context, {weights.self_attention.out_weight, std::nullopt});
    attn_out = cast_planner_activation(ctx, attn_out, activation_type);
    auto x = modules::AddModule{}.build(ctx, input, attn_out);
    x = cast_planner_activation(ctx, x, activation_type);

    auto ff_in = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, weights.post_norm.bias.has_value()})
                     .build(ctx, x, weights.post_norm);
    auto gate = modules::LinearModule({config.hidden_size, config.intermediate_size, false})
                    .build(ctx, ff_in, {weights.mlp.gate_proj.weight, std::nullopt});
    gate = modules::SiluModule{}.build(ctx, gate);
    gate = cast_planner_activation(ctx, gate, activation_type);
    auto up = modules::LinearModule({config.hidden_size, config.intermediate_size, false})
                  .build(ctx, ff_in, {weights.mlp.up_proj.weight, std::nullopt});
    up = cast_planner_activation(ctx, up, activation_type);
    auto gated = modules::MulModule{}.build(ctx, gate, up);
    gated = cast_planner_activation(ctx, gated, activation_type);
    auto ff = modules::LinearModule({config.intermediate_size, config.hidden_size, false})
                  .build(ctx, gated, {weights.mlp.down_proj.weight, std::nullopt});
    ff = cast_planner_activation(ctx, ff, activation_type);
    auto output = modules::AddModule{}.build(ctx, x, ff);
    output = cast_planner_activation(ctx, output, activation_type);
    return {output, k, v};
}

modules::QwenDecoderLayerOutputs planner_decoder_layer_with_compact_cache_batched(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const QwenDecoderLayerWeights & weights,
    const AceStepPlannerConfig & config,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const core::TensorValue & cache_slot,
    const core::TensorValue & attention_mask,
    ggml_type activation_type) {
    const int64_t dim = planner_attention_head_dim(config);

    auto x_norm = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, weights.input_norm.bias.has_value()})
                      .build(ctx, input, weights.input_norm);
    auto q = modules::LinearModule({config.hidden_size, config.num_attention_heads * dim, false})
                 .build(ctx, x_norm, {weights.self_attention.q_weight, std::nullopt});
    auto k = modules::LinearModule({config.hidden_size, config.num_key_value_heads * dim, false})
                 .build(ctx, x_norm, {weights.self_attention.k_weight, std::nullopt});
    auto v = modules::LinearModule({config.hidden_size, config.num_key_value_heads * dim, false})
                 .build(ctx, x_norm, {weights.self_attention.v_weight, std::nullopt});
    q = cast_planner_activation(ctx, q, activation_type);
    k = cast_planner_activation(ctx, k, activation_type);
    v = cast_planner_activation(ctx, v, activation_type);

    q = modules::RMSNormModule({dim, config.rms_norm_eps, true, weights.q_norm.bias.has_value()})
            .build(ctx, reshape_planner_heads(ctx, q, config.num_attention_heads, dim), weights.q_norm);
    k = modules::RMSNormModule({dim, config.rms_norm_eps, true, weights.k_norm.bias.has_value()})
            .build(ctx, reshape_planner_heads(ctx, k, config.num_key_value_heads, dim), weights.k_norm);
    v = reshape_planner_heads(ctx, v, config.num_key_value_heads, dim);
    v = cast_planner_activation(ctx, v, activation_type);

    q = apply_planner_rope_batched(ctx, q, positions, dim, config.rope_theta);
    k = apply_planner_rope_batched(ctx, k, positions, dim, config.rope_theta);
    q = cast_planner_activation(ctx, q, activation_type);
    k = cast_planner_activation(ctx, k, activation_type);

    auto updated_cache_key = planner_set_compact_kv_row(ctx, cache_key, k, cache_slot);
    auto updated_cache_value = planner_set_compact_kv_row(ctx, cache_value, v, cache_slot);

    std::array<int, core::kMaxTensorRank> head_axes = {0, 2, 1, 3};
    auto q_heads = modules::TransposeModule({head_axes, q.shape.rank}).build(ctx, q);
    q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    auto k_heads = modules::TransposeModule({head_axes, updated_cache_key.shape.rank}).build(ctx, updated_cache_key);
    auto v_heads = modules::TransposeModule({head_axes, updated_cache_value.shape.rank}).build(ctx, updated_cache_value);
    k_heads = core::wrap_tensor(ggml_cont(ctx.ggml, k_heads.tensor), k_heads.shape, k_heads.type);
    v_heads = core::wrap_tensor(ggml_cont(ctx.ggml, v_heads.tensor), v_heads.shape, v_heads.type);
    auto context = sdpa_from_planner_grouped_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.num_attention_heads * dim}));
    context = cast_planner_activation(ctx, context, activation_type);

    auto attn_out = modules::LinearModule({config.hidden_size, config.hidden_size, false})
                        .build(ctx, context, {weights.self_attention.out_weight, std::nullopt});
    attn_out = cast_planner_activation(ctx, attn_out, activation_type);
    auto x = modules::AddModule{}.build(ctx, input, attn_out);
    x = cast_planner_activation(ctx, x, activation_type);

    auto ff_in = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, weights.post_norm.bias.has_value()})
                     .build(ctx, x, weights.post_norm);
    auto gate = modules::LinearModule({config.hidden_size, config.intermediate_size, false})
                    .build(ctx, ff_in, {weights.mlp.gate_proj.weight, std::nullopt});
    gate = modules::SiluModule{}.build(ctx, gate);
    gate = cast_planner_activation(ctx, gate, activation_type);
    auto up = modules::LinearModule({config.hidden_size, config.intermediate_size, false})
                  .build(ctx, ff_in, {weights.mlp.up_proj.weight, std::nullopt});
    up = cast_planner_activation(ctx, up, activation_type);
    auto gated = modules::MulModule{}.build(ctx, gate, up);
    gated = cast_planner_activation(ctx, gated, activation_type);
    auto ff = modules::LinearModule({config.intermediate_size, config.hidden_size, false})
                  .build(ctx, gated, {weights.mlp.down_proj.weight, std::nullopt});
    ff = cast_planner_activation(ctx, ff, activation_type);
    auto output = modules::AddModule{}.build(ctx, x, ff);
    output = cast_planner_activation(ctx, output, activation_type);
    return {output, k, v};
}

std::string format_cot_text(const AceStepPlan & plan) {
    std::ostringstream out;
    out << "<think>\n";
    bool has_item = false;
    if (plan.metadata.bpm.has_value()) {
        out << "bpm: " << *plan.metadata.bpm << "\n";
        has_item = true;
    }
    const std::string & cot_caption = !plan.cot_caption.empty() ? plan.cot_caption : plan.caption;
    if (!cot_caption.empty()) {
        out << "caption: " << cot_caption << "\n";
        has_item = true;
    }
    if (plan.metadata.duration.has_value()) {
        out << "duration: " << *plan.metadata.duration << "\n";
        has_item = true;
    }
    if (plan.metadata.keyscale.has_value() && !plan.metadata.keyscale->empty()) {
        out << "keyscale: " << *plan.metadata.keyscale << "\n";
        has_item = true;
    }
    if (plan.metadata.language.has_value() && !plan.metadata.language->empty()) {
        out << "language: " << *plan.metadata.language << "\n";
        has_item = true;
    }
    if (plan.metadata.timesignature.has_value() && !plan.metadata.timesignature->empty()) {
        std::string value = *plan.metadata.timesignature;
        if (value.size() >= 2 && value.compare(value.size() - 2, 2, "/4") == 0) {
            value.resize(value.size() - 2);
        }
        out << "timesignature: " << value << "\n";
        has_item = true;
    }
    if (!has_item) {
        out << "\n";
    }
    out << "</think>";
    return out.str();
}

std::string normalized_negative_prompt(const AceStepRequest & request) {
    if (!request.negative_prompt.empty()) {
        return request.negative_prompt;
    }
    return "NO USER INPUT";
}

std::string empty_cot_text() {
    return "<think>\n\n</think>";
}

AceStepMetadata request_metadata(const AceStepRequest & request) {
    AceStepMetadata metadata;
    if (request.bpm.has_value() && *request.bpm > 0) {
        metadata.bpm = request.bpm;
    }
    if (request.keyscale.has_value() && ace_step_metadata_text_is_provided(*request.keyscale)) {
        metadata.keyscale = request.keyscale;
    }
    if (request.timesignature.has_value() && ace_step_metadata_text_is_provided(*request.timesignature)) {
        metadata.timesignature = request.timesignature;
    }
    if (request.generation.duration_seconds > 0.0F) {
        metadata.duration = static_cast<int64_t>(request.generation.duration_seconds);
    }
    return metadata;
}

bool has_all_request_metas(const AceStepRequest & request) {
    return request.bpm.has_value() &&
           *request.bpm > 0 &&
           request.keyscale.has_value() &&
           ace_step_metadata_text_is_provided(*request.keyscale) &&
           request.timesignature.has_value() &&
           ace_step_metadata_text_is_provided(*request.timesignature) &&
           request.generation.duration_seconds > 0.0F;
}

void overlay_request_metadata(AceStepPlan & plan, const AceStepRequest & request) {
    if (request.bpm.has_value() && *request.bpm > 0) {
        plan.metadata.bpm = request.bpm;
    }
    if (request.keyscale.has_value() && ace_step_metadata_text_is_provided(*request.keyscale)) {
        plan.metadata.keyscale = request.keyscale;
    }
    if (request.timesignature.has_value() && ace_step_metadata_text_is_provided(*request.timesignature)) {
        plan.metadata.timesignature = request.timesignature;
    }
    if (request.generation.duration_seconds > 0.0F) {
        plan.metadata.duration = static_cast<int64_t>(request.generation.duration_seconds);
    }
}

class Phase2CodesProcessor {
public:
    Phase2CodesProcessor(
        int32_t eos_token_id,
        int64_t target_codes)
        : eos_token_id_(eos_token_id),
          target_codes_(target_codes) {
        if (eos_token_id_ < 0) {
            throw std::runtime_error("ACE-Step planner phase-2 processor requires eos_token_id");
        }
        if (target_codes_ <= 0) {
            throw std::runtime_error("ACE-Step planner phase-2 processor requires positive target code count");
        }
    }

    void update_state(int32_t token) {
        if (completed_) {
            return;
        }
        ++codes_count_;
        if (token == eos_token_id_) {
            completed_ = true;
        }
    }

    bool completed() const noexcept {
        return completed_;
    }

private:
    int32_t eos_token_id_ = -1;
    int64_t target_codes_ = 0;
    int64_t codes_count_ = 0;
    bool completed_ = false;
};

class Phase2CandidateSampler {
public:
    Phase2CandidateSampler(
        const std::vector<int32_t> & candidate_token_ids,
        int32_t eos_token_id)
        : candidate_token_ids_(candidate_token_ids),
          eos_token_id_(eos_token_id) {
        if (candidate_token_ids_.empty()) {
            throw std::runtime_error("ACE-Step planner phase-2 sampler requires candidate tokens");
        }
        int32_t max_token_id = -1;
        for (const int32_t token : candidate_token_ids_) {
            if (token < 0) {
                throw std::runtime_error("ACE-Step planner phase-2 sampler candidate token id is negative");
            }
            max_token_id = std::max(max_token_id, token);
        }
        values_.resize(candidate_token_ids_.size(), -std::numeric_limits<float>::infinity());
        weights_.resize(candidate_token_ids_.size(), 0.0);
        top_p_kept_.resize(candidate_token_ids_.size(), 0);
        seen_candidate_stamp_.resize(candidate_token_ids_.size(), 0);
        token_to_candidate_index_.assign(static_cast<size_t>(max_token_id) + 1, -1);
        for (size_t i = 0; i < candidate_token_ids_.size(); ++i) {
            token_to_candidate_index_[static_cast<size_t>(candidate_token_ids_[i])] = static_cast<int32_t>(i);
            if (candidate_token_ids_[i] == eos_token_id_) {
                eos_index_ = i;
            }
        }
        if (!eos_index_.has_value()) {
            throw std::runtime_error("ACE-Step planner phase-2 sampler candidate set is missing eos");
        }
    }

    void write_branch_logits(const std::vector<float> & logits) {
        top_p_active_indices_.clear();
        for (size_t i = 0; i < candidate_token_ids_.size(); ++i) {
            values_[i] = logits[static_cast<size_t>(candidate_token_ids_[i])];
        }
    }

    void write_cfg_logits(
        const std::vector<float> & conditional,
        const std::vector<float> & unconditional,
        float scale) {
        if (conditional.size() != unconditional.size()) {
            throw std::runtime_error("ACE-Step planner CFG logits size mismatch");
        }
        top_p_active_indices_.clear();
        for (size_t i = 0; i < candidate_token_ids_.size(); ++i) {
            const size_t token = static_cast<size_t>(candidate_token_ids_[i]);
            const float cond = conditional[token];
            const float uncond = unconditional[token];
            const float value = uncond + scale * (cond - uncond);
            values_[i] = std::isnan(value) ? -std::numeric_limits<float>::infinity() : value;
        }
    }

    void apply_duration_constraint(int64_t codes_count, int64_t target_codes) {
        const size_t eos_index = *eos_index_;
        if (codes_count < target_codes) {
            values_[eos_index] = -std::numeric_limits<float>::infinity();
            return;
        }
        const float eos_score = values_[eos_index];
        std::fill(values_.begin(), values_.end(), -std::numeric_limits<float>::infinity());
        values_[eos_index] = eos_score;
    }

    void apply_repetition_penalty(const std::vector<int32_t> & history, float repetition_penalty) {
        if (repetition_penalty == 1.0F) {
            return;
        }
        ++seen_stamp_;
        if (seen_stamp_ == 0) {
            std::fill(seen_candidate_stamp_.begin(), seen_candidate_stamp_.end(), 0);
            seen_stamp_ = 1;
        }
        for (const int32_t token : history) {
            if (token < 0 || static_cast<size_t>(token) >= token_to_candidate_index_.size()) {
                continue;
            }
            const int32_t candidate_index = token_to_candidate_index_[static_cast<size_t>(token)];
            if (candidate_index < 0) {
                continue;
            }
            const size_t index = static_cast<size_t>(candidate_index);
            if (seen_candidate_stamp_[index] == seen_stamp_) {
                continue;
            }
            seen_candidate_stamp_[index] = seen_stamp_;
            float & value = values_[index];
            if (!std::isfinite(value)) {
                continue;
            }
            value = value < 0.0F ? value * repetition_penalty : value / repetition_penalty;
        }
    }

    void apply_top_k_filter(int64_t top_k) {
        if (top_k <= 0 || top_k >= static_cast<int64_t>(values_.size())) {
            return;
        }
        finite_values_.clear();
        finite_values_.reserve(values_.size());
        for (const float value : values_) {
            if (std::isfinite(value)) {
                finite_values_.push_back(value);
            }
        }
        if (static_cast<int64_t>(finite_values_.size()) <= top_k) {
            return;
        }
        const auto nth = finite_values_.begin() + (top_k - 1);
        std::nth_element(finite_values_.begin(), nth, finite_values_.end(), std::greater<float>());
        const float threshold = *nth;
        int64_t kept = 0;
        for (float & value : values_) {
            if (!std::isfinite(value)) {
                continue;
            }
            if (value > threshold) {
                ++kept;
                continue;
            }
            if (value == threshold && kept < top_k) {
                ++kept;
                continue;
            }
            value = -std::numeric_limits<float>::infinity();
        }
    }

    void apply_top_p_filter(float top_p) {
        top_p_active_indices_.clear();
        if (!(top_p > 0.0F && top_p < 1.0F)) {
            return;
        }
        sorted_.clear();
        sorted_.reserve(values_.size());
        std::fill(top_p_kept_.begin(), top_p_kept_.end(), 0);
        float max_logit = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < values_.size(); ++i) {
            if (std::isfinite(values_[i])) {
                sorted_.push_back({i, values_[i]});
                if (values_[i] > max_logit) {
                    max_logit = values_[i];
                }
            }
        }
        if (sorted_.empty()) {
            return;
        }
        double denom = 0.0;
        for (const IndexedValue & item : sorted_) {
            denom += std::exp(static_cast<double>(item.value - max_logit));
        }
        const auto heap_less = [](const IndexedValue & lhs, const IndexedValue & rhs) {
            return lhs.value < rhs.value;
        };
        std::make_heap(sorted_.begin(), sorted_.end(), heap_less);
        double cumulative = 0.0;
        size_t kept = 0;
        while (!sorted_.empty() && (kept == 0 || cumulative <= static_cast<double>(top_p))) {
            std::pop_heap(sorted_.begin(), sorted_.end(), heap_less);
            const IndexedValue item = sorted_.back();
            sorted_.pop_back();
            top_p_kept_[item.index] = 1;
            cumulative += std::exp(static_cast<double>(item.value - max_logit)) / denom;
            ++kept;
        }
        for (size_t i = 0; i < values_.size(); ++i) {
            if (!std::isfinite(values_[i])) {
                continue;
            }
            if (top_p_kept_[i] != 0) {
                top_p_active_indices_.push_back(i);
            } else {
                values_[i] = -std::numeric_limits<float>::infinity();
            }
        }
    }

    int32_t sample(float temperature, std::mt19937 & rng) {
        if (!(temperature > 0.0F)) {
            int32_t best = -1;
            float best_value = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < values_.size(); ++i) {
                if (std::isfinite(values_[i]) && (best < 0 || values_[i] > best_value)) {
                    best = candidate_token_ids_[i];
                    best_value = values_[i];
                }
            }
            if (best < 0) {
                throw std::runtime_error("ACE-Step planner masked decode found no valid token");
            }
            return best;
        }

        if (!top_p_active_indices_.empty()) {
            float max_logit = -std::numeric_limits<float>::infinity();
            for (const size_t candidate_index : top_p_active_indices_) {
                max_logit = std::max(max_logit, values_[candidate_index]);
            }
            active_weights_.resize(top_p_active_indices_.size());
            for (size_t i = 0; i < top_p_active_indices_.size(); ++i) {
                const size_t candidate_index = top_p_active_indices_[i];
                active_weights_[i] = std::exp(static_cast<double>((values_[candidate_index] - max_logit) / temperature));
            }
            std::discrete_distribution<int32_t> dist(active_weights_.begin(), active_weights_.end());
            return candidate_token_ids_[top_p_active_indices_[static_cast<size_t>(dist(rng))]];
        }
        float max_logit = -std::numeric_limits<float>::infinity();
        for (const float value : values_) {
            if (std::isfinite(value) && value > max_logit) {
                max_logit = value;
            }
        }
        if (!std::isfinite(max_logit)) {
            throw std::runtime_error("ACE-Step planner phase-2 sampling found no finite logits");
        }
        for (size_t i = 0; i < values_.size(); ++i) {
            weights_[i] = std::isfinite(values_[i])
                ? std::exp(static_cast<double>((values_[i] - max_logit) / temperature))
                : 0.0;
        }
        std::discrete_distribution<int32_t> dist(weights_.begin(), weights_.end());
        return candidate_token_ids_[static_cast<size_t>(dist(rng))];
    }

private:
    struct IndexedValue {
        size_t index = 0;
        float value = -std::numeric_limits<float>::infinity();
    };

    const std::vector<int32_t> & candidate_token_ids_;
    int32_t eos_token_id_ = -1;
    std::optional<size_t> eos_index_;
    std::vector<float> values_;
    std::vector<float> finite_values_;
    std::vector<IndexedValue> sorted_;
    std::vector<uint8_t> top_p_kept_;
    std::vector<size_t> top_p_active_indices_;
    std::vector<double> active_weights_;
    std::vector<double> weights_;
    std::vector<int32_t> token_to_candidate_index_;
    std::vector<uint32_t> seen_candidate_stamp_;
    uint32_t seen_stamp_ = 1;
};

template <typename Allowed>
int32_t masked_argmax_index(const std::vector<float> & values, Allowed && allowed) {
    int32_t best = -1;
    float best_value = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < values.size(); ++i) {
        const int32_t token = static_cast<int32_t>(i);
        if (!allowed(token)) {
            continue;
        }
        if (best < 0 || values[i] > best_value) {
            best = token;
            best_value = values[i];
        }
    }
    if (best < 0) {
        throw std::runtime_error("ACE-Step planner masked decode found no valid token");
    }
    return best;
}

constexpr int64_t kPlannerBpmMin = 30;
constexpr int64_t kPlannerBpmMax = 300;
constexpr int64_t kPlannerDurationMin = 10;
constexpr int64_t kPlannerDurationMax = 600;
constexpr int64_t kPlannerCodesPhaseExtraBudget = 10;
constexpr std::array<int, 4> kPlannerTimeSignatures = {2, 3, 4, 6};
constexpr std::array<const char *, 51> kPlannerValidLanguages = {
    "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en",
    "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id",
    "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no",
    "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw",
    "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh",
    "unknown",
};

template <typename T>
void append_unique(std::vector<T> & values, const T & value) {
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

bool starts_with_tokens(const std::vector<int32_t> & tokens, const std::vector<int32_t> & prefix) {
    return tokens.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), tokens.begin());
}

std::map<std::vector<int32_t>, std::vector<int32_t>> build_field_tail_prefix_map(
    const AceStepTextTokenizer & tokenizer,
    const std::string & prefix_for_matching,
    const std::string & prefix_for_tokenization,
    const std::vector<std::string> & values) {
    const std::vector<int32_t> prefix_tokens = tokenizer.encode(prefix_for_matching);
    std::map<std::vector<int32_t>, std::vector<int32_t>> prefix_map;
    for (const std::string & value : values) {
        const std::vector<int32_t> full_tokens = tokenizer.encode(prefix_for_tokenization + value + "\n");
        if (!starts_with_tokens(full_tokens, prefix_tokens)) {
            throw std::runtime_error("ACE-Step planner failed to build constrained prefix map for " + prefix_for_matching);
        }
        std::vector<int32_t> tail(full_tokens.begin() + static_cast<std::ptrdiff_t>(prefix_tokens.size()), full_tokens.end());
        std::vector<int32_t> tail_prefix;
        for (const int32_t token : tail) {
            append_unique(prefix_map[tail_prefix], token);
            tail_prefix.push_back(token);
        }
    }
    return prefix_map;
}

std::vector<std::string> planner_numeric_values(int64_t min_value, int64_t max_value) {
    std::vector<std::string> values;
    values.reserve(static_cast<size_t>(std::max<int64_t>(0, max_value - min_value + 1)));
    for (int64_t value = min_value; value <= max_value; ++value) {
        values.push_back(std::to_string(value));
    }
    return values;
}

std::vector<std::string> planner_timesignature_values() {
    std::vector<std::string> values;
    values.reserve(kPlannerTimeSignatures.size());
    for (const int value : kPlannerTimeSignatures) {
        values.push_back(std::to_string(value));
    }
    return values;
}

std::vector<std::string> planner_keyscale_values() {
    static constexpr std::array<const char *, 7> kNotes = {"A", "B", "C", "D", "E", "F", "G"};
    static constexpr std::array<const char *, 5> kAccidentals = {"", "#", "b", "\xE2\x99\xAF", "\xE2\x99\xAD"};
    static constexpr std::array<const char *, 2> kModes = {"major", "minor"};
    std::vector<std::string> values;
    values.reserve(kNotes.size() * kAccidentals.size() * kModes.size());
    for (const char * note : kNotes) {
        for (const char * accidental : kAccidentals) {
            for (const char * mode : kModes) {
                values.push_back(std::string(note) + accidental + " " + mode);
            }
        }
    }
    return values;
}

int64_t planner_codes_phase_max_new_tokens(
    const AceStepRequest & request,
    int64_t fallback_max_new_tokens) {
    if (request.generation.duration_seconds > 0.0F) {
        const float clamped_duration = std::clamp(
            request.generation.duration_seconds,
            static_cast<float>(kPlannerDurationMin),
            static_cast<float>(kPlannerDurationMax));
        return static_cast<int64_t>(clamped_duration * 5.0F) + kPlannerCodesPhaseExtraBudget;
    }
    return fallback_max_new_tokens;
}

std::vector<std::string> planner_language_values() {
    std::vector<std::string> values;
    values.reserve(kPlannerValidLanguages.size());
    for (const char * language : kPlannerValidLanguages) {
        values.emplace_back(language);
    }
    return values;
}

Phase1ConstraintTables build_phase1_constraint_tables(
    const AceStepTextTokenizer & tokenizer,
    const AceStepPlannerConfig & config) {
    Phase1ConstraintTables tables;
    const auto newline_tokens = tokenizer.encode("\n");
    if (!newline_tokens.empty()) {
        tables.newline_token = newline_tokens.back();
    }
    tables.backtick_token = tokenizer.find_token_id("`");
    if (config.pad_token_id >= 0) {
        tables.forbidden_tokens.push_back(config.pad_token_id);
    }
    if (config.eos_token_id >= 0) {
        tables.forbidden_tokens.push_back(config.eos_token_id);
    }
    if (config.eos_token_id < 0) {
        throw std::runtime_error("ACE-Step planner requires eos_token_id for constrained decode");
    }
    tables.bpm_prefix_map = build_field_tail_prefix_map(
        tokenizer,
        "bpm:",
        "bpm: ",
        planner_numeric_values(kPlannerBpmMin, kPlannerBpmMax));
    tables.duration_prefix_map = build_field_tail_prefix_map(
        tokenizer,
        "duration:",
        "duration: ",
        planner_numeric_values(kPlannerDurationMin, kPlannerDurationMax));
    tables.keyscale_prefix_map = build_field_tail_prefix_map(
        tokenizer,
        "keyscale:",
        "keyscale: ",
        planner_keyscale_values());
    tables.language_prefix_map = build_field_tail_prefix_map(
        tokenizer,
        "language:",
        "language: ",
        planner_language_values());
    tables.timesig_prefix_map = build_field_tail_prefix_map(
        tokenizer,
        "timesignature:",
        "timesignature: ",
        planner_timesignature_values());
    return tables;
}

enum class Phase1State {
    ThinkTag,
    NewlineAfterThink,
    BpmName,
    BpmValue,
    CaptionName,
    CaptionValue,
    DurationName,
    DurationValue,
    KeyscaleName,
    KeyscaleValue,
    LanguageName,
    LanguageValue,
    TimesigName,
    TimesigValue,
    EndOfTurn,
    Completed,
};

class Phase1ConstrainedDecoder {
public:
    Phase1ConstrainedDecoder(
        const AceStepTextTokenizer & tokenizer,
        const AceStepPlannerConfig & config,
        const AceStepRequest & request,
        const std::vector<uint8_t> & is_audio_code_token,
        const Phase1ConstraintTables & tables)
        : tokenizer_(tokenizer),
          config_(config),
          is_audio_code_token_(is_audio_code_token),
          tables_(tables) {
        const int64_t request_duration = request.generation.duration_seconds > 0.0F
            ? static_cast<int64_t>(request.generation.duration_seconds)
            : 0;
        if (request_duration > 0) {
            duration_user_tokens_ = tokenizer_.encode(" " + std::to_string(request_duration) + "\n");
        }
    }

    bool completed() const noexcept {
        return state_ == Phase1State::Completed;
    }

    int32_t select_next_token(const std::vector<float> & logits) {
        if (state_ == Phase1State::Completed) {
            throw std::runtime_error("ACE-Step planner phase-1 decoder is already complete");
        }
        if (caption_ending_) {
            return raw_argmax(logits);
        }
        while (true) {
            switch (state_) {
            case Phase1State::ThinkTag:
            case Phase1State::NewlineAfterThink:
            case Phase1State::BpmName:
            case Phase1State::CaptionName:
            case Phase1State::DurationName:
            case Phase1State::KeyscaleName:
            case Phase1State::LanguageName:
            case Phase1State::TimesigName:
                return fixed_state_token(logits);
            case Phase1State::EndOfTurn:
                return static_cast<int32_t>(config_.eos_token_id);
            case Phase1State::BpmValue:
                return constrained_value_token(logits, tables_.bpm_prefix_map);
            case Phase1State::DurationValue:
                if (!duration_user_tokens_.empty()) {
                    if (duration_user_position_ >= duration_user_tokens_.size()) {
                        throw std::runtime_error("ACE-Step planner duration token queue is exhausted");
                    }
                    return duration_user_tokens_[duration_user_position_];
                }
                return constrained_value_token(logits, tables_.duration_prefix_map);
            case Phase1State::KeyscaleValue:
                return constrained_value_token(logits, tables_.keyscale_prefix_map);
            case Phase1State::LanguageValue:
                return constrained_value_token(logits, tables_.language_prefix_map);
            case Phase1State::TimesigValue:
                return constrained_value_token(logits, tables_.timesig_prefix_map);
            case Phase1State::CaptionValue:
                if (caption_after_newline_) {
                    const std::string token_text = tokenizer_.decode({raw_argmax(logits)}, false);
                    if (!token_text.empty() && token_text.front() != ' ' && token_text.front() != '\t') {
                        caption_after_newline_ = false;
                        caption_ending_ = true;
                        pending_field_name_.clear();
                        return raw_argmax(logits);
                    }
                    caption_after_newline_ = false;
                }
                if (caption_token_count_ >= 512 && tables_.newline_token.has_value()) {
                    return *tables_.newline_token;
                }
                return masked_argmax_index(logits, [&](int32_t candidate) {
                    if (candidate < 0 || candidate >= static_cast<int32_t>(logits.size())) {
                        return false;
                    }
                    if (std::find(tables_.forbidden_tokens.begin(), tables_.forbidden_tokens.end(), candidate) !=
                        tables_.forbidden_tokens.end()) {
                        return false;
                    }
                    if (tables_.backtick_token.has_value() && candidate == *tables_.backtick_token) {
                        return false;
                    }
                    return candidate >= static_cast<int32_t>(is_audio_code_token_.size())
                        || is_audio_code_token_[static_cast<size_t>(candidate)] == 0;
                });
            case Phase1State::Completed:
                break;
            }
        }
        throw std::runtime_error("ACE-Step planner phase-1 decoder entered an invalid state");
    }

    void observe_token(int32_t token) {
        switch (state_) {
        case Phase1State::ThinkTag:
        case Phase1State::NewlineAfterThink:
        case Phase1State::BpmName:
        case Phase1State::CaptionName:
        case Phase1State::DurationName:
        case Phase1State::KeyscaleName:
        case Phase1State::LanguageName:
        case Phase1State::TimesigName:
            consume_fixed_token(token);
            return;
        case Phase1State::EndOfTurn:
            if (token != config_.eos_token_id) {
                throw std::runtime_error("ACE-Step planner expected eos_token_id at end of turn");
            }
            enter_state(Phase1State::Completed);
            return;
        case Phase1State::BpmValue:
            consume_value_token(token, Phase1State::CaptionName);
            return;
        case Phase1State::DurationValue:
            if (!duration_user_tokens_.empty()) {
                if (duration_user_position_ >= duration_user_tokens_.size() ||
                    duration_user_tokens_[duration_user_position_] != token) {
                    throw std::runtime_error("ACE-Step planner duration token queue desynchronized");
                }
                ++duration_user_position_;
                if (duration_user_position_ >= duration_user_tokens_.size()) {
                    duration_user_tokens_.clear();
                    duration_user_position_ = 0;
                    enter_state(Phase1State::KeyscaleName);
                }
                return;
            }
            consume_value_token(token, Phase1State::KeyscaleName);
            return;
        case Phase1State::KeyscaleValue:
            consume_value_token(token, Phase1State::LanguageName);
            return;
        case Phase1State::LanguageValue:
            consume_value_token(token, Phase1State::TimesigName);
            return;
        case Phase1State::TimesigValue:
            consume_value_token(token, Phase1State::EndOfTurn);
            return;
        case Phase1State::CaptionValue: {
            ++caption_token_count_;
            const std::string token_text = tokenizer_.decode({token}, false);
            caption_after_newline_ = token_text.find('\n') != std::string::npos;
            if (caption_ending_) {
                pending_field_name_ += token_text;
                if (token_text.find(':') != std::string::npos || trim_ascii(token_text) == ":") {
                    const std::string field_name = normalize_field_name(pending_field_name_);
                    caption_ending_ = false;
                    pending_field_name_.clear();
                    if (field_name == "duration") {
                        enter_state(Phase1State::DurationValue);
                    } else if (field_name == "keyscale") {
                        enter_state(Phase1State::KeyscaleValue);
                    } else if (field_name == "language") {
                        enter_state(Phase1State::LanguageValue);
                    } else if (field_name == "timesignature") {
                        enter_state(Phase1State::TimesigValue);
                    } else {
                        enter_state(Phase1State::DurationName);
                    }
                }
            }
            return;
        }
        case Phase1State::Completed:
            return;
        }
    }

private:
    static std::string trim_ascii(std::string_view text) {
        size_t begin = 0;
        while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
            ++begin;
        }
        size_t end = text.size();
        while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
            --end;
        }
        return std::string(text.substr(begin, end - begin));
    }

    static std::string normalize_field_name(std::string value) {
        value = trim_ascii(value);
        while (!value.empty() && value.back() == ':') {
            value.pop_back();
        }
        value = trim_ascii(value);
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return value;
    }

    static std::string normalize_fixed_fragment(std::string value) {
        const auto it = std::find_if(value.begin(), value.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        });
        value.erase(value.begin(), it);
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return value;
    }

    static bool starts_with(const std::string & value, const std::string & prefix) {
        return value.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), value.begin());
    }

    void enter_state(Phase1State next) {
        state_ = next;
        fixed_position_ = 0;
        current_value_tokens_.clear();
        if (state_ != Phase1State::CaptionValue) {
            caption_after_newline_ = false;
            caption_ending_ = false;
            pending_field_name_.clear();
        }
    }

    int32_t raw_argmax(const std::vector<float> & logits) const {
        int32_t best = -1;
        float best_value = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < logits.size(); ++i) {
            if (best < 0 || logits[i] > best_value) {
                best = static_cast<int32_t>(i);
                best_value = logits[i];
            }
        }
        if (best < 0) {
            throw std::runtime_error("ACE-Step planner masked decode found no valid token");
        }
        return best;
    }

    std::string fixed_string() const {
        switch (state_) {
        case Phase1State::ThinkTag:
            return "<think>";
        case Phase1State::NewlineAfterThink:
            return "\n";
        case Phase1State::BpmName:
            return "bpm:";
        case Phase1State::CaptionName:
            return "caption:";
        case Phase1State::DurationName:
            return "duration:";
        case Phase1State::KeyscaleName:
            return "keyscale:";
        case Phase1State::LanguageName:
            return "language:";
        case Phase1State::TimesigName:
            return "timesignature:";
        case Phase1State::EndOfTurn:
        case Phase1State::Completed:
        case Phase1State::BpmValue:
        case Phase1State::CaptionValue:
        case Phase1State::DurationValue:
        case Phase1State::KeyscaleValue:
        case Phase1State::LanguageValue:
        case Phase1State::TimesigValue:
            break;
        }
        throw std::runtime_error("ACE-Step planner fixed string requested for non-fixed state");
    }

    std::vector<int32_t> allowed_tokens_for_fixed_string() const {
        const std::string text = fixed_string();
        if (fixed_position_ >= text.size()) {
            return {};
        }
        const std::string remaining = text.substr(fixed_position_);
        for (size_t end = remaining.size(); end > 0; --end) {
            const std::vector<int32_t> tokens = tokenizer_.encode(remaining.substr(0, end));
            if (tokens.size() == 1) {
                return {tokens.front()};
            }
        }
        std::map<int32_t, size_t> best_prefix;
        const size_t max_end = std::min<size_t>(remaining.size(), 20);
        for (size_t end = 1; end <= max_end; ++end) {
            const std::string prefix = remaining.substr(0, end);
            const std::vector<int32_t> tokens = tokenizer_.encode(prefix);
            if (tokens.empty()) {
                continue;
            }
            const int32_t first_token = tokens.front();
            const std::string decoded = tokenizer_.decode({first_token}, false);
            const std::string normalized_prefix = normalize_fixed_fragment(prefix);
            const std::string normalized_decoded = normalize_fixed_fragment(decoded);
            if (starts_with(normalized_decoded, normalized_prefix) || starts_with(normalized_prefix, normalized_decoded)) {
                auto it = best_prefix.find(first_token);
                if (it == best_prefix.end() || end > it->second) {
                    best_prefix[first_token] = end;
                }
            }
        }
        std::vector<std::pair<int32_t, size_t>> sorted(best_prefix.begin(), best_prefix.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto & lhs, const auto & rhs) {
            return lhs.second > rhs.second;
        });
        std::vector<int32_t> result;
        result.reserve(sorted.size());
        for (const auto & entry : sorted) {
            result.push_back(entry.first);
        }
        return result;
    }

    int32_t fixed_state_token(const std::vector<float> & logits) const {
        const std::vector<int32_t> allowed = allowed_tokens_for_fixed_string();
        if (allowed.empty()) {
            throw std::runtime_error("ACE-Step planner fixed-string state has no continuation tokens");
        }
        return best_allowed_token(logits, allowed);
    }

    int32_t constrained_value_token(
        const std::vector<float> & logits,
        const std::map<std::vector<int32_t>, std::vector<int32_t>> & prefix_map) const {
        const auto it = prefix_map.find(current_value_tokens_);
        if (it == prefix_map.end() || it->second.empty()) {
            throw std::runtime_error("ACE-Step planner constrained decoder found no continuation tokens");
        }
        return best_allowed_token(logits, it->second);
    }

    static int32_t best_allowed_token(const std::vector<float> & logits, const std::vector<int32_t> & allowed) {
        int32_t best = -1;
        float best_value = -std::numeric_limits<float>::infinity();
        for (const int32_t token : allowed) {
            if (token < 0 || token >= static_cast<int32_t>(logits.size())) {
                continue;
            }
            const float value = logits[static_cast<size_t>(token)];
            if (best < 0 || value > best_value) {
                best = token;
                best_value = value;
            }
        }
        if (best < 0) {
            throw std::runtime_error("ACE-Step planner masked decode found no valid token");
        }
        return best;
    }

    void consume_fixed_token(int32_t token) {
        fixed_position_ += tokenizer_.decode({token}, false).size();
        if (fixed_position_ < fixed_string().size()) {
            return;
        }
        switch (state_) {
        case Phase1State::ThinkTag:
            enter_state(Phase1State::NewlineAfterThink);
            break;
        case Phase1State::NewlineAfterThink:
            enter_state(Phase1State::BpmName);
            break;
        case Phase1State::BpmName:
            enter_state(Phase1State::BpmValue);
            break;
        case Phase1State::CaptionName:
            enter_state(Phase1State::CaptionValue);
            caption_token_count_ = 0;
            break;
        case Phase1State::DurationName:
            enter_state(Phase1State::DurationValue);
            break;
        case Phase1State::KeyscaleName:
            enter_state(Phase1State::KeyscaleValue);
            break;
        case Phase1State::LanguageName:
            enter_state(Phase1State::LanguageValue);
            break;
        case Phase1State::TimesigName:
            enter_state(Phase1State::TimesigValue);
            break;
        case Phase1State::EndOfTurn:
            enter_state(Phase1State::Completed);
            break;
        case Phase1State::Completed:
        case Phase1State::BpmValue:
        case Phase1State::CaptionValue:
        case Phase1State::DurationValue:
        case Phase1State::KeyscaleValue:
        case Phase1State::LanguageValue:
        case Phase1State::TimesigValue:
            break;
        }
    }

    void consume_value_token(int32_t token, Phase1State next_state) {
        current_value_tokens_.push_back(token);
        if (tables_.newline_token.has_value() && token == *tables_.newline_token) {
            enter_state(next_state);
        }
    }

    const AceStepTextTokenizer & tokenizer_;
    const AceStepPlannerConfig & config_;
    const std::vector<uint8_t> & is_audio_code_token_;
    const Phase1ConstraintTables & tables_;
    std::vector<int32_t> duration_user_tokens_;
    size_t duration_user_position_ = 0;
    Phase1State state_ = Phase1State::ThinkTag;
    size_t fixed_position_ = 0;
    std::vector<int32_t> current_value_tokens_;
    bool caption_after_newline_ = false;
    bool caption_ending_ = false;
    int64_t caption_token_count_ = 0;
    std::string pending_field_name_;
};

int64_t target_code_count(const AceStepRequest & request, const AceStepPlan & plan) {
    if (request.generation.duration_seconds > 0.0F) {
        return static_cast<int64_t>(request.generation.duration_seconds) * 5;
    }
    if (plan.metadata.duration.has_value() && *plan.metadata.duration > 0) {
        return *plan.metadata.duration * 5;
    }
    throw std::runtime_error("ACE-Step planner requires a positive duration to determine target audio-code count");
}

bool is_eos(const AceStepPlannerConfig & config, int32_t token) {
    return token == config.eos_token_id;
}

std::string require_lm_head_name(const assets::TensorSource & source) {
    return source.require_tensor_name({"lm_head.weight", "embed_tokens.weight"});
}

PlannerWeights load_planner_weights(
    const AceStepAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    const auto & config = assets.config.planner;
    const auto & source = *assets.lm_weights;
    PlannerWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "ace_step.planner.weights",
        weight_context_bytes);
    weights.token_embedding = weights.store->load_tensor(
        source,
        "embed_tokens.weight",
        storage_type,
        {config.vocab_size, config.hidden_size});
    weights.layers.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = "layers." + std::to_string(layer);
        QwenDecoderLayerWeights w;
        w.input_norm.weight = weights.store->load_f32_tensor(
            source, prefix + ".input_layernorm.weight", {config.hidden_size});
        w.q_norm.weight = weights.store->load_f32_tensor(
            source, prefix + ".self_attn.q_norm.weight", {config.head_dim});
        w.k_norm.weight = weights.store->load_f32_tensor(
            source, prefix + ".self_attn.k_norm.weight", {config.head_dim});
        w.post_norm.weight = weights.store->load_f32_tensor(
            source, prefix + ".post_attention_layernorm.weight", {config.hidden_size});
        w.self_attention.q_weight = weights.store->load_tensor(
            source,
            prefix + ".self_attn.q_proj.weight",
            storage_type,
            {config.num_attention_heads * config.head_dim, config.hidden_size});
        w.self_attention.k_weight = weights.store->load_tensor(
            source,
            prefix + ".self_attn.k_proj.weight",
            storage_type,
            {config.num_key_value_heads * config.head_dim, config.hidden_size});
        w.self_attention.v_weight = weights.store->load_tensor(
            source,
            prefix + ".self_attn.v_proj.weight",
            storage_type,
            {config.num_key_value_heads * config.head_dim, config.hidden_size});
        w.self_attention.out_weight = weights.store->load_tensor(
            source,
            prefix + ".self_attn.o_proj.weight",
            storage_type,
            {config.hidden_size, config.num_attention_heads * config.head_dim});
        w.mlp.gate_proj = {
            weights.store->load_tensor(
                source,
                prefix + ".mlp.gate_proj.weight",
                storage_type,
                {config.intermediate_size, config.hidden_size}),
            std::nullopt,
        };
        w.mlp.up_proj = {
            weights.store->load_tensor(
                source,
                prefix + ".mlp.up_proj.weight",
                storage_type,
                {config.intermediate_size, config.hidden_size}),
            std::nullopt,
        };
        w.mlp.down_proj = {
            weights.store->load_tensor(
                source,
                prefix + ".mlp.down_proj.weight",
                storage_type,
                {config.hidden_size, config.intermediate_size}),
            std::nullopt,
        };
        weights.layers.layers.push_back(std::move(w));
    }
    weights.norm = weights.store->load_f32_tensor(source, "norm.weight", {config.hidden_size});
    const std::string lm_head_name = require_lm_head_name(source);
    if (lm_head_name == "embed_tokens.weight") {
        weights.lm_head = weights.token_embedding;
    } else {
        weights.lm_head = weights.store->load_tensor(
            source,
            lm_head_name,
            storage_type,
            {config.vocab_size, config.hidden_size});
    }
    weights.store->upload();
    return weights;
}

}  // namespace

class PlannerWeightsRuntime {
public:
    PlannerWeightsRuntime(
        std::shared_ptr<const AceStepAssets> assets,
        core::ExecutionContext & execution,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets_(std::move(assets)),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          weights_(std::make_shared<PlannerWeights>(
              load_planner_weights(*assets_, backend_, backend_type_, weight_context_bytes, storage_type))) {
        if (assets_ == nullptr) {
            throw std::runtime_error("ACE-Step planner weights runtime requires assets");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("ACE-Step planner backend is not initialized");
        }
    }

    const AceStepAssets & assets() const noexcept { return *assets_; }
    const PlannerWeights & weights() const noexcept { return *weights_; }
    ggml_backend_t backend() const noexcept { return backend_; }
    core::BackendType backend_type() const noexcept { return backend_type_; }
    int threads() const noexcept { return threads_; }

private:
    std::shared_ptr<const AceStepAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const PlannerWeights> weights_;
};

struct PrefillOutput {
    std::vector<float> logits;
    runtime::TransformerKVState kv_state;
};

struct CfgLayerState {
    std::vector<float> key;
    std::vector<float> value;
};

struct CfgPrefillOutput {
    std::vector<float> conditional_logits;
    std::vector<float> unconditional_logits;
    int64_t current_end = 0;
    int64_t valid_steps = 0;
    std::vector<CfgLayerState> layers;
};

class PrefillGraph {
public:
    PrefillGraph(
        std::shared_ptr<PlannerWeightsRuntime> runtime,
        int64_t prompt_steps,
        size_t graph_arena_bytes)
        : runtime_(std::move(runtime)),
          prompt_steps_(prompt_steps) {
        if (prompt_steps_ <= 0) {
            throw std::runtime_error("ACE-Step planner prefill requires positive prompt length");
        }
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize ACE-Step planner prefill graph context");
        }
        const auto & config = runtime_->assets().config.planner;
        const auto & weights = runtime_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), "ace_step.planner.prefill", runtime_->backend_type()};
        token_ids_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
        auto token_ids = core::wrap_tensor(token_ids_, core::TensorShape::from_dims({prompt_steps_}), GGML_TYPE_I32);
        auto x = modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                     .build(ctx, token_ids, weights.token_embedding);
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, prompt_steps_, config.hidden_size}));
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({prompt_steps_}), GGML_TYPE_I32);
        attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, prompt_steps_, prompt_steps_, 1, 1);
        auto attention_mask = core::wrap_tensor(
            attention_mask_,
            core::TensorShape::from_dims({1, 1, prompt_steps_, prompt_steps_}),
            GGML_TYPE_F16);
        query_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F32, 1, 1, prompt_steps_, 1);
        auto query_mask = core::wrap_tensor(
            query_mask_,
            core::TensorShape::from_dims({1, prompt_steps_, 1, 1}),
            GGML_TYPE_F32);
        for (size_t layer_index = 0; layer_index < weights.layers.layers.size(); ++layer_index) {
            const auto & layer = weights.layers.layers[layer_index];
            auto out = planner_decoder_layer_batched(
                ctx,
                x,
                positions,
                layer,
                config,
                attention_mask,
                query_mask,
                GGML_TYPE_F32);
            x = out.output;
            keys_.push_back(out.key.tensor);
            values_.push_back(out.value.tensor);
        }
        x = modules::SliceModule({1, prompt_steps_ - 1, 1}).build(ctx, x);
        x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                .build(ctx, x, {weights.norm, std::nullopt});
        auto logits = modules::LinearModule({config.hidden_size, config.vocab_size, false})
                          .build(ctx, x, {weights.lm_head, std::nullopt});
        logits_ = logits.tensor;
        ggml_set_output(logits_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, logits_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate ACE-Step planner prefill graph");
        }
    }

    ~PrefillGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool can_run(const PlannerWeightsRuntime & runtime, int64_t prompt_steps) const {
        return runtime_.get() == &runtime && prompt_steps_ == prompt_steps;
    }

    PrefillOutput run(const std::vector<int32_t> & token_ids, const std::vector<int32_t> & attention_mask) {
        const auto & config = runtime_->assets().config.planner;
        if (static_cast<int64_t>(token_ids.size()) != prompt_steps_ ||
            static_cast<int64_t>(attention_mask.size()) != prompt_steps_) {
            throw std::runtime_error("ACE-Step planner prefill token or attention-mask count mismatch");
        }
        ggml_backend_tensor_set(token_ids_, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
        std::vector<int32_t> position_ids(static_cast<size_t>(prompt_steps_), 0);
        for (int64_t i = 0; i < prompt_steps_; ++i) {
            position_ids[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        ggml_backend_tensor_set(positions_, position_ids.data(), 0, position_ids.size() * sizeof(int32_t));
        const auto attention_mask_values = build_prefill_attention_mask_values(attention_mask);
        ggml_backend_tensor_set(
            attention_mask_,
            attention_mask_values.data(),
            0,
            attention_mask_values.size() * sizeof(ggml_fp16_t));
        std::vector<float> query_mask_values(static_cast<size_t>(prompt_steps_), 0.0F);
        for (int64_t i = 0; i < prompt_steps_; ++i) {
            query_mask_values[static_cast<size_t>(i)] = attention_mask[static_cast<size_t>(i)] != 0 ? 1.0F : 0.0F;
        }
        ggml_backend_tensor_set(query_mask_, query_mask_values.data(), 0, query_mask_values.size() * sizeof(float));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ACE-Step planner prefill graph compute failed");
        }
        PrefillOutput out;
        out.logits.resize(static_cast<size_t>(config.vocab_size));
        ggml_backend_tensor_get(logits_, out.logits.data(), 0, out.logits.size() * sizeof(float));
        out.kv_state.current_end = prompt_steps_;
        out.kv_state.layers.resize(keys_.size());
        const size_t layer_values = static_cast<size_t>(prompt_steps_ * config.num_key_value_heads * config.head_dim);
        for (size_t layer = 0; layer < keys_.size(); ++layer) {
            auto & state = out.kv_state.layers[layer];
            state.valid_steps = prompt_steps_;
            state.key.resize(layer_values);
            state.value.resize(layer_values);
            ggml_backend_tensor_get(keys_[layer], state.key.data(), 0, state.key.size() * sizeof(float));
            ggml_backend_tensor_get(values_[layer], state.value.data(), 0, state.value.size() * sizeof(float));
        }
        return out;
    }

private:
    std::shared_ptr<PlannerWeightsRuntime> runtime_;
    int64_t prompt_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_ids_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * query_mask_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

class DecodeGraph {
public:
    DecodeGraph(
        std::shared_ptr<PlannerWeightsRuntime> runtime,
        int64_t cache_steps,
        size_t graph_arena_bytes)
        : runtime_(std::move(runtime)),
          cache_steps_(cache_steps) {
        if (cache_steps_ <= 0) {
            throw std::runtime_error("ACE-Step planner decode requires positive cache length");
        }
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize ACE-Step planner decode graph context");
        }
        const auto & config = runtime_->assets().config.planner;
        const auto & weights = runtime_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), "ace_step.planner.decode", runtime_->backend_type()};
        token_id_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto token = core::wrap_tensor(token_id_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto x = modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                     .build(ctx, token, weights.token_embedding);
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, 1, config.hidden_size}));
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, cache_steps_ + 1, 1, 1, 1);
        auto attention_mask = core::wrap_tensor(
            attention_mask_,
            core::TensorShape::from_dims({1, 1, 1, cache_steps_ + 1}),
            GGML_TYPE_F16);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);

        std::vector<core::TensorValue> cache_keys;
        std::vector<core::TensorValue> cache_values;
        for (const auto & layer : weights.layers.layers) {
            cache_keys.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_ + 1, config.num_key_value_heads, config.head_dim})));
            cache_values.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_ + 1, config.num_key_value_heads, config.head_dim})));
            auto out = planner_decoder_layer_with_static_cache_tail_batched(
                ctx,
                graph_,
                x,
                positions,
                layer,
                config,
                cache_keys.back(),
                cache_values.back(),
                attention_mask,
                GGML_TYPE_F32);
            x = out.output;
            key_sources_.push_back(ggml_view_1d(ctx_.get(), out.key.tensor, config.num_key_value_heads * config.head_dim, 0));
            value_sources_.push_back(ggml_view_1d(ctx_.get(), out.value.tensor, config.num_key_value_heads * config.head_dim, 0));
        }
        step_cache_ = runtime::TransformerKVCache(
            cache_steps_ + 1,
            config.num_key_value_heads * config.head_dim,
            std::move(cache_keys),
            std::move(cache_values));
        build_transfer_views(config.num_key_value_heads * config.head_dim);
        x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                .build(ctx, x, {weights.norm, std::nullopt});
        auto logits = modules::LinearModule({config.hidden_size, config.vocab_size, false})
                          .build(ctx, x, {weights.lm_head, std::nullopt});
        logits_ = logits.tensor;
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, logits_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate ACE-Step planner decode graph");
        }
        attention_mask_values_.assign(
            static_cast<size_t>(cache_steps_ + 1),
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
    }

    ~DecodeGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool can_run(const PlannerWeightsRuntime & runtime, int64_t required_steps) const {
        return runtime_.get() == &runtime && cache_steps_ >= required_steps;
    }

    void import_state(const runtime::TransformerKVState & state) {
        step_cache_.import_state(state);
    }

    void reset_state() {
        runtime::TransformerKVState state;
        state.layers.resize(key_sources_.size());
        step_cache_.import_state(state);
    }

    runtime::TransformerKVState export_state() const {
        return step_cache_.export_state();
    }

    void run_step_into(
        int32_t token,
        std::vector<float> & logits,
        const std::vector<int32_t> * visible_prefix = nullptr) {
        const auto & config = runtime_->assets().config.planner;
        if (step_cache_.valid_steps() >= cache_steps_) {
            throw std::runtime_error("ACE-Step planner decode cache exhausted");
        }
        ggml_backend_tensor_set(token_id_, &token, 0, sizeof(int32_t));
        const int32_t position = static_cast<int32_t>(step_cache_.current_end());
        if (visible_prefix != nullptr &&
            static_cast<int64_t>(visible_prefix->size()) != step_cache_.current_end()) {
            throw std::runtime_error("ACE-Step planner decode visible-prefix length mismatch");
        }
        ggml_backend_tensor_set(positions_, &position, 0, sizeof(int32_t));
        const auto masked = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        const auto visible = ggml_fp32_to_fp16(0.0F);
        std::fill(attention_mask_values_.begin(), attention_mask_values_.end(), masked);
        if (visible_prefix != nullptr) {
            for (int64_t i = 0; i < step_cache_.valid_steps(); ++i) {
                if ((*visible_prefix)[static_cast<size_t>(i)] != 0) {
                    attention_mask_values_[static_cast<size_t>(i)] = visible;
                }
            }
        } else {
            for (int64_t i = 0; i < step_cache_.valid_steps(); ++i) {
                attention_mask_values_[static_cast<size_t>(i)] = visible;
            }
        }
        attention_mask_values_[static_cast<size_t>(cache_steps_)] = visible;
        ggml_backend_tensor_set(
            attention_mask_,
            attention_mask_values_.data(),
            0,
            attention_mask_values_.size() * sizeof(ggml_fp16_t));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ACE-Step planner decode graph compute failed");
        }
        logits.resize(static_cast<size_t>(config.vocab_size));
        ggml_backend_tensor_get(logits_, logits.data(), 0, logits.size() * sizeof(float));
        const size_t dst_slot = static_cast<size_t>(step_cache_.valid_steps());
        for (size_t layer = 0; layer < key_sources_.size(); ++layer) {
            ggml_backend_tensor_copy(key_sources_[layer], key_destinations_[dst_slot][layer]);
            ggml_backend_tensor_copy(value_sources_[layer], value_destinations_[dst_slot][layer]);
        }
        step_cache_.advance_after_direct_append(1);
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
                key_slot.push_back(ggml_view_1d(
                    ctx_.get(),
                    step_cache_.key_tensor(layer).tensor,
                    step_elems,
                    byte_offset));
                value_slot.push_back(ggml_view_1d(
                    ctx_.get(),
                    step_cache_.value_tensor(layer).tensor,
                    step_elems,
                    byte_offset));
            }
        }
    }

    std::shared_ptr<PlannerWeightsRuntime> runtime_;
    int64_t cache_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_id_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_tensor *> key_sources_;
    std::vector<ggml_tensor *> value_sources_;
    std::vector<std::vector<ggml_tensor *>> key_destinations_;
    std::vector<std::vector<ggml_tensor *>> value_destinations_;
    std::vector<ggml_fp16_t> attention_mask_values_;
    runtime::TransformerKVCache step_cache_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

PrefillOutput run_decode_graph_prompt_prefill(DecodeGraph & graph, const AceStepTokenizedText & prompt) {
    graph.reset_state();
    std::vector<float> logits;
    for (size_t index = 0; index < prompt.input_ids.size(); ++index) {
        std::vector<int32_t> visible_prefix(
            prompt.attention_mask.begin(),
            prompt.attention_mask.begin() + index);
        graph.run_step_into(prompt.input_ids[index], logits, &visible_prefix);
    }
    return {std::move(logits), graph.export_state()};
}

CfgPrefillOutput pack_cfg_prefill_output(
    int64_t prompt_steps,
    PrefillOutput conditional,
    PrefillOutput unconditional) {
    CfgPrefillOutput output;
    output.conditional_logits = std::move(conditional.logits);
    output.unconditional_logits = std::move(unconditional.logits);
    output.current_end = prompt_steps;
    output.valid_steps = prompt_steps;
    if (conditional.kv_state.layers.size() != unconditional.kv_state.layers.size()) {
        throw std::runtime_error("ACE-Step planner CFG branch prefill layer-count mismatch");
    }
    output.layers.resize(conditional.kv_state.layers.size());
    for (size_t layer = 0; layer < output.layers.size(); ++layer) {
        const auto & conditional_layer = conditional.kv_state.layers[layer];
        const auto & unconditional_layer = unconditional.kv_state.layers[layer];
        auto & target = output.layers[layer];
        target.key.reserve(conditional_layer.key.size() + unconditional_layer.key.size());
        target.key.insert(target.key.end(), conditional_layer.key.begin(), conditional_layer.key.end());
        target.key.insert(target.key.end(), unconditional_layer.key.begin(), unconditional_layer.key.end());
        target.value.reserve(conditional_layer.value.size() + unconditional_layer.value.size());
        target.value.insert(target.value.end(), conditional_layer.value.begin(), conditional_layer.value.end());
        target.value.insert(target.value.end(), unconditional_layer.value.begin(), unconditional_layer.value.end());
    }
    return output;
}

class CfgPrefillGraph {
public:
    CfgPrefillGraph(
        std::shared_ptr<PlannerWeightsRuntime> runtime,
        int64_t prompt_steps,
        size_t graph_arena_bytes)
        : runtime_(std::move(runtime)),
          prompt_steps_(prompt_steps),
          activation_type_(planner_phase2_activation_type(runtime_->backend_type())) {
        if (prompt_steps_ <= 0) {
            throw std::runtime_error("ACE-Step planner CFG prefill requires positive prompt length");
        }
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize ACE-Step planner CFG prefill graph context");
        }
        const auto & config = runtime_->assets().config.planner;
        const auto & weights = runtime_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), "ace_step.planner.cfg_prefill", runtime_->backend_type()};
        token_ids_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_I32, prompt_steps_, 2);
        ggml_set_input(token_ids_);
        auto token_ids = core::wrap_tensor(token_ids_, core::TensorShape::from_dims({2, prompt_steps_}), GGML_TYPE_I32);
        auto x = modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                     .build(ctx, token_ids, weights.token_embedding);
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({2, prompt_steps_, config.hidden_size}));
        x = cast_planner_activation(ctx, x, activation_type_);
        positions_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_I32, prompt_steps_, 2);
        ggml_set_input(positions_);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({2, prompt_steps_}), GGML_TYPE_I32);
        attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, prompt_steps_, prompt_steps_, 1, 2);
        ggml_set_input(attention_mask_);
        auto attention_mask = core::wrap_tensor(
            attention_mask_,
            core::TensorShape::from_dims({2, 1, prompt_steps_, prompt_steps_}),
            GGML_TYPE_F16);
        query_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F32, 1, 1, prompt_steps_, 2);
        ggml_set_input(query_mask_);
        auto query_mask = core::wrap_tensor(
            query_mask_,
            core::TensorShape::from_dims({2, prompt_steps_, 1, 1}),
            GGML_TYPE_F32);
        for (size_t layer_index = 0; layer_index < weights.layers.layers.size(); ++layer_index) {
            const auto & layer = weights.layers.layers[layer_index];
            auto out = planner_decoder_layer_batched(
                ctx,
                x,
                positions,
                layer,
                config,
                attention_mask,
                query_mask,
                activation_type_);
            x = out.output;
            keys_.push_back(out.key.tensor);
            values_.push_back(out.value.tensor);
        }
        x = modules::SliceModule({1, prompt_steps_ - 1, 1}).build(ctx, x);
        x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                .build(ctx, x, {weights.norm, std::nullopt});
        x = cast_planner_activation(ctx, x, activation_type_);
        auto logits = modules::LinearModule({config.hidden_size, config.vocab_size, false})
                          .build(ctx, x, {weights.lm_head, std::nullopt});
        auto conditional_logits = ensure_planner_contiguous(
            ctx,
            modules::SliceModule({0, 0, 1}).build(ctx, logits));
        auto unconditional_logits = ensure_planner_contiguous(
            ctx,
            modules::SliceModule({0, 1, 1}).build(ctx, logits));
        logits_cond_ = conditional_logits.tensor;
        logits_uncond_ = unconditional_logits.tensor;
        ggml_set_output(logits_cond_);
        ggml_set_output(logits_uncond_);
        key_cond_outputs_.reserve(keys_.size());
        key_uncond_outputs_.reserve(keys_.size());
        value_cond_outputs_.reserve(values_.size());
        value_uncond_outputs_.reserve(values_.size());
        for (size_t layer_index = 0; layer_index < keys_.size(); ++layer_index) {
            auto key_tensor = core::wrap_tensor(
                keys_[layer_index],
                core::TensorShape::from_dims({2, prompt_steps_, config.num_key_value_heads, config.head_dim}),
                GGML_TYPE_F32);
            auto value_tensor = core::wrap_tensor(
                values_[layer_index],
                core::TensorShape::from_dims({2, prompt_steps_, config.num_key_value_heads, config.head_dim}),
                GGML_TYPE_F32);
            auto key_cond = ensure_planner_contiguous(
                ctx,
                modules::SliceModule({0, 0, 1}).build(ctx, key_tensor));
            auto key_uncond = ensure_planner_contiguous(
                ctx,
                modules::SliceModule({0, 1, 1}).build(ctx, key_tensor));
            auto value_cond = ensure_planner_contiguous(
                ctx,
                modules::SliceModule({0, 0, 1}).build(ctx, value_tensor));
            auto value_uncond = ensure_planner_contiguous(
                ctx,
                modules::SliceModule({0, 1, 1}).build(ctx, value_tensor));
            key_cond_outputs_.push_back(key_cond.tensor);
            key_uncond_outputs_.push_back(key_uncond.tensor);
            value_cond_outputs_.push_back(value_cond.tensor);
            value_uncond_outputs_.push_back(value_uncond.tensor);
            ggml_set_output(key_cond_outputs_.back());
            ggml_set_output(key_uncond_outputs_.back());
            ggml_set_output(value_cond_outputs_.back());
            ggml_set_output(value_uncond_outputs_.back());
        }
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, logits_cond_);
        ggml_build_forward_expand(graph_, logits_uncond_);
        for (size_t layer = 0; layer < key_cond_outputs_.size(); ++layer) {
            ggml_build_forward_expand(graph_, key_cond_outputs_[layer]);
            ggml_build_forward_expand(graph_, key_uncond_outputs_[layer]);
            ggml_build_forward_expand(graph_, value_cond_outputs_[layer]);
            ggml_build_forward_expand(graph_, value_uncond_outputs_[layer]);
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend()));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate ACE-Step planner CFG prefill graph");
        }
    }

    ~CfgPrefillGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool can_run(const PlannerWeightsRuntime & runtime, int64_t prompt_steps) const {
        return runtime_.get() == &runtime && prompt_steps_ == prompt_steps;
    }

    CfgPrefillOutput run(
        const AceStepTokenizedText & conditional_prompt,
        const AceStepTokenizedText & unconditional_prompt) {
        const auto & config = runtime_->assets().config.planner;
        if (static_cast<int64_t>(conditional_prompt.input_ids.size()) != prompt_steps_ ||
            static_cast<int64_t>(unconditional_prompt.input_ids.size()) != prompt_steps_ ||
            static_cast<int64_t>(conditional_prompt.attention_mask.size()) != prompt_steps_ ||
            static_cast<int64_t>(unconditional_prompt.attention_mask.size()) != prompt_steps_) {
            throw std::runtime_error("ACE-Step planner CFG prefill prompt size mismatch");
        }
        std::vector<int32_t> token_ids(static_cast<size_t>(2 * prompt_steps_), 0);
        std::copy(conditional_prompt.input_ids.begin(), conditional_prompt.input_ids.end(), token_ids.begin());
        std::copy(
            unconditional_prompt.input_ids.begin(),
            unconditional_prompt.input_ids.end(),
            token_ids.begin() + static_cast<std::ptrdiff_t>(prompt_steps_));
        ggml_backend_tensor_set(token_ids_, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));

        std::vector<int32_t> position_ids(static_cast<size_t>(2 * prompt_steps_), 0);
        for (int64_t i = 0; i < prompt_steps_; ++i) {
            const int32_t position = static_cast<int32_t>(i);
            position_ids[static_cast<size_t>(i)] = position;
            position_ids[static_cast<size_t>(prompt_steps_ + i)] = position;
        }
        ggml_backend_tensor_set(positions_, position_ids.data(), 0, position_ids.size() * sizeof(int32_t));
        const auto attention_mask_values = build_cfg_prefill_attention_mask_values(
            conditional_prompt.attention_mask,
            unconditional_prompt.attention_mask);
        ggml_backend_tensor_set(
            attention_mask_,
            attention_mask_values.data(),
            0,
            attention_mask_values.size() * sizeof(ggml_fp16_t));
        std::vector<float> query_mask_values(static_cast<size_t>(2 * prompt_steps_), 0.0F);
        for (int64_t i = 0; i < prompt_steps_; ++i) {
            query_mask_values[static_cast<size_t>(i)] =
                conditional_prompt.attention_mask[static_cast<size_t>(i)] != 0 ? 1.0F : 0.0F;
            query_mask_values[static_cast<size_t>(prompt_steps_ + i)] =
                unconditional_prompt.attention_mask[static_cast<size_t>(i)] != 0 ? 1.0F : 0.0F;
        }
        ggml_backend_tensor_set(query_mask_, query_mask_values.data(), 0, query_mask_values.size() * sizeof(float));

        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ACE-Step planner CFG prefill graph compute failed");
        }

        CfgPrefillOutput out;
        out.current_end = prompt_steps_;
        out.valid_steps = prompt_steps_;
        out.conditional_logits.resize(static_cast<size_t>(config.vocab_size));
        out.unconditional_logits.resize(static_cast<size_t>(config.vocab_size));
        ggml_backend_tensor_get(
            logits_cond_,
            out.conditional_logits.data(),
            0,
            out.conditional_logits.size() * sizeof(float));
        ggml_backend_tensor_get(
            logits_uncond_,
            out.unconditional_logits.data(),
            0,
            out.unconditional_logits.size() * sizeof(float));
        out.layers.resize(keys_.size());
        const size_t branch_values = static_cast<size_t>(prompt_steps_ * config.num_key_value_heads * config.head_dim);
        for (size_t layer = 0; layer < key_cond_outputs_.size(); ++layer) {
            auto & state = out.layers[layer];
            state.key.resize(2 * branch_values);
            state.value.resize(2 * branch_values);
            ggml_backend_tensor_get(key_cond_outputs_[layer], state.key.data(), 0, branch_values * sizeof(float));
            ggml_backend_tensor_get(
                key_uncond_outputs_[layer],
                state.key.data() + static_cast<std::ptrdiff_t>(branch_values),
                0,
                branch_values * sizeof(float));
            ggml_backend_tensor_get(value_cond_outputs_[layer], state.value.data(), 0, branch_values * sizeof(float));
            ggml_backend_tensor_get(
                value_uncond_outputs_[layer],
                state.value.data() + static_cast<std::ptrdiff_t>(branch_values),
                0,
                branch_values * sizeof(float));
        }
        return out;
    }

private:
    std::shared_ptr<PlannerWeightsRuntime> runtime_;
    int64_t prompt_steps_ = 0;
    ggml_type activation_type_ = GGML_TYPE_F32;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_ids_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * query_mask_ = nullptr;
    ggml_tensor * logits_cond_ = nullptr;
    ggml_tensor * logits_uncond_ = nullptr;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    std::vector<ggml_tensor *> key_cond_outputs_;
    std::vector<ggml_tensor *> key_uncond_outputs_;
    std::vector<ggml_tensor *> value_cond_outputs_;
    std::vector<ggml_tensor *> value_uncond_outputs_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

class CfgDecodeGraph {
public:
    CfgDecodeGraph(
        std::shared_ptr<PlannerWeightsRuntime> runtime,
        int64_t cache_steps,
        size_t graph_arena_bytes)
        : runtime_(std::move(runtime)),
          cache_steps_(cache_steps),
          activation_type_(planner_phase2_activation_type(runtime_->backend_type())) {
        if (cache_steps_ <= 0) {
            throw std::runtime_error("ACE-Step planner CFG decode requires positive cache length");
        }
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize ACE-Step planner CFG decode graph context");
        }

        const auto & config = runtime_->assets().config.planner;
        const auto & weights = runtime_->weights();

        core::ModuleBuildContext ctx{ctx_.get(), "ace_step.planner.cfg_decode", runtime_->backend_type()};
        token_ids_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 2);
        auto token_ids = core::wrap_tensor(token_ids_, core::TensorShape::from_dims({2}), GGML_TYPE_I32);
        auto x = modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                     .build(ctx, token_ids, weights.token_embedding);
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({2, 1, config.hidden_size}));
        x = cast_planner_activation(ctx, x, activation_type_);
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 2);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({2}), GGML_TYPE_I32);
        cache_slot_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 2);
        auto cache_slot = core::wrap_tensor(cache_slot_, core::TensorShape::from_dims({2}), GGML_TYPE_I32);
        attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, cache_steps_, 1, 1, 2);
        auto attention_mask = core::wrap_tensor(
            attention_mask_,
            core::TensorShape::from_dims({2, 1, 1, cache_steps_}),
            GGML_TYPE_F16);

        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        for (const auto & layer : weights.layers.layers) {
            auto cache_key = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({2, cache_steps_, config.num_key_value_heads, config.head_dim}));
            auto cache_value = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({2, cache_steps_, config.num_key_value_heads, config.head_dim}));
            cache_keys_.push_back(cache_key);
            cache_values_.push_back(cache_value);
            auto out = planner_decoder_layer_with_compact_cache_batched(
                ctx,
                x,
                positions,
                layer,
                config,
                cache_key,
                cache_value,
                cache_slot,
                attention_mask,
                activation_type_);
            x = out.output;
        }

        x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                .build(ctx, x, {weights.norm, std::nullopt});
        x = cast_planner_activation(ctx, x, activation_type_);
        auto logits = modules::LinearModule({config.hidden_size, config.vocab_size, false})
                          .build(ctx, x, {weights.lm_head, std::nullopt});
        auto conditional_logits = ensure_planner_contiguous(
            ctx,
            modules::SliceModule({0, 0, 1}).build(ctx, logits));
        auto unconditional_logits = ensure_planner_contiguous(
            ctx,
            modules::SliceModule({0, 1, 1}).build(ctx, logits));
        logits_cond_ = conditional_logits.tensor;
        logits_uncond_ = unconditional_logits.tensor;
        ggml_set_output(logits_cond_);
        ggml_set_output(logits_uncond_);
        ggml_build_forward_expand(graph_, logits_cond_);
        ggml_build_forward_expand(graph_, logits_uncond_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate ACE-Step planner CFG decode graph");
        }

        attention_mask_values_.assign(
            static_cast<size_t>(2 * cache_steps_),
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
    }

    ~CfgDecodeGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool can_run(const PlannerWeightsRuntime & runtime, int64_t required_steps) const {
        return runtime_.get() == &runtime && cache_steps_ >= required_steps;
    }

    void initialize(const CfgPrefillOutput & prefill) {
        if (prefill.layers.size() != cache_keys_.size()) {
            throw std::runtime_error("ACE-Step planner CFG decode state layer count mismatch");
        }
        const int64_t step_elems = runtime_->assets().config.planner.num_key_value_heads * runtime_->assets().config.planner.head_dim;
        current_end_ = prefill.current_end;
        valid_steps_ = prefill.valid_steps;
        if (valid_steps_ > cache_steps_ || current_end_ > cache_steps_) {
            throw std::runtime_error("ACE-Step planner CFG decode state exceeds cache capacity");
        }
        const size_t branch_values = static_cast<size_t>(valid_steps_ * step_elems);
        const size_t total_values = static_cast<size_t>(2 * cache_steps_ * step_elems);
        const size_t uncond_offset = static_cast<size_t>(cache_steps_ * step_elems);
        for (size_t layer = 0; layer < cache_keys_.size(); ++layer) {
            const auto & source = prefill.layers[layer];
            if (source.key.size() != 2 * branch_values || source.value.size() != 2 * branch_values) {
                throw std::runtime_error("ACE-Step planner CFG decode KV tensor size mismatch");
            }
            std::vector<float> cache_key_values(total_values, 0.0F);
            std::vector<float> cache_value_values(total_values, 0.0F);
            std::copy(source.key.begin(), source.key.begin() + static_cast<std::ptrdiff_t>(branch_values), cache_key_values.begin());
            std::copy(source.value.begin(), source.value.begin() + static_cast<std::ptrdiff_t>(branch_values), cache_value_values.begin());
            std::copy(
                source.key.begin() + static_cast<std::ptrdiff_t>(branch_values),
                source.key.end(),
                cache_key_values.begin() + static_cast<std::ptrdiff_t>(uncond_offset));
            std::copy(
                source.value.begin() + static_cast<std::ptrdiff_t>(branch_values),
                source.value.end(),
                cache_value_values.begin() + static_cast<std::ptrdiff_t>(uncond_offset));
            core::write_tensor_f32(cache_keys_[layer], cache_key_values);
            core::write_tensor_f32(cache_values_[layer], cache_value_values);
        }
    }

    void run_step_into(
        int32_t token,
        const std::vector<int32_t> & conditional_visible,
        const std::vector<int32_t> & unconditional_visible,
        std::vector<float> & conditional_logits,
        std::vector<float> & unconditional_logits) {
        run_step_into(
            token,
            token,
            conditional_visible,
            unconditional_visible,
            conditional_logits,
            unconditional_logits);
    }

    void run_step_into(
        int32_t conditional_token,
        int32_t unconditional_token,
        const std::vector<int32_t> & conditional_visible,
        const std::vector<int32_t> & unconditional_visible,
        std::vector<float> & conditional_logits,
        std::vector<float> & unconditional_logits) {
        if (current_end_ >= cache_steps_) {
            throw std::runtime_error("ACE-Step planner CFG decode cache exhausted");
        }
        if (static_cast<int64_t>(conditional_visible.size()) != current_end_ ||
            static_cast<int64_t>(unconditional_visible.size()) != current_end_) {
            throw std::runtime_error("ACE-Step planner CFG decode visible-prefix length mismatch");
        }

        const int32_t tokens[2] = {conditional_token, unconditional_token};
        ggml_backend_tensor_set(token_ids_, tokens, 0, sizeof(tokens));
        int32_t positions[2] = {
            static_cast<int32_t>(current_end_),
            static_cast<int32_t>(current_end_),
        };
        ggml_backend_tensor_set(positions_, positions, 0, sizeof(positions));
        const int32_t cache_slots[2] = {
            static_cast<int32_t>(current_end_),
            static_cast<int32_t>(cache_steps_ + current_end_),
        };
        ggml_backend_tensor_set(cache_slot_, cache_slots, 0, sizeof(cache_slots));

        const auto masked = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        const auto visible = ggml_fp32_to_fp16(0.0F);
        std::fill(attention_mask_values_.begin(), attention_mask_values_.end(), masked);
        for (int64_t i = 0; i < valid_steps_; ++i) {
            if (conditional_visible[static_cast<size_t>(i)] != 0) {
                attention_mask_values_[static_cast<size_t>(i)] = visible;
            }
            if (unconditional_visible[static_cast<size_t>(i)] != 0) {
                attention_mask_values_[static_cast<size_t>(cache_steps_ + i)] = visible;
            }
        }
        attention_mask_values_[static_cast<size_t>(current_end_)] = visible;
        attention_mask_values_[static_cast<size_t>(cache_steps_ + current_end_)] = visible;
        ggml_backend_tensor_set(
            attention_mask_,
            attention_mask_values_.data(),
            0,
            attention_mask_values_.size() * sizeof(ggml_fp16_t));

        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ACE-Step planner CFG decode graph compute failed");
        }

        ++current_end_;
        ++valid_steps_;

        const size_t vocab_size = static_cast<size_t>(runtime_->assets().config.planner.vocab_size);
        conditional_logits.resize(vocab_size);
        unconditional_logits.resize(vocab_size);
        ggml_backend_tensor_get(
            logits_cond_,
            conditional_logits.data(),
            0,
            conditional_logits.size() * sizeof(float));
        ggml_backend_tensor_get(
            logits_uncond_,
            unconditional_logits.data(),
            0,
            unconditional_logits.size() * sizeof(float));
    }

    CfgPrefillOutput export_prefill() const {
        const int64_t step_elems = runtime_->assets().config.planner.num_key_value_heads * runtime_->assets().config.planner.head_dim;
        const size_t branch_values = static_cast<size_t>(valid_steps_ * step_elems);
        const size_t uncond_offset = static_cast<size_t>(cache_steps_ * step_elems);
        CfgPrefillOutput out;
        out.current_end = current_end_;
        out.valid_steps = valid_steps_;
        out.layers.resize(cache_keys_.size());
        for (size_t layer = 0; layer < cache_keys_.size(); ++layer) {
            const auto key_values = core::read_tensor_f32(cache_keys_[layer].tensor);
            const auto value_values = core::read_tensor_f32(cache_values_[layer].tensor);
            auto & target = out.layers[layer];
            target.key.reserve(2 * branch_values);
            target.value.reserve(2 * branch_values);
            target.key.insert(target.key.end(), key_values.begin(), key_values.begin() + static_cast<std::ptrdiff_t>(branch_values));
            target.key.insert(
                target.key.end(),
                key_values.begin() + static_cast<std::ptrdiff_t>(uncond_offset),
                key_values.begin() + static_cast<std::ptrdiff_t>(uncond_offset + branch_values));
            target.value.insert(
                target.value.end(),
                value_values.begin(),
                value_values.begin() + static_cast<std::ptrdiff_t>(branch_values));
            target.value.insert(
                target.value.end(),
                value_values.begin() + static_cast<std::ptrdiff_t>(uncond_offset),
                value_values.begin() + static_cast<std::ptrdiff_t>(uncond_offset + branch_values));
        }
        return out;
    }

private:
    std::shared_ptr<PlannerWeightsRuntime> runtime_;
    int64_t cache_steps_ = 0;
    int64_t current_end_ = 0;
    int64_t valid_steps_ = 0;
    ggml_type activation_type_ = GGML_TYPE_F32;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_ids_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * cache_slot_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * logits_cond_ = nullptr;
    ggml_tensor * logits_uncond_ = nullptr;
    std::vector<core::TensorValue> cache_keys_;
    std::vector<core::TensorValue> cache_values_;
    std::vector<ggml_fp16_t> attention_mask_values_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

CfgPrefillOutput run_cfg_decode_graph_prompt_prefill(
    CfgDecodeGraph & graph,
    const AceStepTokenizedText & conditional_prompt,
    const AceStepTokenizedText & unconditional_prompt,
    size_t layer_count) {
    if (conditional_prompt.input_ids.size() != unconditional_prompt.input_ids.size() ||
        conditional_prompt.attention_mask.size() != unconditional_prompt.attention_mask.size() ||
        conditional_prompt.input_ids.size() != conditional_prompt.attention_mask.size()) {
        throw std::runtime_error("ACE-Step planner CFG decode prefill prompt size mismatch");
    }
    CfgPrefillOutput empty;
    empty.layers.resize(layer_count);
    graph.initialize(empty);
    std::vector<float> conditional_logits;
    std::vector<float> unconditional_logits;
    for (size_t index = 0; index < conditional_prompt.input_ids.size(); ++index) {
        std::vector<int32_t> conditional_visible(
            conditional_prompt.attention_mask.begin(),
            conditional_prompt.attention_mask.begin() + static_cast<std::ptrdiff_t>(index));
        std::vector<int32_t> unconditional_visible(
            unconditional_prompt.attention_mask.begin(),
            unconditional_prompt.attention_mask.begin() + static_cast<std::ptrdiff_t>(index));
        graph.run_step_into(
            conditional_prompt.input_ids[index],
            unconditional_prompt.input_ids[index],
            conditional_visible,
            unconditional_visible,
            conditional_logits,
            unconditional_logits);
    }
    auto out = graph.export_prefill();
    out.conditional_logits = std::move(conditional_logits);
    out.unconditional_logits = std::move(unconditional_logits);
    return out;
}

AceStepPlannerPreparedInput prepare_phase1_prompt(const AceStepTextTokenizer & tokenizer, const AceStepRequest & request, int64_t max_prompt_tokens) {
    const std::string user_prompt = ace_step_build_lm_prompt(request.prompt, request.lyrics);
    const std::string formatted = tokenizer.apply_chat_template(
        std::string("# Instruction\n") + kDefaultLmInstruction + "\n\n",
        user_prompt,
        true);
    return {formatted, trim_to_valid_tokens(tokenizer.tokenize_text(formatted, max_prompt_tokens))};
}

AceStepPlannerPreparedInput prepare_phase2_prompt(
    const AceStepTextTokenizer & tokenizer,
    const AceStepRequest & request,
    const AceStepPlan & cot_plan,
    int64_t max_prompt_tokens) {
    const std::string user_prompt = ace_step_build_lm_prompt(request.prompt, request.lyrics);
    std::string formatted = tokenizer.apply_chat_template(
        std::string("# Instruction\n") + kDefaultLmInstruction + "\n\n",
        user_prompt,
        true);
    formatted += format_cot_text(cot_plan) + "\n\n";
    return {formatted, trim_to_valid_tokens(tokenizer.tokenize_text(formatted, max_prompt_tokens))};
}

AceStepPlannerPreparedInput prepare_phase2_unconditional_prompt(
    const AceStepTextTokenizer & tokenizer,
    const AceStepRequest & request,
    int64_t max_prompt_tokens) {
    std::string formatted = tokenizer.apply_chat_template(
        std::string("# Instruction\n") + kDefaultLmInstruction + "\n\n",
        normalized_negative_prompt(request),
        true);
    formatted += empty_cot_text() + "\n\n";
    return {formatted, trim_to_valid_tokens(tokenizer.tokenize_text(formatted, max_prompt_tokens))};
}

AceStepPlannerRuntime::AceStepPlannerRuntime(
    std::shared_ptr<const AceStepAssets> assets,
    core::ExecutionContext & execution)
    : AceStepPlannerRuntime(
          std::move(assets),
          execution,
          assets::TensorStorageType::Native,
          256ull * 1024ull * 1024ull,
          128ull * 1024ull * 1024ull,
          GenerationConfig{4096, 512, 4032}) {}

AceStepPlannerRuntime::AceStepPlannerRuntime(
    std::shared_ptr<const AceStepAssets> assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType weight_storage_type,
    size_t decode_graph_arena_bytes,
    size_t weight_context_bytes,
    GenerationConfig generation)
    : assets_(std::move(assets)),
      tokenizer_(assets_, AceStepTextTokenizer::ResourceSet::Planner),
      generation_(generation),
      weights_runtime_(std::make_shared<PlannerWeightsRuntime>(
          assets_,
          execution,
          weight_context_bytes,
          weight_storage_type)),
      decode_graph_arena_bytes_(decode_graph_arena_bytes),
      phase1_constraints_(build_phase1_constraint_tables(tokenizer_, assets_->config.planner)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("ACE-Step planner runtime requires assets");
    }
    if (planner_prefill_uses_host_backend(execution.backend_type())) {
        core::BackendConfig config;
        config.type = core::BackendType::Cpu;
        config.threads = execution.config().threads;
        host_planner_prefill_execution_ = std::make_unique<core::ExecutionContext>(config);
        planner_prefill_weights_runtime_ = std::make_shared<PlannerWeightsRuntime>(
            assets_,
            *host_planner_prefill_execution_,
            weight_context_bytes,
            weight_storage_type);
    }
    is_audio_code_token_.assign(static_cast<size_t>(assets_->config.planner.vocab_size), 0);
    assets_->lm_weights->release_storage();
    for (int32_t code = 0; code < 64000; ++code) {
        const auto token_id = tokenizer_.find_token_id("<|audio_code_" + std::to_string(code) + "|>");
        if (!token_id.has_value()) {
            throw std::runtime_error("ACE-Step planner tokenizer is missing audio code token " + std::to_string(code));
        }
        if (*token_id >= 0 && *token_id < static_cast<int32_t>(is_audio_code_token_.size())) {
            is_audio_code_token_[static_cast<size_t>(*token_id)] = 1;
        }
    }
    for (int32_t token = 0; token < static_cast<int32_t>(is_audio_code_token_.size()); ++token) {
        if (is_audio_code_token_[static_cast<size_t>(token)] != 0 ||
            token == assets_->config.planner.eos_token_id) {
            phase2_candidate_token_ids_.push_back(token);
        }
    }
}

AceStepPlannerRuntime::~AceStepPlannerRuntime() = default;

AceStepPlannerPreparedInput AceStepPlannerRuntime::prepare_prompt(const AceStepRequest & request) const {
    return prepare_phase1_prompt(tokenizer_, request, generation_.max_prompt_tokens);
}

std::string AceStepPlannerRuntime::decode_tokens(const std::vector<int32_t> & token_ids) const {
    return tokenizer_.decode(token_ids, false);
}

AceStepPlan AceStepPlannerRuntime::parse_output(const std::string & output_text) const {
    return ace_step_parse_lm_output(output_text);
}

void AceStepPlannerRuntime::release_graph_workspace() const {
    cfg_decode_graph_.reset();
    cfg_prefill_graph_.reset();
    decode_graph_.reset();
    prefill_graph_.reset();
}

AceStepPlan AceStepPlannerRuntime::generate(const AceStepRequest & request, bool generate_audio_codes) const {
    const auto total_start = Clock::now();
    double cot_decode_graph_step_ms = 0.0;
    double code_decode_graph_step_ms = 0.0;
    double code_sampling_ms = 0.0;
    const auto prepare_prefill = [&](
        const AceStepPlannerPreparedInput & prepared,
        int64_t max_new_tokens,
        std::string_view timing_prefix) {
        const auto & config = assets_->config.planner;
        const auto & prompt_ids = prepared.tokenized_prompt.input_ids;
        const auto & attention_mask = prepared.tokenized_prompt.attention_mask;
        if (static_cast<int64_t>(prompt_ids.size()) + max_new_tokens > config.max_position_embeddings) {
            throw std::runtime_error("ACE-Step planner request exceeds max_position_embeddings");
        }
        auto prefill_runtime = planner_prefill_weights_runtime_ != nullptr
            ? planner_prefill_weights_runtime_
            : weights_runtime_;
        const auto graph_prepare_start = Clock::now();
        const int64_t required_cache_steps = static_cast<int64_t>(prompt_ids.size()) + max_new_tokens;
        if (!decode_graph_ || !decode_graph_->can_run(*weights_runtime_, required_cache_steps)) {
            decode_graph_ = std::make_unique<DecodeGraph>(
                weights_runtime_,
                required_cache_steps,
                decode_graph_arena_bytes_);
        }
        const bool use_metal_prompt_step_prefill = prefill_runtime->backend_type() == core::BackendType::Metal;
        if (!use_metal_prompt_step_prefill &&
            (!prefill_graph_ || !prefill_graph_->can_run(*prefill_runtime, static_cast<int64_t>(prompt_ids.size())))) {
            prefill_graph_ = std::make_unique<PrefillGraph>(
                prefill_runtime,
                static_cast<int64_t>(prompt_ids.size()),
                decode_graph_arena_bytes_);
        }
        engine::debug::timing_log_scalar(
            std::string(timing_prefix) + ".prefill.graph.prepare_ms",
            engine::debug::elapsed_ms(graph_prepare_start, Clock::now()));
        engine::debug::trace_log_scalar(
            std::string(timing_prefix) + ".prefill_prompt_tokens",
            prompt_ids.size());
        engine::debug::trace_log_scalar(
            std::string(timing_prefix) + ".prefill_cache_steps",
            required_cache_steps);
        const auto prefill_run_start = Clock::now();
        PrefillOutput out;
        if (use_metal_prompt_step_prefill) {
            out = run_decode_graph_prompt_prefill(*decode_graph_, prepared.tokenized_prompt);
        } else {
            out = prefill_graph_->run(prompt_ids, attention_mask);
        }
        engine::debug::timing_log_scalar(
            std::string(timing_prefix) + ".prefill_run_ms",
            engine::debug::elapsed_ms(prefill_run_start, Clock::now()));
        return out;
    };

    const auto decode_cot_phase = [&](const AceStepPlannerPreparedInput & prepared) {
        const auto phase_start = Clock::now();
        std::vector<int32_t> generated;
        generated.reserve(static_cast<size_t>(generation_.max_cot_tokens));
        PrefillOutput prefill = prepare_prefill(prepared, generation_.max_cot_tokens, "ace_step.planner.cot");
        std::vector<float> logits = std::move(prefill.logits);
        decode_graph_->import_state(prefill.kv_state);
        Phase1ConstrainedDecoder constrained_decoder(
            tokenizer_,
            assets_->config.planner,
            request,
            is_audio_code_token_,
            phase1_constraints_);
        for (int64_t step = 0; step < generation_.max_cot_tokens; ++step) {
            const int32_t token = constrained_decoder.select_next_token(logits);
            generated.push_back(token);
            constrained_decoder.observe_token(token);
            if (constrained_decoder.completed()) {
                break;
            }
            const auto step_start = Clock::now();
            decode_graph_->run_step_into(token, logits);
            cot_decode_graph_step_ms += engine::debug::elapsed_ms(step_start, Clock::now());
        }
        engine::debug::timing_log_scalar("ace_step.planner.cot.decode.graph.step_ms", cot_decode_graph_step_ms);
        engine::debug::timing_log_scalar("ace_step.planner.cot.phase_ms", engine::debug::elapsed_ms(phase_start, Clock::now()));
        return generated;
    };

    const auto decode_code_phase = [&](const AceStepPlannerPreparedInput & prepared, int64_t desired_codes) {
        const auto phase_start = Clock::now();
        if (desired_codes <= 0) {
            throw std::runtime_error("ACE-Step planner requires a positive target audio-code count");
        }
        if (desired_codes > generation_.max_code_tokens) {
            throw std::runtime_error("ACE-Step planner target audio-code count exceeds configured decode budget");
        }
        const int64_t max_new_tokens = planner_codes_phase_max_new_tokens(request, generation_.max_code_tokens);
        std::vector<int32_t> generated;
        generated.reserve(static_cast<size_t>(max_new_tokens));
        std::mt19937 rng(request.generation.seed);
        const float lm_cfg_scale = request.generation.lm_cfg_scale;
        const bool use_cfg = lm_cfg_scale > 1.0F;
        std::vector<float> conditional_logits;
        std::vector<int32_t> conditional_visible;
        std::vector<int32_t> conditional_history;
        std::vector<float> unconditional_logits;
        std::vector<int32_t> unconditional_visible;
        Phase2CodesProcessor phase2_processor(
            assets_->config.planner.eos_token_id,
            desired_codes);
        Phase2CandidateSampler phase2_sampler(
            phase2_candidate_token_ids_,
            assets_->config.planner.eos_token_id);
        if (use_cfg) {
            auto unconditional_prompt =
                prepare_phase2_unconditional_prompt(tokenizer_, request, generation_.max_prompt_tokens);
            auto aligned = left_pad_pair_to_shared_length(
                prepared.tokenized_prompt,
                unconditional_prompt.tokenized_prompt,
                tokenizer_.pad_token_id());
            AceStepPlannerPreparedInput conditional_prompt = prepared;
            conditional_prompt.tokenized_prompt = std::move(aligned.first);
            unconditional_prompt.tokenized_prompt = std::move(aligned.second);
            const int64_t prompt_steps = static_cast<int64_t>(conditional_prompt.tokenized_prompt.input_ids.size());
            auto cfg_prefill_runtime =
                planner_prefill_weights_runtime_ != nullptr ? planner_prefill_weights_runtime_ : weights_runtime_;
            const bool use_metal_prompt_step_cfg_prefill =
                cfg_prefill_runtime->backend_type() == core::BackendType::Metal;
            const int64_t cfg_required_cache_steps = prompt_steps + max_new_tokens;
            const auto cfg_graph_prepare_start = Clock::now();
            if (!use_metal_prompt_step_cfg_prefill &&
                (!cfg_prefill_graph_ ||
                 !cfg_prefill_graph_->can_run(*cfg_prefill_runtime, prompt_steps))) {
                cfg_prefill_graph_ = std::make_unique<CfgPrefillGraph>(
                    cfg_prefill_runtime,
                    prompt_steps,
                    decode_graph_arena_bytes_);
            }
            engine::debug::timing_log_scalar(
                "ace_step.planner.code.cfg_prefill.graph.prepare_ms",
                engine::debug::elapsed_ms(cfg_graph_prepare_start, Clock::now()));
            const auto cfg_prefill_run_start = Clock::now();
            CfgPrefillOutput cfg_prefill;
            if (use_metal_prompt_step_cfg_prefill) {
                CfgDecodeGraph prompt_graph(weights_runtime_, cfg_required_cache_steps, decode_graph_arena_bytes_);
                cfg_prefill = run_cfg_decode_graph_prompt_prefill(
                    prompt_graph,
                    conditional_prompt.tokenized_prompt,
                    unconditional_prompt.tokenized_prompt,
                    weights_runtime_->weights().layers.layers.size());
            } else {
                cfg_prefill =
                    cfg_prefill_graph_->run(conditional_prompt.tokenized_prompt, unconditional_prompt.tokenized_prompt);
            }
            engine::debug::timing_log_scalar(
                "ace_step.planner.code.cfg_prefill_run_ms",
                engine::debug::elapsed_ms(cfg_prefill_run_start, Clock::now()));
            conditional_logits = std::move(cfg_prefill.conditional_logits);
            unconditional_logits = std::move(cfg_prefill.unconditional_logits);
            conditional_history = conditional_prompt.tokenized_prompt.input_ids;
            conditional_visible = conditional_prompt.tokenized_prompt.attention_mask;
            unconditional_visible = unconditional_prompt.tokenized_prompt.attention_mask;
            const auto cfg_decode_prepare_start = Clock::now();
            if (!cfg_decode_graph_ || !cfg_decode_graph_->can_run(*weights_runtime_, cfg_required_cache_steps)) {
                cfg_decode_graph_ = std::make_unique<CfgDecodeGraph>(
                    weights_runtime_,
                    cfg_required_cache_steps,
                    decode_graph_arena_bytes_);
            }
            cfg_decode_graph_->initialize(cfg_prefill);
            engine::debug::timing_log_scalar(
                "ace_step.planner.code.cfg_decode.graph.prepare_ms",
                engine::debug::elapsed_ms(cfg_decode_prepare_start, Clock::now()));
            engine::debug::trace_log_scalar("ace_step.planner.code.prefill_prompt_tokens",
                prompt_steps);
            engine::debug::trace_log_scalar("ace_step.planner.code.prefill_cache_steps",
                cfg_required_cache_steps);
        } else {
            PrefillOutput conditional_prefill =
                prepare_prefill(prepared, max_new_tokens, "ace_step.planner.code");
            conditional_logits = std::move(conditional_prefill.logits);
            conditional_history = prepared.tokenized_prompt.input_ids;
            conditional_visible = prepared.tokenized_prompt.attention_mask;
            decode_graph_->import_state(conditional_prefill.kv_state);
        }
        for (int64_t step = 0; step < max_new_tokens; ++step) {
            const auto sampling_start = Clock::now();
            if (use_cfg) {
                phase2_sampler.write_cfg_logits(
                    conditional_logits,
                    unconditional_logits,
                    lm_cfg_scale);
            } else {
                phase2_sampler.write_branch_logits(conditional_logits);
            }
            phase2_sampler.apply_duration_constraint(
                static_cast<int64_t>(generated.size()),
                desired_codes);
            phase2_sampler.apply_repetition_penalty(
                conditional_history,
                request.generation.lm_repetition_penalty);
            phase2_sampler.apply_top_k_filter(request.generation.lm_top_k);
            phase2_sampler.apply_top_p_filter(request.generation.lm_top_p);
            const int32_t token = phase2_sampler.sample(request.generation.lm_temperature, rng);
            generated.push_back(token);
            conditional_history.push_back(token);
            phase2_processor.update_state(token);
            code_sampling_ms += engine::debug::elapsed_ms(sampling_start, Clock::now());
            if (phase2_processor.completed() || is_eos(assets_->config.planner, token)) {
                break;
            }
            const auto step_start = Clock::now();
            if (use_cfg) {
                cfg_decode_graph_->run_step_into(
                    token,
                    conditional_visible,
                    unconditional_visible,
                    conditional_logits,
                    unconditional_logits);
                conditional_visible.push_back(1);
                unconditional_visible.push_back(1);
            } else {
                decode_graph_->run_step_into(token, conditional_logits, &conditional_visible);
                conditional_visible.push_back(1);
            }
            code_decode_graph_step_ms += engine::debug::elapsed_ms(step_start, Clock::now());
        }
        engine::debug::trace_log_scalar("ace_step.planner.code.use_cfg", use_cfg);
        engine::debug::trace_log_scalar("ace_step.planner.code.desired_codes", desired_codes);
        engine::debug::timing_log_scalar("ace_step.planner.code.sampling_ms", code_sampling_ms);
        engine::debug::timing_log_scalar("ace_step.planner.code.decode.graph.step_ms", code_decode_graph_step_ms);
        engine::debug::timing_log_scalar("ace_step.planner.code.phase_ms", engine::debug::elapsed_ms(phase_start, Clock::now()));
        return generated;
    };

    AceStepPlan plan;
    if (!request.generation.use_cot_metas || has_all_request_metas(request)) {
        const auto metadata_start = Clock::now();
        plan.metadata = request_metadata(request);
        engine::debug::timing_log_scalar(
            "ace_step.planner.request_metadata_ms",
            engine::debug::elapsed_ms(metadata_start, Clock::now()));
        engine::debug::trace_log_scalar("ace_step.planner.used_request_metadata", true);
    } else {
        const auto cot_prompt_start = Clock::now();
        const auto cot_prompt = prepare_phase1_prompt(tokenizer_, request, generation_.max_prompt_tokens);
        engine::debug::timing_log_scalar(
            "ace_step.planner.cot.prompt_prepare_ms",
            engine::debug::elapsed_ms(cot_prompt_start, Clock::now()));
        std::vector<int32_t> cot_tokens = decode_cot_phase(cot_prompt);
        const auto cot_decode_text_start = Clock::now();
        std::string cot_text = decode_tokens(cot_tokens);
        plan = parse_output(cot_text);
        engine::debug::timing_log_scalar(
            "ace_step.planner.cot.decode_text_parse_ms",
            engine::debug::elapsed_ms(cot_decode_text_start, Clock::now()));
        engine::debug::trace_log_scalar("ace_step.planner.used_request_metadata", false);
    }
    const auto overlay_start = Clock::now();
    overlay_request_metadata(plan, request);
    engine::debug::timing_log_scalar(
        "ace_step.planner.overlay_metadata_ms",
        engine::debug::elapsed_ms(overlay_start, Clock::now()));

    if (!generate_audio_codes) {
        engine::debug::trace_log_scalar("ace_step.planner.code.use_cfg", false);
        engine::debug::trace_log_scalar("ace_step.planner.code.desired_codes", 0);
        engine::debug::timing_log_scalar("ace_step.planner.code.sampling_ms", 0.0);
        engine::debug::timing_log_scalar("ace_step.planner.code.decode.graph.step_ms", 0.0);
        engine::debug::timing_log_scalar("ace_step.planner.code.phase_ms", 0.0);
        engine::debug::timing_log_scalar("ace_step.planner.finalize_ms", 0.0);
        engine::debug::timing_log_scalar("ace_step.planner.total_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
        return plan;
    }

    const auto code_prompt_start = Clock::now();
    const auto code_prompt = prepare_phase2_prompt(tokenizer_, request, plan, generation_.max_prompt_tokens);
    const int64_t desired_codes = target_code_count(request, plan);
    engine::debug::timing_log_scalar(
        "ace_step.planner.code.prompt_prepare_ms",
        engine::debug::elapsed_ms(code_prompt_start, Clock::now()));
    std::vector<int32_t> code_tokens = decode_code_phase(code_prompt, desired_codes);
    const auto code_decode_text_start = Clock::now();
    std::string code_text = decode_tokens(code_tokens);
    AceStepPlan codes_plan = parse_output(code_text);
    engine::debug::timing_log_scalar(
        "ace_step.planner.code.decode_text_parse_ms",
        engine::debug::elapsed_ms(code_decode_text_start, Clock::now()));
    const auto finalize_start = Clock::now();
    if (codes_plan.metadata.bpm.has_value() && !request.bpm.has_value()) {
        plan.metadata.bpm = codes_plan.metadata.bpm;
    }
    if (codes_plan.metadata.duration.has_value() && !(request.generation.duration_seconds > 0.0F)) {
        plan.metadata.duration = codes_plan.metadata.duration;
    }
    if (codes_plan.metadata.keyscale.has_value() && !request.keyscale.has_value()) {
        plan.metadata.keyscale = codes_plan.metadata.keyscale;
    }
    if (codes_plan.metadata.timesignature.has_value() && !request.timesignature.has_value()) {
        plan.metadata.timesignature = codes_plan.metadata.timesignature;
    }
    if (codes_plan.metadata.language.has_value() &&
        (request.vocal_language.empty() || request.vocal_language == "unknown")) {
        plan.metadata.language = codes_plan.metadata.language;
    }
    if (codes_plan.metadata.genres.has_value()) {
        plan.metadata.genres = codes_plan.metadata.genres;
    }
    if (!codes_plan.caption.empty()) {
        plan.caption = codes_plan.caption;
    }
    if (!codes_plan.cot_caption.empty()) {
        plan.cot_caption = codes_plan.cot_caption;
    }
    plan.audio_codes_text = std::move(codes_plan.audio_codes_text);
    plan.audio_code_ids = std::move(codes_plan.audio_code_ids);
    plan.frames_5hz = static_cast<int64_t>(plan.audio_code_ids.size());

    if (plan.frames_5hz > desired_codes) {
        plan.audio_code_ids.resize(static_cast<size_t>(desired_codes));
        plan.frames_5hz = desired_codes;
        std::ostringstream rebuilt;
        for (const int32_t code : plan.audio_code_ids) {
            rebuilt << "<|audio_code_" << code << "|>";
        }
        plan.audio_codes_text = rebuilt.str();
    }
    engine::debug::timing_log_scalar("ace_step.planner.finalize_ms", engine::debug::elapsed_ms(finalize_start, Clock::now()));
    engine::debug::timing_log_scalar("ace_step.planner.total_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
    return plan;
}

}  // namespace engine::models::ace_step
