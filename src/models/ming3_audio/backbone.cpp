#include "engine/models/ming3_audio/backbone.h"

#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/packed_linear_weights.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/kv_cache.h"

#include <algorithm>
#include <cmath>
#include <ggml-alloc.h>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace engine::models::ming3_audio {

ThinkerWeights::ThinkerWeights(core::ExecutionContext & execution, const assets::TensorSource & source,
                               const ThinkerConfig & c, const ThinkerOptions & options)
    : store(execution.backend(), execution.backend_type(), "ming3_audio.thinker.weights", 4 * 1024 * 1024) {
    namespace binding = modules::binding;
    const auto native = assets::TensorStorageType::Native;
    if (options.disk_experts) {
        if (options.expert_cache_size < c.experts_per_token || options.expert_cache_size > c.experts) {
            throw std::runtime_error("Ming3 Audio expert_cache_size must cover top-k and not exceed the expert count");
        }
        disk_store = std::make_unique<core::DiskExpertWeightStore>(source.source_path(), options.pin_experts,
            static_cast<size_t>(options.disk_expert_read_threads));
    }
    embedding = store.load_tensor(source, "model.word_embeddings.weight", native, {c.vocabulary_size, c.hidden_size});
    head = store.load_tensor(source, "lm_head.weight", native, {c.vocabulary_size, c.hidden_size});
    norm = binding::norm_weight_from_source(store, source, "model.norm", c.hidden_size);
    std::vector<float> identity(c.expert_groups * c.expert_groups, 0.0f);
    for (int64_t i = 0; i < c.expert_groups; ++i) identity[i * c.expert_groups + i] = 1.0f;
    group_identity = store.make_f32(core::TensorShape::from_dims({c.expert_groups, c.expert_groups}), std::move(identity));
    for (int64_t layer = 0; layer < c.layers; ++layer) {
        const auto prefix = "model.layers." + std::to_string(layer);
        const auto attn = prefix + ".attention";
        ThinkerLayerWeights w;
        w.input_norm = binding::norm_weight_from_source(store, source, prefix + ".input_layernorm", c.hidden_size);
        w.post_norm = binding::norm_weight_from_source(store, source, prefix + ".post_attention_layernorm", c.hidden_size);
        const bool mla = (layer + 1) % c.attention_interval == 0 ||
                         layer >= c.layers / c.attention_interval * c.attention_interval;
        if (mla) {
            auto & a = w.mla;
            a.query_down = binding::linear_from_source(store, source, attn + ".q_a_proj", native, c.query_rank, c.hidden_size, false);
            a.query_norm = binding::norm_weight_from_source(store, source, attn + ".q_a_layernorm", c.query_rank);
            a.query_up = binding::linear_from_source(store, source, attn + ".q_b_proj", native,
                c.heads * (c.qk_nope_dim + c.qk_rope_dim), c.query_rank, false);
            a.kv_down = binding::linear_from_source(store, source, attn + ".kv_a_proj_with_mqa", native,
                c.kv_rank + c.qk_rope_dim, c.hidden_size, false);
            a.kv_norm = binding::norm_weight_from_source(store, source, attn + ".kv_a_layernorm", c.kv_rank);
            a.kv_up = binding::linear_from_source(store, source, attn + ".kv_b_proj", native,
                c.heads * (c.qk_nope_dim + c.value_dim), c.kv_rank, false);
            a.gate = binding::linear_from_source(store, source, attn + ".g_proj", native, c.heads, c.hidden_size, false);
            a.output = binding::linear_from_source(store, source, attn + ".dense", native, c.hidden_size, c.heads * c.value_dim, false);
        } else {
            auto & a = w.kda;
            const int64_t width = c.heads * c.head_dim;
            a.input_proj = modules::PackedLinearWeightsBuilder({c.hidden_size, {
                {attn + ".q_proj.weight", std::nullopt, width},
                {attn + ".k_proj.weight", std::nullopt, width},
                {attn + ".v_proj.weight", std::nullopt, width},
                {attn + ".f_proj.weight", std::nullopt, width},
                {attn + ".g_proj.weight", std::nullopt, width},
                {attn + ".b_proj.weight", std::nullopt, c.heads}}, false, true}).build(store, source, native);
            const std::array<std::string, 3> names = {"q", "k", "v"};
            for (size_t i = 0; i < names.size(); ++i) {
                a.conv[i] = store.load_tensor_as_shape(source, attn + "." + names[i] + "_conv1d.weight",
                    assets::TensorStorageType::F32, {width, 1, c.conv_kernel_size},
                    core::TensorShape::from_dims({width, c.conv_kernel_size}));
            }
            a.output = binding::linear_from_source(store, source, attn + ".o_proj", native, c.hidden_size, width, false);
            a.a_log = store.load_f32_tensor(source, attn + ".A_log", {c.heads});
            a.dt_bias = store.load_f32_tensor(source, attn + ".dt_bias", {width});
            a.norm = binding::norm_weight_from_source(store, source, attn + ".o_norm", c.head_dim);
        }
        const bool routed = layer >= c.dense_layers;
        const auto dense = prefix + (routed ? ".mlp.shared_experts" : ".mlp");
        const auto middle = routed ? c.shared_intermediate_size : c.intermediate_size;
        w.mlp_gate_up = modules::PackedLinearWeightsBuilder({c.hidden_size, {
            {dense + ".gate_proj.weight", std::nullopt, middle},
            {dense + ".up_proj.weight", std::nullopt, middle}}, false, true}).build(store, source, native);
        w.mlp_down = binding::linear_from_source(store, source, dense + ".down_proj", native, c.hidden_size, middle, false);
        if (routed) {
            w.router = store.load_f32_tensor(source, prefix + ".mlp.gate.weight", {c.experts, c.hidden_size});
            auto bias = source.require_f32(prefix + ".mlp.gate.expert_bias", {c.experts});
            // The official BF16 loader casts this buffer before the F32 router uses it.
            core::round_f32_to_bf16_in_place(bias);
            w.router_bias = store.make_f32(core::TensorShape::from_dims({c.experts}), std::move(bias));
            if (disk_store) {
                w.disk_store = disk_store.get();
                w.disk_group = disk_store->add_group({"thinker/" + prefix + ".mlp.experts.gate_proj.weight",
                    "thinker/" + prefix + ".mlp.experts.up_proj.weight", "thinker/" + prefix + ".mlp.experts.down_proj.weight"},
                    execution.backend(), options.expert_cache_size);
                w.expert_gate = disk_store->tensor(w.disk_group, 0);
                w.expert_up = disk_store->tensor(w.disk_group, 1);
                w.expert_down = disk_store->tensor(w.disk_group, 2);
            } else {
                w.expert_gate = store.load_tensor(source, prefix + ".mlp.experts.gate_proj.weight", native,
                    {c.experts, c.expert_intermediate_size, c.hidden_size});
                w.expert_up = store.load_tensor(source, prefix + ".mlp.experts.up_proj.weight", native,
                    {c.experts, c.expert_intermediate_size, c.hidden_size});
                w.expert_down = store.load_tensor(source, prefix + ".mlp.experts.down_proj.weight", native,
                    {c.experts, c.hidden_size, c.expert_intermediate_size});
            }
        }
        layers.push_back(std::move(w));
    }
    store.upload();
    if (disk_store) {
        const auto stats = disk_store->statistics();
        debug::timing_log_scalar("ming3_audio.thinker.expert_cache_mb", stats.buffer_bytes / (1024.0 * 1024.0));
        debug::timing_log_scalar("ming3_audio.thinker.host_experts_mb", stats.host_buffer_bytes / (1024.0 * 1024.0));
    }
}

KdaOutput build_kda(core::ModuleBuildContext & ctx, const core::TensorValue & input,
                    const std::array<core::TensorValue, 3> & tails,
                    const core::TensorValue & state, const KdaWeights & w, const ThinkerConfig & c) {
    const int64_t steps = input.shape.dims[1];
    const int64_t width = c.heads * c.head_dim;
    KdaOutput result;
    const auto projected = modules::LinearModule({c.hidden_size, 5 * width + c.heads, false})
        .build(ctx, input, w.input_proj);
    std::array<ggml_tensor *, 3> qkv;
    for (size_t i = 0; i < qkv.size(); ++i) {
        auto part = modules::SliceModule({2, static_cast<int64_t>(i) * width, width}).build(ctx, projected);
        part = modules::Bf16RoundModule{}.build(ctx, part);
        part = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, part);
        part = core::ensure_backend_addressable_layout(ctx, part);
        const auto padded = modules::ConcatModule({2}).build(ctx, tails[i], part);
        auto * convolved = ggml_ssm_conv(ctx.ggml, padded.tensor, w.conv[i].tensor);
        auto conv = core::wrap_tensor(convolved, core::TensorShape::from_dims({1, steps, width}));
        conv = modules::SiluModule{}.build(ctx, conv);
        conv = modules::Bf16RoundModule{}.build(ctx, conv);
        qkv[i] = ggml_reshape_4d(ctx.ggml, conv.tensor, c.head_dim, c.heads, steps, 1);
        result.conv_tails[i] = core::ensure_backend_addressable_layout(ctx,
            modules::SliceModule({2, steps, c.conv_kernel_size - 1}).build(ctx, padded));
    }
    // FLA normalizes Q/K in F32 inside the recurrent kernel, without BF16 rounding.
    const float normalization_scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
    auto * query = ggml_scale(ctx.ggml, ggml_rms_norm(ctx.ggml, qkv[0], 1e-6f / c.head_dim), normalization_scale);
    auto * key = ggml_scale(ctx.ggml, ggml_rms_norm(ctx.ggml, qkv[1], 1e-6f / c.head_dim), normalization_scale);
    const auto decay = modules::Bf16RoundModule{}.build(ctx,
        modules::SliceModule({2, 3 * width, width}).build(ctx, projected));
    auto * g = ggml_add(ctx.ggml, decay.tensor, w.dt_bias.tensor);
    g = ggml_reshape_4d(ctx.ggml, g, c.head_dim, c.heads, steps, 1);
    auto * a = ggml_reshape_4d(ctx.ggml, ggml_exp(ctx.ggml, w.a_log.tensor), 1, c.heads, 1, 1);
    g = ggml_scale(ctx.ggml, ggml_sigmoid(ctx.ggml, ggml_mul(ctx.ggml, g, a)), c.kda_lower_bound);
    const auto beta = modules::Bf16RoundModule{}.build(ctx,
        modules::SliceModule({2, 5 * width, c.heads}).build(ctx, projected));
    auto * b = ggml_reshape_4d(ctx.ggml, ggml_sigmoid(ctx.ggml, beta.tensor), 1, c.heads, steps, 1);
    auto * delta = ggml_gated_delta_net(ctx.ggml, query, key, qkv[2], g, b, state.tensor);
    auto * output = ggml_view_2d(ctx.ggml, delta, width, steps, width * sizeof(float), 0);
    auto hidden = core::wrap_tensor(ggml_reshape_2d(ctx.ggml, output, c.head_dim, steps * c.heads),
        core::TensorShape::from_dims({steps * c.heads, c.head_dim}));
    hidden = modules::Bf16RoundModule{}.build(ctx, hidden);
    hidden = modules::RMSNormModule({c.head_dim, c.rms_norm_eps, true, false}).build(ctx, hidden, w.norm);
    const auto gate = modules::Bf16RoundModule{}.build(ctx,
        modules::SliceModule({2, 4 * width, width}).build(ctx, projected));
    auto * gated = ggml_mul(ctx.ggml, hidden.tensor,
        ggml_reshape_2d(ctx.ggml, ggml_sigmoid(ctx.ggml, gate.tensor), c.head_dim, steps * c.heads));
    hidden = modules::Bf16RoundModule{}.build(ctx, core::wrap_tensor(gated, hidden.shape));
    hidden = core::reshape_tensor(ctx, hidden, core::TensorShape::from_dims({1, steps, width}));
    result.hidden = modules::Bf16RoundModule{}.build(ctx,
        modules::LinearModule({width, c.hidden_size, false}).build(ctx, hidden, w.output));
    result.recurrent_state = core::wrap_tensor(
        ggml_view_3d(ctx.ggml, delta, state.tensor->ne[0], state.tensor->ne[1], state.tensor->ne[2],
            state.tensor->nb[1], state.tensor->nb[2], width * steps * sizeof(float)), state.shape);
    return result;
}

core::TensorValue build_mla(core::ModuleBuildContext & ctx, const core::TensorValue & input,
                            const core::TensorValue & positions, const core::TensorValue & attention_mask,
                            const core::TensorValue & key_cache, const core::TensorValue & value_cache,
                            const MlaWeights & w, const ThinkerConfig & c) {
    const auto steps = input.shape.dims[1];
    const auto capacity = attention_mask.shape.dims[3];
    const auto query_dim = c.qk_nope_dim + c.qk_rope_dim;
    modules::NormConfig query_norm{c.query_rank, c.rms_norm_eps, true, false};
    query_norm.normalized_round_type = GGML_TYPE_BF16;
    auto query = modules::Bf16RoundModule{}.build(ctx,
        modules::LinearModule({c.hidden_size, c.query_rank, false}).build(ctx, input, w.query_down));
    query = modules::Bf16RoundModule{}.build(ctx, modules::RMSNormModule(query_norm).build(ctx, query, w.query_norm));
    query = modules::Bf16RoundModule{}.build(ctx,
        modules::LinearModule({c.query_rank, c.heads * query_dim, false}).build(ctx, query, w.query_up));
    query = core::reshape_tensor(ctx, query, core::TensorShape::from_dims({1, steps, c.heads, query_dim}));
    auto q_pass = modules::SliceModule({3, 0, c.qk_nope_dim}).build(ctx, query);
    auto q_rot = modules::SliceModule({3, c.qk_nope_dim, c.qk_rope_dim}).build(ctx, query);
    q_rot = core::ensure_backend_addressable_layout(ctx, q_rot);
    q_rot = modules::Bf16RoundModule{}.build(ctx,
        modules::RoPEModule({c.qk_rope_dim, GGML_ROPE_TYPE_NORMAL, c.rope_theta}).build(ctx, q_rot, positions));
    query = modules::ConcatModule({3}).build(ctx, q_pass, q_rot);

    auto compressed = modules::Bf16RoundModule{}.build(ctx,
        modules::LinearModule({c.hidden_size, c.kv_rank + c.qk_rope_dim, false}).build(ctx, input, w.kv_down));
    auto kv = modules::SliceModule({2, 0, c.kv_rank}).build(ctx, compressed);
    auto k_rot = modules::SliceModule({2, c.kv_rank, c.qk_rope_dim}).build(ctx, compressed);
    modules::NormConfig kv_norm{c.kv_rank, c.rms_norm_eps, true, false};
    kv_norm.normalized_round_type = GGML_TYPE_BF16;
    kv = modules::Bf16RoundModule{}.build(ctx, modules::RMSNormModule(kv_norm).build(ctx, kv, w.kv_norm));
    kv = modules::Bf16RoundModule{}.build(ctx,
        modules::LinearModule({c.kv_rank, c.heads * (c.qk_nope_dim + c.value_dim), false}).build(ctx, kv, w.kv_up));
    kv = core::reshape_tensor(ctx, kv, core::TensorShape::from_dims({1, steps, c.heads, c.qk_nope_dim + c.value_dim}));
    auto key = modules::SliceModule({3, 0, c.qk_nope_dim}).build(ctx, kv);
    auto value = core::ensure_backend_addressable_layout(ctx,
        modules::SliceModule({3, c.qk_nope_dim, c.value_dim}).build(ctx, kv));
    k_rot = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, k_rot),
        core::TensorShape::from_dims({1, steps, 1, c.qk_rope_dim}));
    k_rot = modules::Bf16RoundModule{}.build(ctx,
        modules::RoPEModule({c.qk_rope_dim, GGML_ROPE_TYPE_NORMAL, c.rope_theta}).build(ctx, k_rot, positions));
    k_rot = modules::RepeatModule({core::TensorShape::from_dims({1, steps, c.heads, c.qk_rope_dim})}).build(ctx, k_rot);
    key = modules::ConcatModule({3}).build(ctx, key, k_rot);
    key = core::ensure_backend_addressable_layout(ctx, key);
    auto * keys = ggml_set_rows(ctx.ggml, key_cache.tensor,
        ggml_reshape_2d(ctx.ggml, key.tensor, c.heads * query_dim, steps), positions.tensor);
    auto * values = ggml_set_rows(ctx.ggml, value_cache.tensor,
        ggml_reshape_2d(ctx.ggml, value.tensor, c.heads * c.value_dim, steps), positions.tensor);
    if (capacity < key_cache.tensor->ne[1]) {
        keys = ggml_view_2d(ctx.ggml, keys, keys->ne[0], capacity, keys->nb[1], 0);
        values = ggml_view_2d(ctx.ggml, values, values->ne[0], capacity, values->nb[1], 0);
    }
    auto cached_key = core::wrap_tensor(ggml_reshape_4d(ctx.ggml, keys, query_dim, c.heads, capacity, 1),
        core::TensorShape::from_dims({1, capacity, c.heads, query_dim}), key_cache.type);
    auto cached_value = core::wrap_tensor(ggml_reshape_4d(ctx.ggml, values, c.value_dim, c.heads, capacity, 1),
        core::TensorShape::from_dims({1, capacity, c.heads, c.value_dim}), value_cache.type);
    auto query_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, query);
    auto key_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, cached_key);
    auto value_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, cached_value);
    auto attended = modules::ScaledDotProductAttentionModule({query_dim}).build(
        ctx, query_heads, key_heads, value_heads, attention_mask);
    attended = modules::Bf16RoundModule{}.build(ctx, attended);
    auto gate = modules::Bf16RoundModule{}.build(ctx,
        modules::LinearModule({c.hidden_size, c.heads, false}).build(ctx, input, w.gate));
    gate = modules::Bf16RoundModule{}.build(ctx, modules::SigmoidModule{}.build(ctx, gate));
    gate = core::reshape_tensor(ctx, gate, core::TensorShape::from_dims({1, steps, c.heads, 1}));
    gate = modules::RepeatModule({attended.shape}).build(ctx, gate);
    attended = modules::Bf16RoundModule{}.build(ctx, modules::MulModule{}.build(ctx, attended, gate));
    attended = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, attended),
        core::TensorShape::from_dims({1, steps, c.heads * c.value_dim}));
    return modules::Bf16RoundModule{}.build(ctx,
        modules::LinearModule({c.heads * c.value_dim, c.hidden_size, false}).build(ctx, attended, w.output));
}

namespace {

core::TensorValue build_shared_mlp(core::ModuleBuildContext & ctx, const core::TensorValue & input,
                                  const ThinkerLayerWeights & w, const ThinkerConfig & c, bool routed) {
    const int64_t middle = routed ? c.shared_intermediate_size : c.intermediate_size;
    const auto projected = modules::LinearModule({c.hidden_size, 2 * middle, false}).build(ctx, input, w.mlp_gate_up);
    auto gate = modules::Bf16RoundModule{}.build(ctx,
        modules::SliceModule({2, 0, middle}).build(ctx, projected));
    const auto up = modules::Bf16RoundModule{}.build(ctx,
        modules::SliceModule({2, middle, middle}).build(ctx, projected));
    gate = modules::Bf16RoundModule{}.build(ctx, modules::SiluModule{}.build(ctx, gate));
    gate = modules::Bf16RoundModule{}.build(ctx, modules::MulModule{}.build(ctx, gate, up));
    return modules::Bf16RoundModule{}.build(ctx,
        modules::LinearModule({middle, c.hidden_size, false}).build(ctx, gate, w.mlp_down));
}

struct MoeRouting {
    ggml_tensor * ids;
    ggml_tensor * scales;
};

MoeRouting build_routing(core::ModuleBuildContext & ctx, ggml_tensor * x,
                         const ThinkerLayerWeights & w, const ThinkerConfig & c,
                         const core::TensorValue & group_identity) {
    const auto steps = x->ne[1];
    auto * router_logits = ggml_mul_mat(ctx.ggml, w.router.tensor, x);
    ggml_mul_mat_set_prec(router_logits, GGML_PREC_F32);
    auto * scores = ggml_sigmoid(ctx.ggml, router_logits);
    auto * routing = ggml_add(ctx.ggml, scores, w.router_bias.tensor);
    const int64_t per_group = c.experts / c.expert_groups;
    auto * groups = ggml_reshape_2d(ctx.ggml, routing, per_group, c.expert_groups * steps);
    auto * top2 = ggml_argsort_top_k(ctx.ggml, groups, 2);
    auto * best = ggml_get_rows(ctx.ggml,
        ggml_reshape_3d(ctx.ggml, groups, 1, per_group, c.expert_groups * steps), top2);
    auto * group_scores = ggml_reshape_2d(ctx.ggml,
        ggml_sum_rows(ctx.ggml, ggml_reshape_2d(ctx.ggml, best, 2, c.expert_groups * steps)), c.expert_groups, steps);
    auto * selected_groups = ggml_argsort_top_k(ctx.ggml, group_scores, c.selected_groups);
    auto * group_rows = ggml_get_rows(ctx.ggml, group_identity.tensor,
        ggml_reshape_1d(ctx.ggml, ggml_cont(ctx.ggml, selected_groups), c.selected_groups * steps));
    group_rows = ggml_reshape_3d(ctx.ggml, group_rows, c.expert_groups, c.selected_groups, steps);
    auto * mask = ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, group_rows, 1, 0, 2, 3));
    mask = ggml_reshape_3d(ctx.ggml, ggml_sum_rows(ctx.ggml, mask), 1, c.expert_groups, steps);
    mask = ggml_repeat(ctx.ggml, mask, ggml_reshape_3d(ctx.ggml, routing, per_group, c.expert_groups, steps));
    mask = ggml_scale_bias(ctx.ggml, ggml_reshape_2d(ctx.ggml, mask, c.experts, steps), 1e30f, -1e30f);
    auto * ids = ggml_argsort_top_k(ctx.ggml, ggml_add(ctx.ggml, routing, mask), c.experts_per_token);
    auto * selected = ggml_get_rows(ctx.ggml, ggml_reshape_3d(ctx.ggml, scores, 1, c.experts, steps), ids);
    selected = ggml_reshape_2d(ctx.ggml, selected, c.experts_per_token, steps);
    if (c.experts_per_token > 1) {
        auto * denominator = ggml_scale_bias(ctx.ggml, ggml_sum_rows(ctx.ggml, selected), 1.0f, 1e-20f);
        selected = ggml_div(ctx.ggml, selected, denominator);
    }
    selected = ggml_scale(ctx.ggml, selected, c.routed_scaling_factor);
    return {ids, selected};
}

ggml_tensor * build_expert_mlp(core::ModuleBuildContext & ctx, ggml_tensor * input,
                              ggml_tensor * ids, const ThinkerLayerWeights & w) {
    auto * expert_gate = ggml_mul_mat_id(ctx.ggml, w.expert_gate.tensor, input, ids);
    auto * expert_up = ggml_mul_mat_id(ctx.ggml, w.expert_up.tensor, input, ids);
    expert_gate = ggml_round_bf16(ctx.ggml, expert_gate);
    expert_up = ggml_round_bf16(ctx.ggml, expert_up);
    expert_gate = ggml_round_bf16(ctx.ggml, ggml_silu(ctx.ggml, expert_gate));
    auto * activated = ggml_mul(ctx.ggml, expert_gate, expert_up);
    activated = ggml_round_bf16(ctx.ggml, activated);
    auto * experts = ggml_mul_mat_id(ctx.ggml, w.expert_down.tensor, activated, ids);
    return ggml_round_bf16(ctx.ggml, experts);
}

}  // namespace

core::TensorValue build_thinker_mlp(core::ModuleBuildContext & ctx, const core::TensorValue & input,
                                   const ThinkerLayerWeights & w, const ThinkerConfig & c,
                                   const core::TensorValue & group_identity, bool routed) {
    const auto steps = input.shape.dims[1];
    auto shared = build_shared_mlp(ctx, input, w, c, routed);
    if (!routed) return shared;
    auto * x = ggml_reshape_2d(ctx.ggml, input.tensor, c.hidden_size, steps);
    auto routing = build_routing(ctx, x, w, c, group_identity);
    if (w.disk_store) routing.ids = w.disk_store->route(ctx, ggml_cont(ctx.ggml, routing.ids), w.disk_group);
    auto * expanded = ggml_reshape_3d(ctx.ggml, x, c.hidden_size, 1, steps);
    auto * experts = build_expert_mlp(ctx, expanded, routing.ids, w);
    experts = ggml_mul(ctx.ggml, experts, ggml_reshape_3d(ctx.ggml, routing.scales, 1, c.experts_per_token, steps));
    experts = ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, experts, 1, 0, 2, 3));
    experts = ggml_reshape_3d(ctx.ggml, ggml_sum_rows(ctx.ggml, experts), c.hidden_size, steps, 1);
    auto routed_output = modules::Bf16RoundModule{}.build(ctx, core::wrap_tensor(experts, shared.shape));
    return modules::Bf16RoundModule{}.build(ctx, modules::AddModule{}.build(ctx, routed_output, shared));
}

struct Thinker::Impl {
    struct Graph {
        explicit Graph(ggml_backend_t backend) : backend(backend) {}
        ~Graph() {
            if (scheduler) {
                if (cleanup_graph) {
                    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                        ggml_graph_set_n_nodes(cleanup_graph, 0);
                        ggml_graph_add_node(cleanup_graph, ggml_graph_node(graph, i));
                        core::release_backend_graph_resources(backend, cleanup_graph, true);
                    }
                }
                ggml_backend_sched_free(scheduler);
            }
            core::release_backend_graph_resources(backend, graph, true);
            plan.reset();
            if (allocator) ggml_gallocr_free(allocator);
            if (cleanup_context) ggml_free(cleanup_context);
            if (context) ggml_free(context);
        }
        ggml_backend_t backend;
        ggml_context * context = nullptr;
        ggml_gallocr_t allocator = nullptr;
        ggml_backend_sched_t scheduler = nullptr;
        ggml_context * cleanup_context = nullptr;
        ggml_cgraph * cleanup_graph = nullptr;
        ggml_cgraph * graph = nullptr;
        core::HostGraphPlan plan;
        core::TensorValue ids;
        core::TensorValue audio;
        core::TensorValue audio_indices;
        core::TensorValue positions;
        core::TensorValue mask;
        ggml_tensor * logits = nullptr;
        int64_t steps = 0;
        int64_t audio_steps = 0;
    };

    struct State {
        std::array<core::TensorValue, 3> tails;
        core::TensorValue recurrent;
        core::TensorValue key;
        core::TensorValue value;
    };

    class OffloadPrefill {
    public:
        OffloadPrefill(core::ExecutionContext & execution, const ThinkerWeights & weights,
                       const ThinkerConfig & config, const std::vector<State> & states,
                       int64_t steps, int64_t capacity, ggml_gallocr_t allocator, bool reserve_workspace)
            : execution_(execution), weights_(weights), config_(config), steps_(steps), capacity_(capacity) {
            constexpr size_t nodes = 1024, small_nodes = 128;
            const auto large_bytes = ggml_graph_overhead_custom(nodes, false) + nodes * ggml_tensor_overhead();
            const auto small_bytes = ggml_graph_overhead_custom(small_nodes, false) + small_nodes * ggml_tensor_overhead();
            context_.reset(ggml_init({static_cast<size_t>(config.layers) * large_bytes +
                static_cast<size_t>(config.layers + 3) * small_bytes, nullptr, true}));
            inputs_.reset(ggml_init({64 * ggml_tensor_overhead(), nullptr, true}));
            if (!context_ || !inputs_) throw std::runtime_error("Ming3 Audio failed to allocate offload prefill metadata");
            const auto h = config.hidden_size, k = config.experts_per_token;
            const auto slots = weights.layers.at(config.dense_layers).expert_gate.tensor->ne[2];
            rows_ = slots * expert_rows;
            auto * persistent = inputs_.get();
            ids_ = ggml_new_tensor_1d(persistent, GGML_TYPE_I32, steps);
            audio_ = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, h, steps);
            audio_indices_ = ggml_new_tensor_1d(persistent, GGML_TYPE_I32, steps);
            positions_ = ggml_new_tensor_1d(persistent, GGML_TYPE_I32, steps);
            mask_ = ggml_new_tensor_4d(persistent, GGML_TYPE_F32, capacity, steps, 1, 1);
            x_ = ggml_new_tensor_3d(persistent, GGML_TYPE_F32, h, steps, 1);
            hidden_ = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, h, steps);
            residual_ = ggml_new_tensor_3d(persistent, GGML_TYPE_F32, h, steps, 1);
            shared_ = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, h, steps);
            scales_ = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, k, steps);
            routing_ids_ = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, k, steps);
            expert_output_ = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, h, k * steps + rows_);
            source_rows_ = ggml_new_tensor_1d(persistent, GGML_TYPE_I32, rows_);
            target_rows_ = ggml_new_tensor_1d(persistent, GGML_TYPE_I32, rows_);
            slot_ids_ = ggml_new_tensor_2d(persistent, GGML_TYPE_I32, 1, rows_);
            for (auto * tensor : {ids_, audio_, audio_indices_, positions_, mask_, x_, hidden_, residual_,
                                  shared_, scales_, routing_ids_, expert_output_, source_rows_, target_rows_, slot_ids_})
                ggml_set_input(tensor);
            buffer_.reset(ggml_backend_alloc_ctx_tensors(persistent, execution.backend()));
            if (!buffer_) throw std::runtime_error("Ming3 Audio failed to allocate offload prefill inputs");
            ggml_backend_buffer_clear(buffer_.get(), 0);
            std::vector<int32_t> slots_by_row(static_cast<size_t>(rows_));
            for (int64_t slot = 0; slot < slots; ++slot)
                std::fill_n(slots_by_row.begin() + slot * expert_rows, expert_rows, static_cast<int32_t>(slot));
            ggml_backend_tensor_set(slot_ids_, slots_by_row.data(), 0, slots_by_row.size() * sizeof(int32_t));

            core::ModuleBuildContext ctx{context_.get(), "ming3_audio.offload_prefill", execution.backend_type()};
            ggml_cgraph * largest = nullptr;
            size_t max_bytes = 0;
            auto add_graph = [&](ggml_cgraph * graph) {
                core::validate_backend_graph_supported(execution.backend(), graph, "ming3_audio.offload_prefill");
                size_t bytes = 0;
                ggml_gallocr_reserve_n_size(allocator, graph, nullptr, nullptr, &bytes);
                if (!largest || bytes > max_bytes) { largest = graph; max_bytes = bytes; }
                graphs_.push_back(graph);
            };
            const auto shape = core::TensorShape::from_dims({1, steps, h});
            auto x = core::wrap_tensor(x_, shape);
            initial_ = ggml_new_graph_custom(ctx.ggml, small_nodes, false);
            auto * embeddings = ggml_get_rows(ctx.ggml, weights.embedding.tensor, ids_);
            // Non-audio positions write to distinct discarded rows, keeping one graph for every text/audio layout.
            auto * padded = ggml_pad(ctx.ggml, embeddings, 0, static_cast<int>(steps), 0, 0);
            auto * replaced = ggml_set_rows(ctx.ggml, padded, audio_, audio_indices_);
            auto embedded = core::wrap_tensor(ggml_reshape_3d(ctx.ggml,
                ggml_view_2d(ctx.ggml, replaced, h, steps, replaced->nb[1], 0), h, steps, 1), shape);
            embedded = modules::Bf16RoundModule{}.build(ctx, embedded);
            ggml_build_forward_expand(initial_, ggml_cpy(ctx.ggml, embedded.tensor, x_));
            add_graph(initial_);
            modules::NormConfig norm{h, config.rms_norm_eps, true, false};
            norm.normalized_round_type = GGML_TYPE_BF16;
            for (int64_t layer = 0; layer < config.layers; ++layer) {
                const auto & w = weights.layers[layer];
                const auto & state = states[layer];
                auto * graph = ggml_new_graph_custom(ctx.ggml, nodes, false);
                auto normalized = modules::Bf16RoundModule{}.build(ctx,
                    modules::RMSNormModule(norm).build(ctx, x, w.input_norm));
                core::TensorValue attention;
                if (state.key.valid()) {
                    attention = build_mla(ctx, normalized,
                        core::wrap_tensor(positions_, core::TensorShape::from_dims({steps}), GGML_TYPE_I32),
                        core::wrap_tensor(mask_, core::TensorShape::from_dims({1, 1, steps, capacity})),
                        state.key, state.value, w.mla, config);
                } else {
                    const auto kda = build_kda(ctx, normalized, state.tails, state.recurrent, w.kda, config);
                    for (size_t i = 0; i < state.tails.size(); ++i)
                        ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, kda.conv_tails[i].tensor, state.tails[i].tensor));
                    ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, kda.recurrent_state.tensor, state.recurrent.tensor));
                    attention = kda.hidden;
                }
                auto residual = modules::Bf16RoundModule{}.build(ctx, modules::AddModule{}.build(ctx, x, attention));
                normalized = modules::Bf16RoundModule{}.build(ctx,
                    modules::RMSNormModule(norm).build(ctx, residual, w.post_norm));
                const bool routed = layer >= config.dense_layers;
                auto shared = build_shared_mlp(ctx, normalized, w, config, routed);
                if (!routed) {
                    auto output = modules::Bf16RoundModule{}.build(ctx,
                        modules::AddModule{}.build(ctx, residual, shared));
                    ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, output.tensor, x_));
                } else {
                    const auto routing = build_routing(ctx,
                        ggml_reshape_2d(ctx.ggml, normalized.tensor, h, steps), w, config, weights.group_identity);
                    for (const auto & copy : std::vector<std::pair<ggml_tensor *, ggml_tensor *>>{
                             {residual.tensor, residual_}, {normalized.tensor, hidden_}, {shared.tensor, shared_},
                             {routing.scales, scales_}, {ggml_cast(ctx.ggml, routing.ids, GGML_TYPE_F32), routing_ids_}}) {
                        ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml,
                            ggml_reshape_2d(ctx.ggml, copy.first, copy.first->ne[0], steps),
                            ggml_reshape_2d(ctx.ggml, copy.second, copy.second->ne[0], steps)));
                    }
                }
                add_graph(graph);
                attention_graphs_.push_back(graph);
                ggml_cgraph * expert_graph = nullptr;
                if (routed) {
                    expert_graph = ggml_new_graph_custom(ctx.ggml, small_nodes, false);
                    auto * input = ggml_reshape_3d(ctx.ggml, ggml_get_rows(ctx.ggml, hidden_, source_rows_), h, 1, rows_);
                    auto * output = build_expert_mlp(ctx, input, slot_ids_, w);
                    ggml_build_forward_expand(expert_graph, ggml_set_rows(ctx.ggml, expert_output_,
                        ggml_reshape_2d(ctx.ggml, output, h, rows_), target_rows_));
                    add_graph(expert_graph);
                }
                expert_graphs_.push_back(expert_graph);
            }
            finish_ = ggml_new_graph_custom(ctx.ggml, small_nodes, false);
            auto * experts = ggml_view_3d(ctx.ggml, expert_output_, h, k, steps,
                h * sizeof(float), h * k * sizeof(float), 0);
            auto * scaled = ggml_mul(ctx.ggml, experts, ggml_reshape_3d(ctx.ggml, scales_, 1, k, steps));
            auto * reduced = ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, scaled, 1, 0, 2, 3));
            auto routed = modules::Bf16RoundModule{}.build(ctx, core::wrap_tensor(
                ggml_reshape_3d(ctx.ggml, ggml_sum_rows(ctx.ggml, reduced), h, steps, 1), shape));
            auto combined = modules::Bf16RoundModule{}.build(ctx, modules::AddModule{}.build(
                ctx, routed, core::wrap_tensor(ggml_reshape_3d(ctx.ggml, shared_, h, steps, 1), shape)));
            combined = modules::Bf16RoundModule{}.build(ctx,
                modules::AddModule{}.build(ctx, core::wrap_tensor(residual_, shape), combined));
            ggml_build_forward_expand(finish_, ggml_cpy(ctx.ggml, combined.tensor, x_));
            add_graph(finish_);
            final_ = ggml_new_graph_custom(ctx.ggml, small_nodes, false);
            auto last = modules::SliceModule({1, steps - 1, 1}).build(ctx, x);
            last = modules::Bf16RoundModule{}.build(ctx, modules::RMSNormModule(norm).build(ctx, last, weights.norm));
            logits_ = ggml_round_bf16(ctx.ggml, ggml_mul_mat(ctx.ggml, weights.head.tensor, last.tensor));
            ggml_set_output(logits_);
            ggml_build_forward_expand(final_, logits_);
            add_graph(final_);
            if (reserve_workspace) {
                if (!ggml_gallocr_reserve(allocator, largest))
                    throw std::runtime_error("Ming3 Audio failed to reserve offload prefill workspace");
            } else if (max_bytes > ggml_gallocr_get_buffer_size(allocator, 0)) {
                throw std::runtime_error("Ming3 Audio partial prefill exceeds the reserved full-batch workspace");
            }
            for (auto * graph : graphs_) {
                if (!ggml_gallocr_reserve(allocator, graph) || !ggml_gallocr_alloc_graph(allocator, graph))
                    throw std::runtime_error("Ming3 Audio failed to allocate offload prefill graph");
                core::prepare_host_graph_plan(execution, graph, plans_[graph]);
            }
            debug::timing_log_scalar("ming3_audio.offload_prefill.workspace_mb",
                ggml_gallocr_get_buffer_size(allocator, 0) / (1024.0 * 1024.0));
        }

        ~OffloadPrefill() {
            for (auto * graph : graphs_) core::release_backend_graph_resources(execution_.backend(), graph, true);
        }

        int64_t steps() const { return steps_; }

        std::vector<float> run(const std::vector<int32_t> & ids, const std::vector<float> & audio,
                               const std::vector<int32_t> & audio_indices, int64_t offset) {
            const auto h = config_.hidden_size, k = config_.experts_per_token;
            std::vector<int32_t> positions(static_cast<size_t>(steps_)), targets(static_cast<size_t>(steps_));
            std::vector<float> projected(static_cast<size_t>(steps_ * h), 0.0f);
            std::vector<float> mask(static_cast<size_t>(steps_ * capacity_), -std::numeric_limits<float>::infinity());
            for (int64_t row = 0; row < steps_; ++row) {
                positions[row] = static_cast<int32_t>(offset + row);
                targets[row] = static_cast<int32_t>(steps_ + row);
                std::fill_n(mask.begin() + row * capacity_, offset + row + 1, 0.0f);
            }
            for (size_t i = 0; i < audio_indices.size(); ++i) {
                const auto row = audio_indices[i];
                targets[row] = row;
                std::copy_n(audio.begin() + i * h, h, projected.begin() + row * h);
            }
            core::round_f32_to_bf16_in_place(projected);
            ggml_backend_tensor_set(ids_, ids.data(), 0, ids.size() * sizeof(int32_t));
            ggml_backend_tensor_set(audio_, projected.data(), 0, projected.size() * sizeof(float));
            ggml_backend_tensor_set(audio_indices_, targets.data(), 0, targets.size() * sizeof(int32_t));
            ggml_backend_tensor_set(positions_, positions.data(), 0, positions.size() * sizeof(int32_t));
            ggml_backend_tensor_set(mask_, mask.data(), 0, mask.size() * sizeof(float));
            compute(initial_);
            std::vector<float> routing(static_cast<size_t>(k * steps_));
            std::vector<std::vector<int32_t>> by_expert(static_cast<size_t>(config_.experts));
            std::vector<int32_t> source(static_cast<size_t>(rows_)), target(static_cast<size_t>(rows_));
            const auto slots = rows_ / expert_rows;
            for (int64_t layer = 0; layer < config_.layers; ++layer) {
                compute(attention_graphs_[layer]);
                if (layer < config_.dense_layers) continue;
                ggml_backend_tensor_get(routing_ids_, routing.data(), 0, routing.size() * sizeof(float));
                for (auto & rows : by_expert) rows.clear();
                for (size_t pair = 0; pair < routing.size(); ++pair)
                    by_expert.at(static_cast<size_t>(routing[pair])).push_back(static_cast<int32_t>(pair));
                std::vector<int32_t> experts;
                for (int32_t expert = 0; expert < config_.experts; ++expert)
                    if (!by_expert[expert].empty()) experts.push_back(expert);
                std::stable_sort(experts.begin(), experts.end(), [&](int32_t left, int32_t right) {
                    return by_expert[left].size() > by_expert[right].size();
                });
                const auto & w = weights_.layers[layer];
                for (size_t start = 0; start < experts.size(); start += static_cast<size_t>(slots)) {
                    const std::vector<int32_t> group(experts.begin() + start,
                        experts.begin() + std::min(experts.size(), start + static_cast<size_t>(slots)));
                    const auto mapped = w.disk_store->stage(w.disk_group, group);
                    size_t maximum = 0;
                    for (const auto expert : group) maximum = std::max(maximum, by_expert[expert].size());
                    for (size_t begin = 0; begin < maximum; begin += expert_rows) {
                        std::fill(source.begin(), source.end(), 0);
                        for (int64_t row = 0; row < rows_; ++row) target[row] = static_cast<int32_t>(steps_ * k + row);
                        for (size_t e = 0; e < group.size(); ++e) {
                            const auto & pairs = by_expert[group[e]];
                            for (size_t row = begin; row < std::min(pairs.size(), begin + expert_rows); ++row) {
                                const auto slot_row = mapped[e] * expert_rows + row - begin;
                                source[slot_row] = static_cast<int32_t>(pairs[row] / k);
                                target[slot_row] = pairs[row];
                            }
                        }
                        ggml_backend_tensor_set(source_rows_, source.data(), 0, source.size() * sizeof(int32_t));
                        ggml_backend_tensor_set(target_rows_, target.data(), 0, target.size() * sizeof(int32_t));
                        compute(expert_graphs_[layer]);
                    }
                }
                compute(finish_);
            }
            compute(final_);
            return core::read_tensor_f32(logits_);
        }

    private:
        void compute(ggml_cgraph * graph) {
            if (core::compute_graph(execution_, graph, plans_.at(graph), "ming3_audio.offload_prefill") != GGML_STATUS_SUCCESS)
                throw std::runtime_error("Ming3 Audio offload prefill execution failed");
        }
        static constexpr size_t expert_rows = 32;
        core::ExecutionContext & execution_;
        const ThinkerWeights & weights_;
        ThinkerConfig config_;
        int64_t steps_, capacity_, rows_;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context_{nullptr, ggml_free}, inputs_{nullptr, ggml_free};
        std::unique_ptr<std::remove_pointer_t<ggml_backend_buffer_t>, decltype(&ggml_backend_buffer_free)>
            buffer_{nullptr, ggml_backend_buffer_free};
        std::vector<ggml_cgraph *> graphs_, attention_graphs_, expert_graphs_;
        std::map<ggml_cgraph *, core::HostGraphPlan> plans_;
        ggml_cgraph * initial_ = nullptr, * finish_ = nullptr, * final_ = nullptr;
        ggml_tensor * ids_ = nullptr, * audio_ = nullptr, * audio_indices_ = nullptr, * positions_ = nullptr, * mask_ = nullptr;
        ggml_tensor * x_ = nullptr, * hidden_ = nullptr, * residual_ = nullptr, * shared_ = nullptr, * scales_ = nullptr;
        ggml_tensor * routing_ids_ = nullptr, * expert_output_ = nullptr, * source_rows_ = nullptr, * target_rows_ = nullptr;
        ggml_tensor * slot_ids_ = nullptr, * logits_ = nullptr;
    };

    Impl(core::ExecutionContext & execution, std::shared_ptr<const assets::TensorSource> source,
         ThinkerConfig config, int64_t initial_capacity, ThinkerOptions options)
        : execution(execution), source(std::move(source)), config(config), options(options),
          weights(execution, *this->source, config, options), state_context(nullptr, ggml_free) {
        if (weights.disk_store) {
            core::BackendConfig host;
            host.threads = execution.config().threads;
            routing_execution = std::make_unique<core::ExecutionContext>(host);
        }
        reserve(initial_capacity);
    }

    void reserve(int64_t requested, bool preserve = false) {
        if (execution.backend_type() == core::BackendType::Cuda) requested = ((requested + 255) / 256) * 256;
        if (requested == capacity || (!options.disk_experts && requested < capacity)) return;
        const auto previous_steps = valid_steps;
        auto previous_context = std::move(state_context);
        auto previous_states = std::move(states);
        std::unique_ptr<std::remove_pointer_t<ggml_backend_buffer_t>, decltype(&ggml_backend_buffer_free)>
            previous_buffer(state_buffer, ggml_backend_buffer_free);
        state_buffer = nullptr;
        release_workspace();
        if (!preserve) {
            previous_states.clear();
            previous_buffer.reset();
            previous_context.reset();
        }
        capacity = requested;
        state_context.reset(ggml_init({1024 * 1024, nullptr, true}));
        if (!state_context) throw std::runtime_error("Ming3 Audio failed to allocate state metadata");
        core::ModuleBuildContext ctx{state_context.get(), "ming3_audio.thinker.state", execution.backend_type()};
        const auto & c = config;
        for (int64_t layer = 0; layer < c.layers; ++layer) {
            State state;
            const bool mla = (layer + 1) % c.attention_interval == 0 ||
                layer >= c.layers / c.attention_interval * c.attention_interval;
            if (mla) {
                state.key = core::make_tensor(ctx, GGML_TYPE_F16,
                    core::TensorShape::from_dims({capacity, c.heads * (c.qk_nope_dim + c.qk_rope_dim)}));
                state.value = core::make_tensor(ctx, GGML_TYPE_F16,
                    core::TensorShape::from_dims({capacity, c.heads * c.value_dim}));
            } else {
                for (auto & tail : state.tails) tail = core::make_tensor(ctx, GGML_TYPE_F32,
                    core::TensorShape::from_dims({1, c.heads * c.head_dim, c.conv_kernel_size - 1}));
                state.recurrent = core::make_tensor(ctx, GGML_TYPE_F32,
                    core::TensorShape::from_dims({1, 1, c.heads * c.head_dim * c.head_dim}));
            }
            states.push_back(state);
        }
        state_buffer = ggml_backend_alloc_ctx_tensors(state_context.get(), execution.backend());
        if (!state_buffer) throw std::runtime_error("Ming3 Audio failed to allocate Thinker state");
        debug::timing_log_scalar("ming3_audio.thinker.state_buffer_mb",
            ggml_backend_buffer_get_size(state_buffer) / (1024.0 * 1024.0));
        debug::trace_log_scalar("ming3_audio.thinker.cache_steps", capacity);
        reset();
        if (preserve && previous_steps > 0) {
            for (size_t i = 0; i < states.size(); ++i) {
                const auto & old = previous_states[i];
                const auto & current = states[i];
                if (old.key.valid()) {
                    for (const auto & pair : {std::pair{old.key, current.key}, std::pair{old.value, current.value}}) {
                        // Same-precision device copies of the live prefix, as in TransformerKVCache.
                        auto source = *pair.first.tensor;
                        auto destination = *pair.second.tensor;
                        source.ne[1] = destination.ne[1] = previous_steps;
                        std::copy(source.nb, source.nb + GGML_MAX_DIMS, destination.nb);
                        ggml_backend_tensor_copy(&source, &destination);
                    }
                } else {
                    for (size_t j = 0; j < old.tails.size(); ++j) {
                        ggml_backend_tensor_copy(old.tails[j].tensor, current.tails[j].tensor);
                    }
                    ggml_backend_tensor_copy(old.recurrent.tensor, current.recurrent.tensor);
                }
            }
            valid_steps = previous_steps;
        }
    }

    void release_workspace() {
        offload_tail.reset();
        offload_prefill.reset();
        offload_allocator.reset();
        prefill_graph.reset();
        step_graphs.clear();
        states.clear();
        if (state_buffer) ggml_backend_buffer_free(state_buffer);
        state_buffer = nullptr;
        state_context.reset();
        capacity = 0;
        valid_steps = 0;
    }

    ~Impl() {
        release_workspace();
    }

    void reset() {
        if (state_buffer) ggml_backend_buffer_clear(state_buffer, 0);
        if (weights.disk_store) weights.disk_store->invalidate_cache();
        valid_steps = 0;
    }

    std::unique_ptr<Graph> build(int64_t steps, int64_t audio_steps, int64_t attention_steps) {
        const auto & c = config;
        auto result = std::make_unique<Graph>(execution.backend());
        constexpr size_t nodes = 32768;
        const size_t arena = ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false);
        result->context = ggml_init({arena, nullptr, true});
        if (!result->context) throw std::runtime_error("Ming3 Audio failed to allocate Thinker graph metadata");
        result->steps = steps;
        result->audio_steps = audio_steps;
        core::ModuleBuildContext ctx{result->context, "ming3_audio.thinker", execution.backend_type()};
        result->graph = ggml_new_graph_custom(ctx.ggml, nodes, false);
        result->ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
        result->positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
        result->mask = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, steps, attention_steps}));
        ggml_set_input(result->ids.tensor);
        ggml_set_input(result->positions.tensor);
        ggml_set_input(result->mask.tensor);
        auto * embeddings = ggml_get_rows(ctx.ggml, weights.embedding.tensor, result->ids.tensor);
        if (audio_steps > 0) {
            result->audio = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({audio_steps, c.hidden_size}));
            result->audio_indices = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({audio_steps}));
            ggml_set_input(result->audio.tensor);
            ggml_set_input(result->audio_indices.tensor);
            embeddings = ggml_set_rows(ctx.ggml, embeddings, result->audio.tensor, result->audio_indices.tensor);
        }
        auto x = core::wrap_tensor(ggml_reshape_3d(ctx.ggml, embeddings, c.hidden_size, steps, 1),
            core::TensorShape::from_dims({1, steps, c.hidden_size}));
        x = modules::Bf16RoundModule{}.build(ctx, x);
        for (int64_t layer = 0; layer < c.layers; ++layer) {
            const auto & w = weights.layers[layer];
            const auto & state = states[layer];
            modules::NormConfig norm_config{c.hidden_size, c.rms_norm_eps, true, false};
            norm_config.normalized_round_type = GGML_TYPE_BF16;
            auto normalized = modules::Bf16RoundModule{}.build(ctx,
                modules::RMSNormModule(norm_config).build(ctx, x, w.input_norm));
            core::TensorValue output;
            if (state.key.valid()) {
                output = build_mla(ctx, normalized, result->positions, result->mask, state.key, state.value, w.mla, c);
            } else {
                const auto kda = build_kda(ctx, normalized, state.tails, state.recurrent, w.kda, c);
                output = kda.hidden;
                for (size_t i = 0; i < state.tails.size(); ++i) {
                    ggml_build_forward_expand(result->graph, ggml_cpy(ctx.ggml, kda.conv_tails[i].tensor, state.tails[i].tensor));
                }
                ggml_build_forward_expand(result->graph, ggml_cpy(ctx.ggml, kda.recurrent_state.tensor, state.recurrent.tensor));
            }
            x = modules::Bf16RoundModule{}.build(ctx, modules::AddModule{}.build(ctx, x, output));
            normalized = modules::Bf16RoundModule{}.build(ctx,
                modules::RMSNormModule(norm_config).build(ctx, x, w.post_norm));
            output = build_thinker_mlp(ctx, normalized, w, c, weights.group_identity, layer >= c.dense_layers);
            x = modules::Bf16RoundModule{}.build(ctx, modules::AddModule{}.build(ctx, x, output));
        }
        x = modules::SliceModule({1, steps - 1, 1}).build(ctx, x);
        modules::NormConfig norm_config{c.hidden_size, c.rms_norm_eps, true, false};
        norm_config.normalized_round_type = GGML_TYPE_BF16;
        x = modules::Bf16RoundModule{}.build(ctx, modules::RMSNormModule(norm_config).build(ctx, x, weights.norm));
        result->logits = ggml_round_bf16(ctx.ggml, ggml_mul_mat(ctx.ggml, weights.head.tensor, x.tensor));
        ggml_set_output(result->logits);
        ggml_build_forward_expand(result->graph, result->logits);
        if (weights.disk_store) {
            result->cleanup_context = ggml_init({ggml_graph_overhead_custom(1, false) + 1024, nullptr, true});
            if (!result->cleanup_context) throw std::runtime_error("Ming3 Audio failed to allocate capture cleanup metadata");
            result->cleanup_graph = ggml_new_graph_custom(result->cleanup_context, 1, false);
            ggml_backend_t backends[] = {execution.backend(), routing_execution->backend()};
            result->scheduler = ggml_backend_sched_new(backends, nullptr, 2, nodes, false, false);
            if (!result->scheduler) throw std::runtime_error("Ming3 Audio failed to allocate expert scheduler");
            ggml_backend_sched_set_eval_callback(result->scheduler, core::DiskExpertWeightStore::evaluate, weights.disk_store.get());
            auto * compute_graph = ggml_new_graph_custom(ctx.ggml, nodes, false);
            for (int i = 0; i < ggml_graph_n_nodes(result->graph); ++i) {
                auto * node = ggml_graph_node(result->graph, i);
                const bool routing = core::DiskExpertWeightStore::evaluate(node, true, weights.disk_store.get());
                ggml_backend_sched_set_tensor_backend(result->scheduler, node,
                    routing ? routing_execution->backend() : execution.backend());
                if (!routing) ggml_graph_add_node(compute_graph, node);
            }
            core::validate_backend_graph_supported(execution.backend(), compute_graph, "ming3_audio.thinker");
            if (!ggml_backend_sched_alloc_graph(result->scheduler, result->graph)) {
                throw std::runtime_error("Ming3 Audio failed to allocate offload graph");
            }
        } else {
            core::validate_backend_graph_supported(execution.backend(), result->graph, "ming3_audio.thinker");
            result->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
            if (!result->allocator || !ggml_gallocr_alloc_graph(result->allocator, result->graph)) {
                throw std::runtime_error("Ming3 Audio failed to allocate Thinker graph");
            }
            core::prepare_host_graph_plan(execution, result->graph, result->plan);
        }
        debug::timing_log_scalar("ming3_audio.thinker.graph_buffer_mb",
            (result->scheduler ? ggml_backend_sched_get_buffer_size(result->scheduler, execution.backend())
                : ggml_gallocr_get_buffer_size(result->allocator, 0)) / (1024.0 * 1024.0));
        debug::timing_log_context_reservation("ming3_audio.thinker.graph", result->context);
        return result;
    }

    std::vector<float> run(const std::vector<int32_t> & ids, const std::vector<float> & audio, int32_t audio_id) {
        const auto steps = static_cast<int64_t>(ids.size());
        if (steps > 0 && options.disk_experts && valid_steps + steps > capacity) {
            reserve(std::max(capacity * 2, valid_steps + steps), true);
        }
        if (steps <= 0 || valid_steps + steps > capacity) throw std::runtime_error("Ming3 Audio Thinker context capacity exceeded");
        std::vector<int32_t> indices;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] < 0 || ids[i] >= config.vocabulary_size) throw std::runtime_error("Ming3 Audio token is outside vocabulary");
            if (ids[i] == audio_id) indices.push_back(static_cast<int32_t>(i));
        }
        if (audio.size() != indices.size() * static_cast<size_t>(config.hidden_size)) {
            throw std::runtime_error("Ming3 Audio placeholder count differs from projected audio length");
        }
        if (weights.disk_store && steps > 1) {
            if (!offload_prefill) {
                offload_allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
                if (!offload_allocator) throw std::runtime_error("Ming3 Audio failed to create offload prefill allocator");
                offload_prefill = std::make_unique<OffloadPrefill>(execution, weights, config, states,
                    std::min<int64_t>(256, capacity), capacity, offload_allocator.get(), true);
            }
            auto * graph = offload_prefill.get();
            if (graph->steps() != steps) {
                if (!offload_tail || offload_tail->steps() != steps) {
                    offload_tail.reset();
                    offload_tail = std::make_unique<OffloadPrefill>(execution, weights, config, states,
                        steps, capacity, offload_allocator.get(), false);
                }
                graph = offload_tail.get();
            }
            auto result = graph->run(ids, audio, indices, valid_steps);
            valid_steps += steps;
            return result;
        }
        const bool decode = steps == 1 && indices.empty();
        const auto attention_steps = decode && execution.backend_type() == core::BackendType::Cuda
            ? std::min(capacity, runtime::bucketed_cache_steps(valid_steps + 1, capacity, 0, 256))
            : capacity;
        auto & cached = decode ? step_graphs[attention_steps] : prefill_graph;
        if (!cached || cached->steps != steps || cached->audio_steps != static_cast<int64_t>(indices.size())) {
            cached.reset();
            cached = build(steps, indices.size(), attention_steps);
        }
        std::vector<int32_t> positions(steps);
        std::vector<float> mask(steps * attention_steps, -std::numeric_limits<float>::infinity());
        for (int64_t i = 0; i < steps; ++i) {
            positions[i] = static_cast<int32_t>(valid_steps + i);
            std::fill_n(mask.begin() + i * attention_steps, valid_steps + i + 1, 0.0f);
        }
        core::write_tensor_i32(cached->ids, ids);
        core::write_tensor_i32(cached->positions, positions);
        core::write_tensor_f32(cached->mask, mask);
        if (!indices.empty()) {
            auto rounded = audio;
            core::round_f32_to_bf16_in_place(rounded);
            core::write_tensor_f32(cached->audio, rounded);
            core::write_tensor_i32(cached->audio_indices, indices);
        }
        const auto status = cached->scheduler ? ggml_backend_sched_graph_compute(cached->scheduler, cached->graph)
            : core::compute_graph(execution, cached->graph, cached->plan, "ming3_audio.thinker");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Ming3 Audio Thinker graph execution failed");
        }
        valid_steps += steps;
        return core::read_tensor_f32(cached->logits);
    }

    core::ExecutionContext & execution;
    std::shared_ptr<const assets::TensorSource> source;
    ThinkerConfig config;
    ThinkerOptions options;
    int64_t capacity = 0;
    ThinkerWeights weights;
    std::unique_ptr<core::ExecutionContext> routing_execution;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> state_context;
    ggml_backend_buffer_t state_buffer = nullptr;
    std::vector<State> states;
    int64_t valid_steps = 0;
    std::unique_ptr<Graph> prefill_graph;
    std::map<int64_t, std::unique_ptr<Graph>> step_graphs;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, decltype(&ggml_gallocr_free)>
        offload_allocator{nullptr, ggml_gallocr_free};
    std::unique_ptr<OffloadPrefill> offload_prefill, offload_tail;
};

Thinker::Thinker(core::ExecutionContext & execution, std::shared_ptr<const assets::TensorSource> source,
                 ThinkerConfig config, int64_t capacity, ThinkerOptions options)
    : impl_(std::make_unique<Impl>(execution, std::move(source), config, capacity, options)) {}
Thinker::~Thinker() = default;
void Thinker::reset() { impl_->reset(); }
void Thinker::reserve(int64_t capacity) { impl_->reserve(capacity); }
void Thinker::release_workspace() { impl_->release_workspace(); }
std::vector<float> Thinker::prefill(const std::vector<int32_t> & ids, const std::vector<float> & audio, int32_t audio_id) {
    const size_t chunk_size = 256;
    if (ids.size() <= chunk_size) return impl_->run(ids, audio, audio_id);
    const auto hidden = static_cast<size_t>(impl_->config.hidden_size);
    const auto audio_tokens = static_cast<size_t>(std::count(ids.begin(), ids.end(), audio_id));
    if (audio.size() != audio_tokens * hidden) {
        throw std::runtime_error("Ming3 Audio placeholder count differs from projected audio length");
    }
    debug::trace_log_scalar("ming3_audio.thinker.prefill_chunks", (ids.size() + chunk_size - 1) / chunk_size);
    std::vector<int32_t> chunk;
    std::vector<float> audio_chunk;
    std::vector<float> logits;
    chunk.reserve(chunk_size);
    audio_chunk.reserve(chunk_size * hidden);
    size_t audio_offset = 0;
    for (size_t offset = 0; offset < ids.size(); offset += chunk_size) {
        chunk.assign(ids.begin() + offset, ids.begin() + std::min(offset + chunk_size, ids.size()));
        const auto count = static_cast<size_t>(std::count(chunk.begin(), chunk.end(), audio_id)) * hidden;
        audio_chunk.assign(audio.begin() + audio_offset, audio.begin() + audio_offset + count);
        logits = impl_->run(chunk, audio_chunk, audio_id);
        audio_offset += count;
    }
    return logits;
}
std::vector<float> Thinker::step(int32_t token) { return impl_->run({token}, {}, -1); }

}  // namespace engine::models::ming3_audio
