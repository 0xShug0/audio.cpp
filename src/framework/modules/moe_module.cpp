#include "engine/framework/modules/moe_module.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace engine::modules {

MoeModule::MoeModule(MoeExpertConfig experts) : experts_(experts) {
    if (experts.fused_swiglu && (experts.activation != MoeActivation::SwiGlu || experts.round_activation))
        throw std::invalid_argument("MoE fused SwiGLU cannot preserve an intermediate activation rounding boundary");
}

MoeRouting MoeModule::build_routing(core::ModuleBuildContext & ctx, ggml_tensor * input,
    ggml_tensor * router, ggml_tensor * selection_bias, const MoeRoutingConfig & c) {
    if (c.experts <= 0 || c.selected <= 0 || c.selected > c.experts || router->ne[1] != c.experts ||
        c.groups <= 0 || c.experts % c.groups || c.selected_groups <= 0 || c.selected_groups > c.groups ||
        (c.groups > 1 && (c.group_top_k <= 0 || c.group_top_k > c.experts / c.groups ||
            c.selected > c.selected_groups * (c.experts / c.groups))) ||
        !std::isfinite(c.scale) || !std::isfinite(c.denominator_epsilon) || c.denominator_epsilon < 0)
        throw std::invalid_argument("invalid MoE routing configuration");
    auto * g = ctx.ggml;
    const auto tokens = input->ne[1];
    auto * logits = ggml_mul_mat(g, router, input);
    if (c.precision != GGML_PREC_DEFAULT) ggml_mul_mat_set_prec(logits, c.precision);
    auto * scores = c.scoring == MoeScoring::Softmax ? ggml_soft_max(g, logits) : ggml_sigmoid(g, logits);
    auto * selection = selection_bias ? ggml_add(g, scores, selection_bias) : scores;
    if (c.groups > 1) {
        const auto per_group = c.experts / c.groups;
        auto * groups = ggml_reshape_3d(g, selection, per_group, c.groups, tokens);
        auto * best = ggml_argsort_top_k(g, groups, static_cast<int>(c.group_top_k));
        best = ggml_get_rows(g, ggml_reshape_4d(g, groups, 1, per_group, c.groups, tokens), best);
        auto * group_scores = ggml_sum_rows(g, ggml_reshape_3d(g, best, c.group_top_k, c.groups, tokens));
        group_scores = ggml_reshape_2d(g, group_scores, c.groups, tokens);
        auto * selected_groups = ggml_argsort_top_k(g, group_scores, static_cast<int>(c.selected_groups));
        auto * selected = ggml_get_rows(g, groups, selected_groups);
        selection = ggml_set_rows(g, ggml_fill(g, groups, -std::numeric_limits<float>::infinity()), selected, selected_groups);
        selection = ggml_reshape_2d(g, selection, c.experts, tokens);
    }
    auto * ids = ggml_argsort_top_k(g, selection, static_cast<int>(c.selected));
    if (c.contiguous_ids && !ggml_is_contiguous(ids)) ids = ggml_cont(g, ids);
    auto * weights = ggml_get_rows(g, ggml_reshape_3d(g, scores, 1, c.experts, tokens), ids);
    weights = ggml_reshape_2d(g, weights, c.selected, tokens);
    if (c.normalization != MoeNormalization::None) {
        auto * denominator = ggml_sum_rows(g, weights);
        if (c.normalization == MoeNormalization::Clamp)
            denominator = ggml_clamp(g, denominator, c.denominator_epsilon, std::numeric_limits<float>::infinity());
        else if (c.normalization == MoeNormalization::AddEpsilon)
            denominator = ggml_scale_bias(g, denominator, 1.0F, c.denominator_epsilon);
        weights = ggml_div(g, weights, denominator);
    }
    if (c.scale != 1.0F) weights = ggml_scale(g, weights, c.scale);
    return {ids, weights};
}

ggml_tensor * MoeModule::build_experts(core::ModuleBuildContext & ctx, ggml_tensor * input,
    ggml_tensor * ids, const MoeExpertWeights & weights) const {
    auto * g = ctx.ggml;
    if (!weights.up || !weights.down || (experts_.activation == MoeActivation::SwiGlu && !weights.gate) ||
        weights.up->ne[0] < input->ne[0] || weights.down->ne[0] < weights.up->ne[1])
        throw std::invalid_argument("invalid MoE expert projection shapes");
    if (weights.up->ne[0] > input->ne[0])
        input = ggml_pad(g, input, static_cast<int>(weights.up->ne[0] - input->ne[0]), 0, 0, 0);
    ggml_tensor * activated;
    if (experts_.activation == MoeActivation::SwiGlu) {
        auto * gate = ggml_mul_mat_id(g, weights.gate, input, ids);
        auto * up = ggml_mul_mat_id(g, weights.up, input, ids);
        if (experts_.round_projections) {
            gate = ggml_round_bf16(g, gate);
            up = ggml_round_bf16(g, up);
        }
        if (experts_.fused_swiglu) activated = ggml_swiglu_split(g, gate, up);
        else {
            gate = ggml_silu(g, gate);
            if (experts_.round_activation) gate = ggml_round_bf16(g, gate);
            activated = ggml_mul(g, gate, up);
        }
    } else {
        auto * up = ggml_mul_mat_id(g, weights.up, input, ids);
        if (experts_.round_projections) up = ggml_round_bf16(g, up);
        activated = ggml_relu(g, up);
        if (experts_.round_activation) activated = ggml_round_bf16(g, activated);
        activated = ggml_sqr(g, activated);
    }
    if (experts_.round_product) activated = ggml_round_bf16(g, activated);
    if (weights.down->ne[0] > activated->ne[0])
        activated = ggml_pad(g, activated, static_cast<int>(weights.down->ne[0] - activated->ne[0]), 0, 0, 0);
    auto * output = ggml_mul_mat_id(g, weights.down, activated, ids);
    return experts_.round_output ? ggml_round_bf16(g, output) : output;
}

ggml_tensor * MoeModule::combine(core::ModuleBuildContext & ctx, ggml_tensor * outputs, ggml_tensor * weights) {
    const auto h = outputs->ne[0], k = outputs->ne[1], tokens = outputs->ne[2];
    auto * g = ctx.ggml;
    auto * scaled = ggml_mul(g, outputs, ggml_reshape_3d(g, weights, 1, k, tokens));
    auto * reduced = ggml_cont(g, ggml_permute(g, scaled, 1, 0, 2, 3));
    return ggml_reshape_2d(g, ggml_sum_rows(g, reduced), h, tokens);
}

ggml_tensor * MoeModule::build(core::ModuleBuildContext & ctx, ggml_tensor * input,
    const MoeRouting & routing, const MoeExpertWeights & weights, ggml_tensor * bank_ids) const {
    auto * rows = ggml_reshape_3d(ctx.ggml, input, input->ne[0], 1, input->ne[1]);
    return combine(ctx, build_experts(ctx, rows, bank_ids ? bank_ids : routing.ids, weights), routing.weights);
}

} // namespace engine::modules
