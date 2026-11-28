#pragma once

#include "ggml-backend.h"
#include "ggml-impl.h"

#include <cstring>
#include <algorithm>

struct ggml_pointwise_match {
    int count = 0, kind = 0, activation = 0;
    bool round_bf16 = false, scale_transform = false;
    float eps = 0.0F, factor = 1.0F, offset = 0.0F;
    ggml_tensor * x = nullptr, * bias = nullptr, * gate = nullptr, * residual = nullptr, * dst = nullptr;
    ggml_tensor * logical = nullptr;
};

// Semantic recognition only. Backend code checks storage, dtype, hardware and
// aliasing. Every removed value must be private to this exact subgraph.
static inline ggml_pointwise_match ggml_match_pointwise(
        const ggml_cgraph * graph, int start, uint32_t mask, bool round_output = true) {
    ggml_pointwise_match match;
    auto * root = graph->nodes[start];
    auto * value = root;
    int end = start + 1;
    int cast_idx = -1;
    const auto next = [&]() -> ggml_tensor * {
        return end < graph->n_nodes ? graph->nodes[end] : nullptr;
    };
    const auto metadata = [&]() {
        while (end - start < 10) {
            auto * node = next();
            if (!node || (node->op != GGML_OP_RESHAPE && node->op != GGML_OP_VIEW) ||
                node->src[0] != value || node->view_offs != 0 ||
                !ggml_is_contiguous(node) || node->ne[0] != value->ne[0] ||
                ggml_nelements(node) != ggml_nelements(value)) break;
            value = node;
            ++end;
        }
    };
    if ((mask & GGML_BACKEND_GRAPH_FUSION_NORM_AFFINE) &&
        (root->op == GGML_OP_NORM || root->op == GGML_OP_RMS_NORM)) {
        match.kind = 2;
        match.x = root->src[0];
        match.activation = root->op == GGML_OP_RMS_NORM ? 1 : 0;
        std::memcpy(&match.eps, root->op_params, sizeof(float));
        metadata();
        ggml_tensor * scale_op = nullptr;
        if (next() && next()->op == GGML_OP_SCALE) { scale_op = next(); ++end; }
        auto * mul = next();
        if (!mul || mul->op != GGML_OP_MUL || mul->src[0] != value) return {};
        match.gate = mul->src[1];
        if (scale_op) {
            if (match.gate != scale_op) return {};
            match.gate = scale_op->src[0];
            match.scale_transform = true;
            std::memcpy(&match.factor, scale_op->op_params, sizeof(float));
            std::memcpy(&match.offset, reinterpret_cast<const char *>(scale_op->op_params) + sizeof(float), sizeof(float));
        }
        value = mul; ++end; metadata();
        auto * add = next();
        if (!add || add->op != GGML_OP_ADD || add->src[0] != value) return {};
        match.bias = add->src[1]; value = add; ++end;
    } else if (root->op == GGML_OP_ADD || root->op == GGML_OP_MUL) {
        match.x = root->src[0];
        // ADD_CAST carries a distinct output-rounding contract.
        if (root->type != match.x->type) return {};
        if (root->op == GGML_OP_ADD) match.bias = root->src[1];
        metadata();
        if (next() && next()->op == GGML_OP_CPY && next()->src[0] == value &&
            next()->src[1] == next() && value->type == GGML_TYPE_F16 && next()->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(next()) && ggml_are_same_shape(next(), value)) {
            cast_idx = end;
            value = next(); ++end; metadata();
        }
        auto * node = next();
        if (root->op == GGML_OP_ADD && (mask & GGML_BACKEND_GRAPH_FUSION_BIAS_ACTIVATION) &&
            node && node->op == GGML_OP_UNARY && node->src[0] == value) {
            switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_GELU: match.activation = 0; break;
                case GGML_UNARY_OP_SILU: match.activation = 1; break;
                case GGML_UNARY_OP_GELU_QUICK: match.activation = 2; break;
                case GGML_UNARY_OP_GELU_ERF: match.activation = 3; break;
                default: return {};
            }
            value = node; ++end;
        } else if (mask & GGML_BACKEND_GRAPH_FUSION_GATED_RESIDUAL) {
            match.kind = 1;
            if (root->op == GGML_OP_ADD) {
                if (!node || node->op != GGML_OP_MUL || node->src[0] != value) return {};
                match.gate = node->src[1]; value = node; ++end; metadata();
            } else match.gate = root->src[1];
            // The epilogue retains F32 multiplication before residual addition.
            // Half multiplication has a separate rounding boundary.
            if (value->type != GGML_TYPE_F32) return {};
            node = next();
            if (!node || node->op != GGML_OP_ADD) return {};
            if (node->src[0] == value) match.residual = node->src[1];
            else if (node->src[1] == value) match.residual = node->src[0];
            else return {};
            value = node; ++end;
        } else return {};
    } else return {};
    match.logical = value;
    const int math_end = end;
    auto * math_value = value;
    metadata();
    if (round_output && next() && next()->op == GGML_OP_UNARY &&
        ggml_get_unary_op(next()) == GGML_UNARY_OP_ROUND_BF16 && next()->src[0] == value) {
        match.round_bf16 = true; value = next(); ++end;
    } else if (match.kind == 2 && next() && next()->op == GGML_OP_CPY &&
        next()->src[0] == value && next()->src[1] == next() && value->type == GGML_TYPE_F32 &&
        next()->type == GGML_TYPE_F16 && ggml_is_contiguous(next()) && ggml_are_same_shape(next(), value)) {
        value = next(); ++end;
    } else {
        end = math_end;
        value = math_value;
    }
    match.dst = value;
    match.count = end - start;
    ggml_op ops[12];
    if (match.count > 12) return {};
    // A kernel input cannot be an intermediate that the same fusion elides.
    for (auto * input : {match.x, match.bias, match.gate, match.residual}) {
        if (!input) continue;
        for (int j = 0; j < match.count; ++j)
            if (graph->nodes[start + j] == input) return {};
    }
    for (int j = 0; j < match.count; ++j) ops[j] = graph->nodes[start + j]->op;
    const int outputs[] = {end - 1, cast_idx};
    // A cast has a self-reference in src[1]. Account for that reference while
    // proving its only external use is the next node in this fused chain.
    if (cast_idx >= 0 && ((graph->nodes[cast_idx]->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        !ggml_node_has_n_uses(graph, cast_idx, 2))) return {};
    if (!ggml_can_fuse_subgraph(graph, start, match.count, ops, outputs, cast_idx >= 0 ? 2 : 1)) return {};
    return match;
}


static inline bool ggml_pointwise_node(const ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE: return true;
        case GGML_OP_CPY: return !node->view_src && node->src[1] == node;
        case GGML_OP_CONT:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SCALE:
        case GGML_OP_UNARY: return !node->view_src;
        default: return false;
    }
}

static inline bool ggml_pointwise_reorderable_node(const ggml_tensor * node) {
    if (ggml_pointwise_node(node)) return true;
    if (node->view_src && node->op != GGML_OP_PERMUTE && node->op != GGML_OP_TRANSPOSE) return false;
    switch (node->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_GET_ROWS:
        case GGML_OP_REPEAT:
        case GGML_OP_DUP:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_SUB:
        case GGML_OP_DIV: return true;
        default: return false;
    }
}

// Defer private pointwise chains until their parameter branches are ready.
// This changes only topological order, never tensor expressions. Do it once,
// before allocation/capture; no scheduling or allocation is added per step.
static inline void ggml_prepare_pointwise_chains(ggml_cgraph * graph) {
    for (int end = 0; end < graph->n_nodes; ++end) {
        auto * current = graph->nodes[end];
        if (!ggml_pointwise_node(current)) continue;
        ggml_tensor * chain[12];
        int count = 0, first = end, cursor = end;
        while (count < 12) {
            ggml_tensor * parent = nullptr;
            int parent_index = -1;
            const int sources = current->op == GGML_OP_ADD || current->op == GGML_OP_MUL ? 2 : 1;
            for (int source = 0; source < sources && !parent; ++source) {
                auto * candidate = current->src[source];
                if (!candidate || !ggml_pointwise_node(candidate) ||
                    (candidate->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT))) continue;
                for (int j = cursor - 1; j >= 0; --j) {
                    if (graph->nodes[j] != candidate) continue;
                    const int uses = candidate->op == GGML_OP_CPY ? 2 : 1;
                    // Moving a private view preserves it and its source; unlike
                    // elision, it does not require a non-view tensor. The final
                    // fusion matcher separately proves ownership and aliasing.
                    if (ggml_node_get_use_count(graph, j) == uses) { parent = candidate; parent_index = j; }
                    break;
                }
            }
            if (!parent) break;
            // Keep a private affine-parameter SCALE beside its normalization
            // so the epilogue can preserve and evaluate that expression too.
            auto * scale = current->src[1];
            auto * normalized = parent;
            while ((normalized->op == GGML_OP_VIEW || normalized->op == GGML_OP_RESHAPE) &&
                normalized->view_offs == 0 && ggml_is_contiguous(normalized) && normalized->src[0] &&
                normalized->ne[0] == normalized->src[0]->ne[0] &&
                ggml_nelements(normalized) == ggml_nelements(normalized->src[0])) normalized = normalized->src[0];
            const bool affine = normalized->op == GGML_OP_NORM || normalized->op == GGML_OP_RMS_NORM;
            if (affine && current->op == GGML_OP_MUL && parent == current->src[0] && scale &&
                scale->op == GGML_OP_SCALE && !scale->view_src && count + 1 < 12 &&
                !(scale->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT))) {
                for (int j = cursor - 1; j >= 0; --j) {
                    if (graph->nodes[j] != scale) continue;
                    if (ggml_node_has_n_uses(graph, j, 1)) {
                        chain[count++] = scale;
                        if (j < first) first = j;
                    }
                    break;
                }
            }
            chain[count++] = parent;
            current = parent;
            cursor = parent_index;
            if (parent_index < first) first = parent_index;
        }
        if (!count) continue;
        bool safe = true;
        for (int j = first; j < end; ++j) {
            if (!ggml_pointwise_reorderable_node(graph->nodes[j])) { safe = false; break; }
        }
        if (!safe) continue;
        for (int k = count - 1; k >= 0; --k) {
            for (int j = first; j < end; ++j) {
                if (graph->nodes[j] != chain[k]) continue;
                std::rotate(graph->nodes + j, graph->nodes + j + 1, graph->nodes + end);
                break;
            }
        }
    }
}
