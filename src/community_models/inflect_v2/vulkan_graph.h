#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/structural_modules.h"

namespace engine::models::inflect_v2::detail {

inline void configure_vulkan_graph(ggml_cgraph * graph, core::BackendType backend_type) {
    if (backend_type != core::BackendType::Vulkan) {
        return;
    }
    // The deployed Inflect weights are FP32. Vulkan's default matmul may use
    // FP16 accumulation; explicitly retain FP32 accumulation in this model.
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_MUL_MAT) {
            ggml_mul_mat_set_prec(node, GGML_PREC_F32);
        }
    }
}

inline core::TensorValue attention_probability_row(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & probabilities,
    int64_t row) {
    auto value = modules::SliceModule({2, row, 1}).build(ctx, probabilities);
    if (ctx.backend_type == core::BackendType::Vulkan) {
        // Vulkan matmul requires aligned bindings; this view can start at an
        // arbitrary token-stride offset. Preserve CPU/CUDA's existing views.
        value = core::ensure_backend_addressable_layout(ctx, value);
    }
    return value;
}

} // namespace engine::models::inflect_v2::detail
