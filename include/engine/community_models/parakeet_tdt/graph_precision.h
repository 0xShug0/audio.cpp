#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stdexcept>

namespace engine::community_models::parakeet_tdt {

inline void configure_cuda_matmul_precision(ggml_backend_t backend, bool force_f32) {
    if (!force_f32) return;
    auto * device = ggml_backend_get_device(backend);
    auto fn = device ? reinterpret_cast<bool (*)(ggml_backend_t, bool)>(
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_backend_cuda_set_f32_matmul")) : nullptr;
    if (!fn || !fn(backend, true)) {
        throw std::runtime_error("Parakeet F32 precision requires the CUDA backend's instance-scoped F32 matmul control");
    }
}

// Opt-in package policy, applied before backend allocation/capture. In
// particular this avoids Vulkan's default half-precision accumulation for
// Phonon. Other Parakeet packages retain their existing backend policy.
inline void configure_matmul_precision(ggml_cgraph * graph, bool force_f32) {
    if (!force_f32) return;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_MUL_MAT) {
            ggml_mul_mat_set_prec(node, GGML_PREC_F32);
        }
    }
}

}  // namespace engine::community_models::parakeet_tdt
