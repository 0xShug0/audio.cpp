#pragma once

#include "engine/framework/core/module.h"
#include "ggml.h"

namespace engine::models::yue2::detail {

// Q is already transposed to ggml [dim, query, head, batch]. SYCL flash
// addresses query/head/batch through nb[1..3]; only dim must be contiguous.
// Keep the packing required by the eager matmul paths and leave every other
// backend's existing behavior unchanged.
inline ggml_tensor * prepare_attention_queries(
    ggml_context * ctx, ggml_tensor * q, core::BackendType backend, bool flash) {
    if (backend == core::BackendType::Sycl && flash &&
        q->type == GGML_TYPE_F32 && q->nb[0] == sizeof(float)) {
        return q;
    }
    return ggml_cont(ctx, q);
}

} // namespace engine::models::yue2::detail
