#pragma once

#include "engine/framework/core/module.h"
#include "ggml.h"

namespace engine::models::yue2::detail {

// The timestep MLP produces one F32 vector, shared by every acoustic frame.
// SYCL's binary-broadcast ADD indexes rhs rows modulo ne[1..3], so a REPEAT
// before ADD only materializes the same vector N times. Keep other backends'
// existing repeat+add graph until they are independently checked.
inline ggml_tensor * add_time_conditioning(
    ggml_context * ctx, ggml_tensor * hidden, ggml_tensor * time,
    core::BackendType backend) {
    GGML_ASSERT(hidden->type == GGML_TYPE_F32 && time->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(hidden) && ggml_is_contiguous(time));
    GGML_ASSERT(hidden->ne[0] == time->ne[0]);
    GGML_ASSERT(time->ne[1] == 1 && time->ne[2] == 1 && time->ne[3] == 1);
    auto * rhs = backend == core::BackendType::Sycl ? time : ggml_repeat(ctx, time, hidden);
    return ggml_add(ctx, hidden, rhs);
}

} // namespace engine::models::yue2::detail
