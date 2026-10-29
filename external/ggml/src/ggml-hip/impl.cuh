#pragma once

#include "ggml.h"
#include <hip/hip_runtime.h>

// False means ineligible, before any kernel launches. Execution errors are not retried.
struct ggml_hip_impl {
    const char * name;
    bool (*mul_mat)(const ggml_tensor *, const ggml_tensor *, ggml_tensor *, hipStream_t);
    bool (*im2col)(const ggml_tensor *, ggml_tensor *, hipStream_t);
    bool (*cpy)(const ggml_tensor *, ggml_tensor *, hipStream_t);
};
