#pragma once
#include "common.cuh"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
void ggml_cuda_op_affine(ggml_backend_cuda_context &ctx, ggml_tensor *dst);
#endif
