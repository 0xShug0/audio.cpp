#pragma once
#include "common.cuh"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
void ggml_cuda_op_gru_cell(ggml_backend_cuda_context &ctx, ggml_tensor *dst);
void ggml_cuda_op_gru_scan(ggml_backend_cuda_context &ctx, ggml_tensor *dst);
#endif
