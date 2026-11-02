#pragma once

#include "common.cuh"

// Mirrors the direct cuDNN kernel contract; false when the build option is off.
bool ggml_cuda_supports_conv3d(const ggml_tensor * op);
void ggml_cuda_op_conv3d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_conv3d_concat_pad_spatial_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
