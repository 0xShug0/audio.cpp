#pragma once
#include "common.cuh"

// Private epilogues selected only by registered denoiser graphs.
void ggml_cuda_op_norm_affine(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_add_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_add_gate_residual(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
