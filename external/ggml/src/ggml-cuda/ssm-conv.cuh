#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

void ggml_cuda_op_ssm_conv_causal(ggml_backend_cuda_context & ctx, ggml_tensor * conv, ggml_tensor * bias, ggml_tensor * dst);
