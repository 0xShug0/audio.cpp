#include "common.cuh"

void ggml_cuda_sage_attn2(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_sage_attn2_i8(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_sage_attn2_supported(int device, const ggml_tensor * dst);

bool ggml_cuda_sage_attn2_i8_supported(int device, const ggml_tensor * dst);
