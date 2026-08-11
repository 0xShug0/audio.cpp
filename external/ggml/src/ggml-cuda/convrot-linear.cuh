#include "common.cuh"

void ggml_cuda_convrot_linear(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_convrot_linear_supported(int device, const ggml_tensor * dst);
