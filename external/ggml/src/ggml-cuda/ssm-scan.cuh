#include "common.cuh"

void ggml_cuda_op_ssm_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
void ggml_cuda_op_ssm_scan_gated(
    ggml_backend_cuda_context & ctx, ggml_tensor * scan, ggml_tensor * scale, ggml_tensor * glu);
#endif
