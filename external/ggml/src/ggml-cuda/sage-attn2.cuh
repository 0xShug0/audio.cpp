#include "common.cuh"

#if defined(GGML_CUDA_SAGE_ATTN2_ENABLED)
void ggml_cuda_sage_attn2(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_sage_attn2_i8(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_sage_attn2_supported(int device, const ggml_tensor * dst);

bool ggml_cuda_sage_attn2_i8_supported(int device, const ggml_tensor * dst);
#else
// SageAttention2 was not built for the selected CUDA architectures (their SM
// is below 89). Provide stub definitions so the ggml-cuda dispatcher still
// links; supported() returns false, so the op is never attached to a graph.
static inline void ggml_cuda_sage_attn2(ggml_backend_cuda_context &, ggml_tensor *) {
    GGML_ABORT("CUDA SageAttention2 was not built for the selected CUDA architectures");
}

static inline void ggml_cuda_sage_attn2_i8(ggml_backend_cuda_context &, ggml_tensor *) {
    GGML_ABORT("CUDA SageAttention2 was not built for the selected CUDA architectures");
}

static inline bool ggml_cuda_sage_attn2_supported(int device, const ggml_tensor * dst) {
    (void) device; (void) dst;
    return false;
}

static inline bool ggml_cuda_sage_attn2_i8_supported(int device, const ggml_tensor * dst) {
    (void) device; (void) dst;
    return false;
}
#endif
