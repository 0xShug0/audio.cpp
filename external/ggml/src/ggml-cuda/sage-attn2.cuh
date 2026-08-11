#pragma once

#include "common.cuh"

#if defined(GGML_USE_HIP)

static inline void ggml_cuda_sage_attn2(ggml_backend_cuda_context &, ggml_tensor *) {
    GGML_ABORT("SAGE_ATTN2 is not supported by the HIP backend");
}

static inline void ggml_cuda_sage_attn2_i8(ggml_backend_cuda_context &, ggml_tensor *) {
    GGML_ABORT("SAGE_ATTN2_I8 is not supported by the HIP backend");
}

static inline bool ggml_cuda_sage_attn2_supported(int, const ggml_tensor *) {
    return false;
}

static inline bool ggml_cuda_sage_attn2_i8_supported(int, const ggml_tensor *) {
    return false;
}

#elif defined(GGML_CUDA_SAGE_ATTN2_ENABLED)

void ggml_cuda_sage_attn2(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_sage_attn2_i8(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_sage_attn2_supported(int device, const ggml_tensor * dst);

bool ggml_cuda_sage_attn2_i8_supported(int device, const ggml_tensor * dst);

#else

static inline void ggml_cuda_sage_attn2(ggml_backend_cuda_context &, ggml_tensor *) {
    GGML_ABORT("CUDA SageAttention2 was not built for the selected CUDA architectures");
}

static inline void ggml_cuda_sage_attn2_i8(ggml_backend_cuda_context &, ggml_tensor *) {
    GGML_ABORT("CUDA SageAttention2 I8 was not built for the selected CUDA architectures");
}

static inline bool ggml_cuda_sage_attn2_supported(int, const ggml_tensor *) {
    return false;
}

static inline bool ggml_cuda_sage_attn2_i8_supported(int, const ggml_tensor *) {
    return false;
}

#endif
