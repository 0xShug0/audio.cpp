#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

// Column-major C = A^T B, matching GGML's FP32 BLAS layout.
void ggml_hip_halo_sgemm(const float * a, const float * b, float * c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    int64_t stride_a, int64_t stride_b, int64_t stride_c, int64_t batches,
    hipStream_t stream);
