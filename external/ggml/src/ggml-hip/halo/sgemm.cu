#include "sgemm.cuh"

namespace {

// Reuse each 64x32 operand tile for 64 output rows/columns. Unlike reduced-
// precision matrix instructions, this retains FP32 inputs and accumulation.
__global__ __launch_bounds__(256) void halo_sgemm(
    const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    int64_t sa, int64_t sb, int64_t sc) {
    __shared__ float aa[32][65];
    __shared__ float bb[32][65];
    const int tid = threadIdx.x;
    const int row = tid % 16;
    const int col = tid / 16;
    const int64_t row0 = int64_t(blockIdx.x) * 64;
    const int64_t col0 = int64_t(blockIdx.y) * 64;
    a += int64_t(blockIdx.z) * sa;
    b += int64_t(blockIdx.z) * sb;
    c += int64_t(blockIdx.z) * sc;
    float sum[4][4] = {};
    for (int64_t base = 0; base < k; base += 32) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int index = tid + i * 256;
            const int kk = index % 32;
            const int rr = index / 32;
            aa[kk][rr] = base + kk < k && row0 + rr < m ? a[(row0 + rr) * lda + base + kk] : 0;
            bb[kk][rr] = base + kk < k && col0 + rr < n ? b[(col0 + rr) * ldb + base + kk] : 0;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 32; ++kk) {
            float av[4], bv[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                av[i] = aa[kk][row + i * 16];
                bv[i] = bb[kk][col + i * 16];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    sum[i][j] = fmaf(av[i], bv[j], sum[i][j]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (row0 + row + i * 16 < m && col0 + col + j * 16 < n) {
                c[(col0 + col + j * 16) * ldc + row0 + row + i * 16] = sum[i][j];
            }
        }
    }
}

} // namespace

void ggml_hip_halo_sgemm(const float * a, const float * b, float * c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    int64_t sa, int64_t sb, int64_t sc, int64_t batches, hipStream_t stream) {
    halo_sgemm<<<dim3((m + 63) / 64, (n + 63) / 64, batches), 256, 0, stream>>>(
        a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
}
