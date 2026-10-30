#include "gfx1151_v1.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/mma.cuh"
#include <climits>
#include <type_traits>

namespace {

// Reuse each 64x32 operand tile for 64 output rows/columns. Unlike reduced-
// precision matrix instructions, this retains FP32 inputs and accumulation.
template<bool batched = false>
__global__ __launch_bounds__(256) void hip_gfx1151_v1_sgemm(
    const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    int64_t sa, int64_t sb, int64_t sc, int64_t sa3, int64_t sb3, uint3 ne2,
    uint3 r2, uint3 r3) {
    __shared__ float aa[32][65];
    __shared__ float bb[32][65];
    const int tid = threadIdx.x;
    const int row = tid % 16;
    const int col = tid / 16;
    const int64_t row0 = int64_t(blockIdx.x) * 64;
    const int64_t col0 = int64_t(blockIdx.y) * 64;
    if constexpr (batched) {
        const uint32_t i3 = fastdiv(blockIdx.z, ne2);
        const uint32_t i2 = blockIdx.z - i3 * ne2.z;
        a += int64_t(fastdiv(i2, r2)) * sa + int64_t(fastdiv(i3, r3)) * sa3;
        b += i2 * sb + i3 * sb3;
    } else {
        a += int64_t(blockIdx.z) * sa;
        b += int64_t(blockIdx.z) * sb;
    }
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

// Default precision only: round F32 operands to BF16 in LDS, accumulate in F32.
// Full K32 chunks and 16-byte-aligned input rows are guaranteed by admission.
template<bool batched = false>
__global__ __launch_bounds__(256) void hip_gfx1151_v1_sgemm_bf16_aligned(
    const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb,
    int64_t sa, int64_t sb, int64_t sa3, int64_t sb3, uint3 ne2, uint3 r2, uint3 r3) {
#if defined(AMD_WMMA_AVAILABLE)
    if constexpr (batched) {
        const uint32_t i3 = fastdiv(blockIdx.z, ne2);
        const uint32_t i2 = blockIdx.z - i3 * ne2.z;
        a += int64_t(fastdiv(i2, r2)) * sa + int64_t(fastdiv(i3, r3)) * sa3;
        b += i2 * sb + i3 * sb3;
        c += int64_t(blockIdx.z) * m * n;
    }
    using namespace ggml_cuda_mma;
    using tile_AB = tile<16, 8, nv_bfloat162, get_input_data_layout()>;
    using tile_C = tile<16, 16, float, DATA_LAYOUT_J_MAJOR>;
    constexpr int pitch = 20;
    __shared__ __align__(16) nv_bfloat162 aa[128][pitch];
    __shared__ __align__(16) nv_bfloat162 bb[128][pitch];
    const int tid = threadIdx.x + 32 * threadIdx.y;
    const int wave_row = (threadIdx.y % 4) * 32;
    const int wave_col = (threadIdx.y / 4) * 64;
    const int64_t row0 = int64_t(blockIdx.x) * 128;
    const int64_t col0 = int64_t(blockIdx.y) * 128;
    tile_C acc[2][4];

    for (int64_t base = 0; base < k; base += 32) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int index = tid + i * 256;
            const int row = index / 4;
            const int pair = 4 * (index % 4);
            const int64_t kk = base + 2 * pair;
            float4 lo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 hi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (row0 + row < m) {
                const float * src = a + (row0 + row) * lda + kk;
                lo = *reinterpret_cast<const float4 *>(src);
                hi = *reinterpret_cast<const float4 *>(src + 4);
            }
            aa[row][pair + 0] = ggml_cuda_cast<nv_bfloat162>(make_float2(lo.x, lo.y));
            aa[row][pair + 1] = ggml_cuda_cast<nv_bfloat162>(make_float2(lo.z, lo.w));
            aa[row][pair + 2] = ggml_cuda_cast<nv_bfloat162>(make_float2(hi.x, hi.y));
            aa[row][pair + 3] = ggml_cuda_cast<nv_bfloat162>(make_float2(hi.z, hi.w));
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int index = tid + i * 256;
            const int col = index / 4;
            const int pair = 4 * (index % 4);
            const int64_t kk = base + 2 * pair;
            float4 lo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 hi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (col0 + col < n) {
                const float * src = b + (col0 + col) * ldb + kk;
                lo = *reinterpret_cast<const float4 *>(src);
                hi = *reinterpret_cast<const float4 *>(src + 4);
            }
            bb[col][pair + 0] = ggml_cuda_cast<nv_bfloat162>(make_float2(lo.x, lo.y));
            bb[col][pair + 1] = ggml_cuda_cast<nv_bfloat162>(make_float2(lo.z, lo.w));
            bb[col][pair + 2] = ggml_cuda_cast<nv_bfloat162>(make_float2(hi.x, hi.y));
            bb[col][pair + 3] = ggml_cuda_cast<nv_bfloat162>(make_float2(hi.z, hi.w));
        }
        __syncthreads();
#pragma unroll
        for (int kp = 0; kp < 16; kp += 8) {
            tile_AB a_frag[2];
            tile_AB b_frag[4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                load_ldmatrix(a_frag[i], &aa[wave_row + 16 * i][kp], pitch);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                load_ldmatrix(b_frag[j], &bb[wave_col + 16 * j][kp], pitch);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    mma(acc[i][j], a_frag[i], b_frag[j]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int64_t row = row0 + wave_row + 16 * i + tile_C::get_i(l);
                const int64_t col = col0 + wave_col + 16 * j + tile_C::get_j(l);
                if (row < m && col < n) {
                    c[col * m + row] = acc[i][j].x[l];
                }
            }
        }
    }
#else
    NO_DEVICE_CODE;
#endif
}

} // namespace

bool ggml_hip_gfx1151_v1_mul_mat(const ggml_tensor * a, const ggml_tensor * b,
                          ggml_tensor * out, hipStream_t stream, int cc) {
    const bool wmma = GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc);
    if (a->type != GGML_TYPE_F32) {
        return wmma && ggml_hip_gfx1151_v1_mul_mat_typed(a, b, out, stream);
    }
    if (out->op != GGML_OP_MUL_MAT ||
        ggml_get_op_params_i32(out, 1) != GGML_MUL_MAT_LOWERING_DEFAULT) {
        return false;
    }
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32 ||
        a->ne[0] != b->ne[0] || b->ne[2] % a->ne[2] != 0 || b->ne[3] % a->ne[3] != 0 ||
        a->nb[0] != sizeof(float) || b->nb[0] != sizeof(float) || !ggml_is_contiguous(out)) {
        return false;
    }
    const int64_t m = a->ne[1], n = b->ne[1], k = a->ne[0];
    if (b->ne[2] > 65535 || b->ne[3] > 65535 / b->ne[2] ||
        (b->ne[2] > 1 && (a->nb[2] % sizeof(float) || b->nb[2] % sizeof(float))) ||
        (b->ne[3] > 1 && (a->nb[3] % sizeof(float) || b->nb[3] % sizeof(float)))) {
        return false;
    }
    const int64_t batch = b->ne[2] * b->ne[3];
    if (m < 1 || n < 1 || k < 1 || m > INT_MAX || n > 65535 * 64 || k > INT_MAX ||
        a->nb[1] < size_t(k) * sizeof(float) || b->nb[1] < size_t(k) * sizeof(float) ||
        a->nb[1] % sizeof(float) || b->nb[1] % sizeof(float)) {
        return false;
    }
    const uint3 r2 = init_fastdiv_values(b->ne[2] / a->ne[2]);
    const uint3 r3 = init_fastdiv_values(b->ne[3] / a->ne[3]);
    const int64_t wide_cols = (n + 127) / 128;
    if (wmma &&
        ggml_get_op_params_i32(out, 0) == GGML_PREC_DEFAULT &&
        ggml_get_op_params_i32(out, 1) == GGML_MUL_MAT_LOWERING_DEFAULT &&
        reinterpret_cast<std::uintptr_t>(a->data) % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(b->data) % 16 == 0 &&
        (a->nb[1] / sizeof(float)) % 4 == 0 && (b->nb[1] / sizeof(float)) % 4 == 0 &&
        k % 32 == 0 &&
        (batch == 1 || (a->nb[2] % 16 == 0 && a->nb[3] % 16 == 0 &&
                        b->nb[2] % 16 == 0 && b->nb[3] % 16 == 0))) {
        const auto launch = [&](auto batched_tag) {
            constexpr bool batched = decltype(batched_tag)::value;
            hip_gfx1151_v1_sgemm_bf16_aligned<batched><<<dim3((m + 127) / 128, wide_cols, batch), dim3(32, 8), 0, stream>>>(
                static_cast<const float *>(a->data), static_cast<const float *>(b->data),
                static_cast<float *>(out->data), m, n, k,
                a->nb[1] / sizeof(float), b->nb[1] / sizeof(float),
                a->nb[2] / sizeof(float), b->nb[2] / sizeof(float),
                a->nb[3] / sizeof(float), b->nb[3] / sizeof(float), init_fastdiv_values(b->ne[2]), r2, r3);
        };
        if (batch > 1) launch(std::true_type{});
        else launch(std::false_type{});
        return true;
    }
    if (batch > 1) {
        hip_gfx1151_v1_sgemm<true><<<dim3((m + 63) / 64, (n + 63) / 64, batch), 256, 0, stream>>>(
            static_cast<const float *>(a->data), static_cast<const float *>(b->data),
            static_cast<float *>(out->data), m, n, k, a->nb[1] / sizeof(float),
            b->nb[1] / sizeof(float), m, a->nb[2] / sizeof(float), b->nb[2] / sizeof(float),
            m * n, a->nb[3] / sizeof(float), b->nb[3] / sizeof(float), init_fastdiv_values(b->ne[2]), r2, r3);
    } else {
        hip_gfx1151_v1_sgemm<false><<<dim3((m + 63) / 64, (n + 63) / 64), 256, 0, stream>>>(
            static_cast<const float *>(a->data), static_cast<const float *>(b->data),
            static_cast<float *>(out->data), m, n, k, a->nb[1] / sizeof(float),
            b->nb[1] / sizeof(float), m, 0, 0, 0, 0, 0, uint3{}, uint3{}, uint3{});
    }
    return true;
}
