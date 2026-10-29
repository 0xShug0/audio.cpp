#include "gfx1151_v1.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/mma.cuh"
#include <climits>

namespace {

template<typename pair_type>
__global__ __launch_bounds__(128) void gfx1151_v1_matmul_tiled(
        const char * a, const float * b, float * c,
        int64_t m, int64_t n, int64_t k, size_t lda, int64_t ldb) {
#if defined(AMD_WMMA_AVAILABLE)
    using namespace ggml_cuda_mma;
    using tile_AB = tile<16, 8, pair_type, get_input_data_layout()>;
    using tile_C = tile<16, 16, float, DATA_LAYOUT_J_MAJOR>;
    constexpr int pitch = 20;
    __shared__ __align__(16) pair_type aa[64][pitch];
    __shared__ __align__(16) pair_type bb[64][pitch];
    const int tid = threadIdx.x + 32 * threadIdx.y;
    const int wave_row = (threadIdx.y % 2) * 32;
    const int wave_col = (threadIdx.y / 2) * 32;
    const int64_t row0 = int64_t(blockIdx.x) * 64;
    const int64_t col0 = int64_t(blockIdx.y) * 64;
    tile_C acc[2][2];
    for (int64_t base = 0; base < k; base += 32) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int index = tid + i * 128;
            const int row = index / 16;
            const int pair = index % 16;
            const int64_t kk = base + 2 * pair;
            pair_type av = ggml_cuda_cast<pair_type>(make_float2(0, 0));
            float2 bv = make_float2(0, 0);
            if (row0 + row < m) {
                const auto * src = reinterpret_cast<const pair_type *>(a + (row0 + row) * lda);
                av = src[kk / 2];
            }
            if (col0 + row < n) {
                const float * src = b + (col0 + row) * ldb + kk;
                bv = make_float2(src[0], src[1]);
            }
            aa[row][pair] = av;
            bb[row][pair] = ggml_cuda_cast<pair_type>(bv);
        }
        __syncthreads();
#pragma unroll
        for (int kp = 0; kp < 16; kp += 8) {
            tile_AB af[2], bf[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                load_ldmatrix(af[i], &aa[wave_row + 16 * i][kp], pitch);
                load_ldmatrix(bf[i], &bb[wave_col + 16 * i][kp], pitch);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    mma(acc[i][j], af[i], bf[j]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
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

template<ggml_type type>
void launch(const ggml_tensor * a, const ggml_tensor * b, ggml_tensor * out, hipStream_t stream) {
    const int64_t m = a->ne[1], n = b->ne[1], k = a->ne[0];
    using pair_type = std::conditional_t<type == GGML_TYPE_BF16, nv_bfloat162, half2>;
    gfx1151_v1_matmul_tiled<pair_type><<<dim3((m + 63) / 64, (n + 63) / 64), dim3(32, 4), 0, stream>>>(
        static_cast<const char *>(a->data), static_cast<const float *>(b->data),
        static_cast<float *>(out->data), m, n, k, a->nb[1], b->nb[1] / sizeof(float));
}

} // namespace

bool ggml_hip_gfx1151_v1_mul_mat_typed(const ggml_tensor * a, const ggml_tensor * b,
                                ggml_tensor * out, hipStream_t stream) {
    if (out->op != GGML_OP_MUL_MAT ||
        ggml_get_op_params_i32(out, 1) != GGML_MUL_MAT_LOWERING_DEFAULT ||
        b->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32 ||
        a->ne[2] != 1 || a->ne[3] != 1 || b->ne[2] != 1 || b->ne[3] != 1 ||
        b->nb[0] != sizeof(float) || !ggml_is_contiguous(out)) {
        return false;
    }
    if (a->type != GGML_TYPE_F16 && a->type != GGML_TYPE_BF16) {
        return false;
    }
    const int64_t m = a->ne[1], n = b->ne[1], k = a->ne[0];
    if (m < 16 || m > INT_MAX || n < 1 || n > 65535 * 64 || k < 32 || k > INT_MAX ||
        k % 32 != 0 || a->nb[0] != ggml_type_size(a->type) ||
        a->nb[1] < ggml_row_size(a->type, k) ||
        b->nb[1] < size_t(k) * sizeof(float) || b->nb[1] % sizeof(float) != 0) {
        return false;
    }
    // Contiguous matrices already have faster HIP GEMV/GEMM implementations.
    if (n < 32 || ggml_get_op_params_i32(out, 0) == GGML_PREC_F32 ||
        (ggml_is_contiguous(a) && ggml_is_contiguous(b)) ||
        reinterpret_cast<std::uintptr_t>(a->data) % alignof(float) != 0 ||
        a->nb[1] % sizeof(float) != 0) {
        return false;
    }
    switch (a->type) {
        case GGML_TYPE_F16:  launch<GGML_TYPE_F16>(a, b, out, stream); break;
        case GGML_TYPE_BF16: launch<GGML_TYPE_BF16>(a, b, out, stream); break;
        default: GGML_ABORT("unreachable gfx1151_v1 matmul type");
    }
    return true;
}
