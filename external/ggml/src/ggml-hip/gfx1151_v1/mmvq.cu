#include "mmvq.cuh"
#include "ggml-cuda/vecdotq.cuh"
#include <climits>
#include <type_traits>

namespace {
using q8_pair = int __attribute__((vector_size(8), aligned(2), may_alias));

template<int mode, int rows, int cols, int unroll, bool tail, bool batched>
#ifdef __HIP_DEVICE_COMPILE__
__attribute__((target("unaligned-access-mode")))
#endif
__global__ __launch_bounds__(128) void ar_q8(
        const block_q8_0 * a, const block_q8_1 * b, float * dst,
        int m, int n, int k, int64_t stride_a, int64_t stride_b,
        int64_t sa2, int64_t sa3, uint3 ne2, uint3 r2, uint3 r3) {
    if constexpr (batched) {
        const uint32_t i3 = fastdiv(blockIdx.z, ne2);
        const uint32_t i2 = blockIdx.z - i3 * ne2.z;
        a += int64_t(fastdiv(i2, r2)) * sa2 + int64_t(fastdiv(i3, r3)) * sa3;
        b += int64_t(blockIdx.z) * n * stride_b;
        dst += int64_t(blockIdx.z) * n * m;
    }
    b += int64_t(blockIdx.y) * cols * stride_b;
    dst += int64_t(blockIdx.y) * cols * m;
    const int lane = threadIdx.x;
    const int row = (blockIdx.x * 4 + threadIdx.y) * rows;
    const int quant = 2 * (lane % 4);
    if constexpr (unroll > 0) {
        if (row >= m) return;
    }
    float sum[cols][rows] = {};
#pragma clang loop unroll_count(unroll > 0 ? unroll : 1)
    for (int block = lane / 4; block < k / QK8_0; block += 8) {
        int u[cols][2];
        float scale_b[cols];
#pragma unroll
        for (int c = 0; c < cols; ++c) {
            if constexpr (tail) {
                if (int64_t(blockIdx.y) * cols + c >= n) continue;
            }
            const auto & x = b[c * stride_b + block];
            u[c][0] = get_int_b4(x.qs, quant);
            u[c][1] = get_int_b4(x.qs, quant + 1);
            scale_b[c] = __half2float(__low2half(x.ds));
        }
#pragma unroll
        for (int r = 0; r < rows; ++r) {
            if constexpr (unroll == 0 || rows > 1) {
                if (row + r >= m) continue;
            }
            const auto & w = a[(row + r) * stride_a + block];
            int v[2];
            if constexpr (mode == 0) {
                v[0] = get_int_b2(w.qs, quant);
                v[1] = get_int_b2(w.qs, quant + 1);
            } else if constexpr (mode == 1) {
                __builtin_memcpy(v, w.qs + quant * 4, sizeof(v));
            } else {
                const q8_pair value = __builtin_nontemporal_load(
                    reinterpret_cast<const q8_pair *>(w.qs + quant * 4));
                v[0] = value[0]; v[1] = value[1];
            }
#pragma unroll
            for (int c = 0; c < cols; ++c) {
                if constexpr (tail) {
                    if (int64_t(blockIdx.y) * cols + c >= n) continue;
                }
                sum[c][r] += vec_dot_q8_0_q8_1_impl<float, 2>(v, u[c], __half2float(w.d), scale_b[c]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < cols; ++c) {
        if constexpr (tail) {
            if (int64_t(blockIdx.y) * cols + c >= n) continue;
        }
#pragma unroll
        for (int r = 0; r < rows; ++r) {
            const float value = warp_reduce_sum<32>(sum[c][r]);
            if (lane == 0 && row + r < m) dst[c * m + row + r] = value;
        }
    }
}
} // namespace

template<int mode, int rows, int unroll>
bool ggml_hip_ar_mmvq(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * ids,
                     const void * q8, const ggml_cuda_mm_fusion_args_device & fusion,
                     ggml_tensor * dst, hipStream_t stream) {
    if (a->type != GGML_TYPE_Q8_0 || ids || a->ne[0] != b->ne[0] || b->ne[2] % a->ne[2] != 0 || b->ne[3] % a->ne[3] != 0 ||
        !ggml_is_contiguous(dst) || fusion.gate || fusion.x_bias || fusion.gate_bias || fusion.residual_only) return false;
    if (a->ne[0] > INT_MAX || a->ne[1] > INT_MAX || b->ne[1] < 1 || b->ne[1] > 2 * 65535 ||
        b->ne[2] > 65535 || b->ne[3] > 65535 / b->ne[2] ||
        a->nb[1] % sizeof(block_q8_0) ||
        (a->ne[2] > 1 && a->nb[2] % sizeof(block_q8_0)) ||
        (a->ne[3] > 1 && a->nb[3] % sizeof(block_q8_0))) {
        return false;
    }
    const auto dispatch = [&](auto batched_tag) {
        constexpr bool batched = decltype(batched_tag)::value;
        const auto launch = [&](auto cols_tag, auto tail_tag) {
            constexpr int cols = decltype(cols_tag)::value;
            constexpr bool tail = decltype(tail_tag)::value;
            const dim3 grid((a->ne[1] + 4 * rows - 1) / (4 * rows),
                            (b->ne[1] + cols - 1) / cols, b->ne[2] * b->ne[3]);
            ar_q8<mode, rows, cols, unroll, tail, batched><<<grid, dim3(32, 4), 0, stream>>>(
                static_cast<const block_q8_0 *>(a->data), static_cast<const block_q8_1 *>(q8),
                static_cast<float *>(dst->data), a->ne[1], b->ne[1], a->ne[0],
                a->nb[1] / sizeof(block_q8_0), GGML_PAD(b->ne[0], MATRIX_ROW_PADDING) / QK8_1,
                a->nb[2] / sizeof(block_q8_0), a->nb[3] / sizeof(block_q8_0),
                init_fastdiv_values(b->ne[2]), init_fastdiv_values(b->ne[2] / a->ne[2]),
                init_fastdiv_values(b->ne[3] / a->ne[3]));
        };
        if (b->ne[1] == 1) launch(std::integral_constant<int, 1>{}, std::false_type{});
        else if (b->ne[1] % 2 == 0) launch(std::integral_constant<int, 2>{}, std::false_type{});
        else launch(std::integral_constant<int, 2>{}, std::true_type{});
    };
    if (b->ne[2] * b->ne[3] > 1) dispatch(std::true_type{});
    else dispatch(std::false_type{});
    return true;
}

template bool ggml_hip_ar_mmvq<0, 1>(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const void *, const ggml_cuda_mm_fusion_args_device &, ggml_tensor *, hipStream_t);
template bool ggml_hip_ar_mmvq<1, 1>(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const void *, const ggml_cuda_mm_fusion_args_device &, ggml_tensor *, hipStream_t);
template bool ggml_hip_ar_mmvq<1, 2>(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const void *, const ggml_cuda_mm_fusion_args_device &, ggml_tensor *, hipStream_t);
template bool ggml_hip_ar_mmvq<2, 1>(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const void *, const ggml_cuda_mm_fusion_args_device &, ggml_tensor *, hipStream_t);
template bool ggml_hip_ar_mmvq<2, 1, 1>(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const void *, const ggml_cuda_mm_fusion_args_device &, ggml_tensor *, hipStream_t);
template bool ggml_hip_ar_mmvq<2, 1, 2>(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const void *, const ggml_cuda_mm_fusion_args_device &, ggml_tensor *, hipStream_t);
template bool ggml_hip_ar_mmvq<2, 1, 4>(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const void *, const ggml_cuda_mm_fusion_args_device &, ggml_tensor *, hipStream_t);
