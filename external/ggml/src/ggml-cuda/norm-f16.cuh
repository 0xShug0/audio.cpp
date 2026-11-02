#pragma once

#include "common.cuh"

// F16 output is isolated from the established F32 normalization kernels.
template <typename T, int block_size>
static __global__ void rms_norm_channels_silu_f16(
        const T *     x,
        const float * gamma,
        half *        dst,
        int64_t       ne0,
        int64_t       ne1,
        int64_t       ne2,
        int64_t       ne3,
        int64_t       nb0,
        int64_t       nb1,
        int64_t       nb2,
        int64_t       nb3,
        int64_t       dnb0,
        int64_t       dnb1,
        int64_t       dnb2,
        int64_t       dnb3,
        int64_t       gnb0,
        const float   eps) {
    const int64_t row = int64_t(blockIdx.x);
    const int tid = threadIdx.x;
    if (row >= ne0 * ne1 * ne2) {
        return;
    }

    const int64_t i0 = row % ne0;
    const int64_t tmp0 = row / ne0;
    const int64_t i1 = tmp0 % ne1;
    const int64_t i2 = tmp0 / ne1;
    const char * x_base = reinterpret_cast<const char *>(x) + i0 * nb0 + i1 * nb1 + i2 * nb2;
    char * dst_base = reinterpret_cast<char *>(dst) + i0 * dnb0 + i1 * dnb1 + i2 * dnb2;
    const char * gamma_base = reinterpret_cast<const char *>(gamma);

    float sum = 0.0f;
    for (int64_t c = tid; c < ne3; c += block_size) {
        const float v = float(*reinterpret_cast<const T *>(x_base + c * nb3));
        sum += v * v;
    }

    extern __shared__ float s_sum[];
    sum = block_reduce<block_reduce_method::SUM, block_size>(sum, s_sum);
    const float scale = rsqrtf(sum / ne3 + eps);

    for (int64_t c = tid; c < ne3; c += block_size) {
        const float v = float(*reinterpret_cast<const T *>(x_base + c * nb3));
        const float g = *reinterpret_cast<const float *>(gamma_base + c * gnb0);
        const float normalized = v * scale * g;
        *reinterpret_cast<half *>(dst_base + c * dnb3) = __float2half_rn(normalized / (1.0f + expf(-normalized)));
    }
}

template <typename T, int rows_per_block, int channels_per_block>
static __global__ void rms_norm_channels_silu_coalesced_f16(
        const T *     x,
        const float * gamma,
        half *        dst,
        int64_t       rows,
        int64_t       channels,
        const float   eps) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int64_t row = int64_t(blockIdx.x) * rows_per_block + lane;
    const bool row_valid = row < rows;

    __shared__ float partial[channels_per_block][rows_per_block];
    __shared__ float scale_s[rows_per_block];

    float sum = 0.0f;
    for (int64_t c = warp; c < channels; c += channels_per_block) {
        const float v = row_valid ? float(x[c * rows + row]) : 0.0f;
        sum += v * v;
    }
    partial[warp][lane] = sum;
    __syncthreads();

    if (warp == 0) {
        float row_sum = 0.0f;
#pragma unroll
        for (int c = 0; c < channels_per_block; ++c) {
            row_sum += partial[c][lane];
        }
        scale_s[lane] = rsqrtf(row_sum / channels + eps);
    }
    __syncthreads();

    if (!row_valid) {
        return;
    }
    const float scale = scale_s[lane];
    for (int64_t c = warp; c < channels; c += channels_per_block) {
        const float v = float(x[c * rows + row]);
        const float normalized = v * scale * gamma[c];
        dst[c * rows + row] = __float2half_rn(normalized / (1.0f + expf(-normalized)));
    }
}

template <typename T>
static void rms_norm_channels_silu_f16_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * gamma = dst->src[1];
    GGML_ASSERT(gamma->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F16);
    GGML_ASSERT(gamma->ne[0] == src0->ne[3]);
    GGML_ASSERT(ggml_is_contiguous(dst));

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);
    const int64_t rows = src0->ne[0] * src0->ne[1] * src0->ne[2];
    cudaStream_t stream = ctx.stream();
    if (ggml_get_op_params_i32(dst, 1) == GGML_RMS_NORM_CHANNELS_LOWERING_CUDA_COALESCED &&
        ggml_is_contiguous(src0) && gamma->nb[0] == sizeof(float) &&
        src0->nb[0] == sizeof(T) && dst->nb[0] == sizeof(half) && src0->ne[3] <= 512) {
        constexpr int rows_per_block = 32;
        constexpr int channels_per_block = 8;
        const int64_t blocks = (rows + rows_per_block - 1) / rows_per_block;
        rms_norm_channels_silu_coalesced_f16<T, rows_per_block, channels_per_block>
            <<<blocks, rows_per_block * channels_per_block, 0, stream>>>(
                (const T *) src0->data, (const float *) gamma->data, (half *) dst->data, rows, src0->ne[3], eps);
    } else {
        rms_norm_channels_silu_f16<T, 256><<<rows, 256, 32 * sizeof(float), stream>>>(
            (const T *) src0->data, (const float *) gamma->data, (half *) dst->data,
            src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], gamma->nb[0], eps);
    }
}
