#include "sage-attn2.cuh"

#include "sage-attn2/fused_kernels.cuh"
#include "sage-attn2/qattn/qk_int_sv_f8_cuda_sm89.cuh"

#include <cuda_fp16.h>

template <int D, int THREADS_N>
static __global__ void sage_attn2_k_mean_f16(
        const half * __restrict__ k,
        half       * __restrict__ mean,
        int nk,
        int nhk) {
    const int d = threadIdx.x;
    const int ty = threadIdx.y;
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    float sum = 0.0f;

    const int64_t head_offset = ((int64_t) b * nhk + h) * nk * D;
    for (int ik = ty; ik < nk; ik += THREADS_N) {
        sum += __half2float(k[head_offset + (int64_t) ik * D + d]);
    }

    __shared__ float partials[THREADS_N][D];
    partials[ty][d] = sum;
    __syncthreads();

    if (ty == 0) {
        float total = 0.0f;
#pragma unroll
        for (int i = 0; i < THREADS_N; ++i) {
            total += partials[i][d];
        }
        mean[((int64_t) b * nhk + h) * D + d] = __float2half_rn(total / nk);
    }
}

template <int D>
static void sage_attn2_quant_q_f16(
        const half * q,
        int8_t     * q_i8,
        float      * q_scale,
        int nq,
        int nhq,
        int nb,
        cudaStream_t stream) {
    constexpr int BLKQ = 128;
    constexpr int WARPQ = 32;
    constexpr int num_pack_per_thread = (WARPQ * (D / 8) + 1023) / 1024;
    const dim3 grid(((nq + BLKQ - 1) / BLKQ) * (BLKQ / WARPQ), nhq, nb);
    const dim3 block(WARPQ * (D / 8) / num_pack_per_thread);

    QuantInt8Kernel<D, WARPQ, num_pack_per_thread, false, false, half><<<grid, block, 0, stream>>>(
        (half *) q,
        nullptr,
        q_i8,
        q_scale,
        0.0f,
        nq,
        nhq * nq * D, D, nq * D,
        0, 0,
        nhq * nq * D, D, nq * D,
        nhq * ((nq + BLKQ - 1) / BLKQ) * (BLKQ / WARPQ),
        ((nq + BLKQ - 1) / BLKQ) * (BLKQ / WARPQ));
}

template <int D>
static void sage_attn2_quant_k_f16(
        const half * k,
        const half * k_mean,
        int8_t     * k_i8,
        float      * k_scale,
        int nk,
        int nhk,
        int nb,
        cudaStream_t stream) {
    constexpr int BLKK = 64;
    constexpr int num_pack_per_thread = (BLKK * (D / 8) + 1023) / 1024;
    const dim3 grid((nk + BLKK - 1) / BLKK, nhk, nb);
    const dim3 block(BLKK * (D / 8) / num_pack_per_thread);

    QuantInt8Kernel<D, BLKK, num_pack_per_thread, false, true, half><<<grid, block, 0, stream>>>(
        (half *) k,
        (half *) k_mean,
        k_i8,
        k_scale,
        0.0f,
        nk,
        nhk * nk * D, D, nk * D,
        nhk * D, D,
        nhk * nk * D, D, nk * D,
        nhk * ((nk + BLKK - 1) / BLKK),
        (nk + BLKK - 1) / BLKK);
}

template <int D>
static void sage_attn2_pack_v_f16(
        const half * v,
        half       * v_transposed,
        int8_t     * v_fp8,
        float      * v_scale,
        int nk,
        int nhk,
        int nb,
        cudaStream_t stream) {
    constexpr int CTA_SIZE = 64;
    const int nk_padded = ((nk + CTA_SIZE - 1) / CTA_SIZE) * CTA_SIZE;

    const dim3 tpp_grid(nk_padded / CTA_SIZE, nhk, nb);
    const dim3 tpp_block(CTA_SIZE * (D / 8));
    TransposePadPermuteKernel<D, CTA_SIZE, true, half><<<tpp_grid, tpp_block, 0, stream>>>(
        (half *) v,
        v_transposed,
        nk,
        nhk * nk * D, D, nk * D,
        nhk * D * nk_padded, nk_padded, D * nk_padded);

    constexpr int SCALE_CTA_SIZE = 256;
    const dim3 scale_grid(nhk, nb, D);
    const dim3 scale_block(SCALE_CTA_SIZE);
    MeanScaleKernel<64, false, half><<<scale_grid, scale_block, 0, stream>>>(
        v_transposed,
        v_fp8,
        nullptr,
        v_scale,
        2.25f,
        nk,
        nhk * D * nk_padded, nk_padded, D * nk_padded,
        nhk * D * nk_padded, nk_padded, D * nk_padded,
        0, 0,
        nhk * D, D);
}

template <int D, bool causal>
static void sage_attn2_attn_f16(
        int8_t * q_i8,
        int8_t * k_i8,
        int8_t * v_fp8,
        half   * dst,
        float  * q_scale,
        float  * k_scale,
        float  * v_scale,
        int nq,
        int nk,
        int nhq,
        int nhk,
        int nb,
        float scale,
        cudaStream_t stream) {
    constexpr int CTA_Q = 128;
    constexpr int CTA_K = 64;
    constexpr int WARP_Q = 32;
    constexpr int WARP_K = 64;
    constexpr MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;

    auto kernel = qk_int_sv_f8_attn_kernel<
        CTA_Q, CTA_K, WARP_Q, WARP_K, D,
        DataType::kInt8,
        QuantGranularity::kPerWarp,
        QuantGranularity::kPerWarp,
        float, true, half, ComputeUnit::kCudaCore,
        mask_mode, false, true, false, true>;

    const size_t smem = std::max(
        CTA_Q * D * sizeof(int8_t) + CTA_K * D * sizeof(int8_t) + CTA_K * D * sizeof(int8_t),
        CTA_Q * D * sizeof(half));
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    const dim3 grid((nq + CTA_Q - 1) / CTA_Q, nhq, nb);
    const dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));
    const int nk_padded = ((nk + CTA_K - 1) / CTA_K) * CTA_K;

    kernel<<<grid, block, smem, stream>>>(
        q_i8,
        k_i8,
        v_fp8,
        dst,
        nullptr,
        q_scale,
        k_scale,
        v_scale,
        nullptr,
        nq,
        nk,
        nhq / nhk,
        nhq * nq * D, D, nq * D,
        nhk * nk * D, D, nk * D,
        nhk * D * nk_padded, D * nk_padded, nk_padded,
        nhq * nq * D, nhq * D, D,
        scale);
}

template <int D>
static void sage_attn2_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    const bool causal = ((const int32_t *) dst->op_params)[1] != 0;

    const int nq  = q->ne[1];
    const int nk  = k->ne[1];
    const int nhq = q->ne[2];
    const int nhk = k->ne[2];
    const int nb  = q->ne[3];
    const int nk_padded = ((nk + 63) / 64) * 64;

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int8_t> q_i8(ctx.pool(), (size_t) nb * nhq * nq * D);
    ggml_cuda_pool_alloc<int8_t> k_i8(ctx.pool(), (size_t) nb * nhk * nk * D);
    ggml_cuda_pool_alloc<half> k_mean(ctx.pool(), (size_t) nb * nhk * D * sizeof(half));
    ggml_cuda_pool_alloc<float> q_scale(ctx.pool(), (size_t) nb * nhq * ((nq + 127) / 128) * 4 * sizeof(float));
    ggml_cuda_pool_alloc<float> k_scale(ctx.pool(), (size_t) nb * nhk * ((nk + 63) / 64) * sizeof(float));
    ggml_cuda_pool_alloc<half> v_transposed(ctx.pool(), (size_t) nb * nhk * D * nk_padded * sizeof(half));
    ggml_cuda_pool_alloc<int8_t> v_fp8(ctx.pool(), (size_t) nb * nhk * D * nk_padded);
    ggml_cuda_pool_alloc<float> v_scale(ctx.pool(), (size_t) nb * nhk * D * sizeof(float));

    constexpr int K_MEAN_THREADS_N = D == 64 ? 8 : 4;
    sage_attn2_k_mean_f16<D, K_MEAN_THREADS_N><<<dim3(nhk, nb), dim3(D, K_MEAN_THREADS_N), 0, stream>>>(
        (const half *) k->data,
        k_mean.get(),
        nk,
        nhk);
    sage_attn2_quant_q_f16<D>((const half *) q->data, q_i8.get(), q_scale.get(), nq, nhq, nb, stream);
    sage_attn2_quant_k_f16<D>((const half *) k->data, k_mean.get(), k_i8.get(), k_scale.get(), nk, nhk, nb, stream);
    sage_attn2_pack_v_f16<D>((const half *) v->data, v_transposed.get(), v_fp8.get(), v_scale.get(), nk, nhk, nb, stream);

    if (causal) {
        sage_attn2_attn_f16<D, true>(
            q_i8.get(), k_i8.get(), v_fp8.get(), (half *) dst->data,
            q_scale.get(), k_scale.get(), v_scale.get(),
            nq, nk, nhq, nhk, nb, scale, stream);
    } else {
        sage_attn2_attn_f16<D, false>(
            q_i8.get(), k_i8.get(), v_fp8.get(), (half *) dst->data,
            q_scale.get(), k_scale.get(), v_scale.get(),
            nq, nk, nhq, nhk, nb, scale, stream);
    }
}

template <int D>
static void sage_attn2_i8_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q_i8    = dst->src[0];
    const ggml_tensor * k_i8    = dst->src[1];
    const ggml_tensor * v       = dst->src[2];
    const ggml_tensor * q_scale = dst->src[3];
    const ggml_tensor * k_scale = dst->src[4];

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    const bool causal = ((const int32_t *) dst->op_params)[1] != 0;

    const int nq  = q_i8->ne[1];
    const int nk  = k_i8->ne[1];
    const int nhq = q_i8->ne[2];
    const int nhk = k_i8->ne[2];
    const int nb  = q_i8->ne[3];
    const int nk_padded = ((nk + 63) / 64) * 64;

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<half> v_transposed(ctx.pool(), (size_t) nb * nhk * D * nk_padded * sizeof(half));
    ggml_cuda_pool_alloc<int8_t> v_fp8(ctx.pool(), (size_t) nb * nhk * D * nk_padded);
    ggml_cuda_pool_alloc<float> v_scale(ctx.pool(), (size_t) nb * nhk * D * sizeof(float));

    sage_attn2_pack_v_f16<D>((const half *) v->data, v_transposed.get(), v_fp8.get(), v_scale.get(), nk, nhk, nb, stream);

    if (causal) {
        sage_attn2_attn_f16<D, true>(
            (int8_t *) q_i8->data, (int8_t *) k_i8->data, v_fp8.get(), (half *) dst->data,
            (float *) q_scale->data, (float *) k_scale->data, v_scale.get(),
            nq, nk, nhq, nhk, nb, scale, stream);
    } else {
        sage_attn2_attn_f16<D, false>(
            (int8_t *) q_i8->data, (int8_t *) k_i8->data, v_fp8.get(), (half *) dst->data,
            (float *) q_scale->data, (float *) k_scale->data, v_scale.get(),
            nq, nk, nhq, nhk, nb, scale, stream);
    }
}

void ggml_cuda_sage_attn2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    GGML_ASSERT(q->type == GGML_TYPE_F16);

    switch (q->ne[0]) {
        case 64:
            sage_attn2_f16_case<64>(ctx, dst);
            break;
        case 128:
            sage_attn2_f16_case<128>(ctx, dst);
            break;
        default:
            GGML_ABORT("unsupported SAGE_ATTN2 head dimension");
    }
}

void ggml_cuda_sage_attn2_i8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q_i8 = dst->src[0];
    GGML_ASSERT(q_i8->type == GGML_TYPE_I8);

    switch (q_i8->ne[0]) {
        case 64:
            sage_attn2_i8_case<64>(ctx, dst);
            break;
        case 128:
            sage_attn2_i8_case<128>(ctx, dst);
            break;
        default:
            GGML_ABORT("unsupported SAGE_ATTN2_I8 head dimension");
    }
}

bool ggml_cuda_sage_attn2_supported(int device, const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const int cc = ggml_cuda_info().devices[device].cc;

    return GGML_CUDA_CC_IS_NVIDIA(cc) &&
        cc >= GGML_CUDA_CC_ADA_LOVELACE &&
        q != nullptr &&
        k != nullptr &&
        v != nullptr &&
        q->type == GGML_TYPE_F16 &&
        k->type == GGML_TYPE_F16 &&
        v->type == GGML_TYPE_F16 &&
        dst->type == GGML_TYPE_F16 &&
        ggml_is_contiguous(q) &&
        ggml_is_contiguous(k) &&
        ggml_is_contiguous(v) &&
        ggml_is_contiguous(dst) &&
        (q->ne[0] == 64 || q->ne[0] == 128) &&
        q->ne[0] == k->ne[0] &&
        q->ne[0] == v->ne[0] &&
        k->ne[1] == v->ne[1] &&
        k->ne[2] == v->ne[2] &&
        q->ne[3] == k->ne[3] &&
        q->ne[3] == v->ne[3] &&
        dst->ne[0] == v->ne[0] &&
        dst->ne[1] == q->ne[2] &&
        dst->ne[2] == q->ne[1] &&
        dst->ne[3] == q->ne[3] &&
        q->ne[2] % k->ne[2] == 0;
}

bool ggml_cuda_sage_attn2_i8_supported(int device, const ggml_tensor * dst) {
    const ggml_tensor * q_i8    = dst->src[0];
    const ggml_tensor * k_i8    = dst->src[1];
    const ggml_tensor * v       = dst->src[2];
    const ggml_tensor * q_scale = dst->src[3];
    const ggml_tensor * k_scale = dst->src[4];
    const int cc = ggml_cuda_info().devices[device].cc;

    return GGML_CUDA_CC_IS_NVIDIA(cc) &&
        cc >= GGML_CUDA_CC_ADA_LOVELACE &&
        q_i8 != nullptr &&
        k_i8 != nullptr &&
        v != nullptr &&
        q_scale != nullptr &&
        k_scale != nullptr &&
        q_i8->type == GGML_TYPE_I8 &&
        k_i8->type == GGML_TYPE_I8 &&
        v->type == GGML_TYPE_F16 &&
        q_scale->type == GGML_TYPE_F32 &&
        k_scale->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F16 &&
        ggml_is_contiguous(q_i8) &&
        ggml_is_contiguous(k_i8) &&
        ggml_is_contiguous(v) &&
        ggml_is_contiguous(q_scale) &&
        ggml_is_contiguous(k_scale) &&
        ggml_is_contiguous(dst) &&
        (q_i8->ne[0] == 64 || q_i8->ne[0] == 128) &&
        q_i8->ne[0] == k_i8->ne[0] &&
        q_i8->ne[0] == v->ne[0] &&
        k_i8->ne[1] == v->ne[1] &&
        k_i8->ne[2] == v->ne[2] &&
        q_i8->ne[3] == k_i8->ne[3] &&
        q_i8->ne[3] == v->ne[3] &&
        dst->ne[0] == v->ne[0] &&
        dst->ne[1] == q_i8->ne[2] &&
        dst->ne[2] == q_i8->ne[1] &&
        dst->ne[3] == q_i8->ne[3] &&
        q_i8->ne[2] % k_i8->ne[2] == 0 &&
        q_scale->ne[0] == ((q_i8->ne[1] + 127) / 128) * 4 &&
        q_scale->ne[1] == q_i8->ne[2] &&
        q_scale->ne[2] == q_i8->ne[3] &&
        k_scale->ne[0] == (k_i8->ne[1] + 63) / 64 &&
        k_scale->ne[1] == k_i8->ne[2] &&
        k_scale->ne[2] == k_i8->ne[3];
}
