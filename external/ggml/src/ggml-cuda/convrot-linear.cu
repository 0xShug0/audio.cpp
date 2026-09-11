#include "convrot-linear.cuh"

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

void ggml_cuda_convrot_linear(ggml_backend_cuda_context &, ggml_tensor *) {
    GGML_ABORT("convrot_linear is only implemented for CUDA");
}

bool ggml_cuda_convrot_linear_supported(int, const ggml_tensor *) {
    return false;
}

#else

namespace {

__device__ __forceinline__ float h4_row_dot(int d, float x0, float x1, float x2, float x3) {
    switch (d) {
        case 0:  return  x0 + x1 + x2 - x3;
        case 1:  return  x0 + x1 - x2 + x3;
        case 2:  return  x0 - x1 + x2 + x3;
        default: return -x0 + x1 + x2 + x3;
    }
}

__device__ __forceinline__ float warp_reduce_max_f32(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

template<int NUM_WARPS>
__device__ __forceinline__ float block_reduce_max_f32(float value, float * warp_smem, float * block_smem) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    value = warp_reduce_max_f32(value);
    if (lane == 0) {
        warp_smem[wid] = value;
    }
    __syncthreads();
    if (wid == 0) {
        float total = lane < NUM_WARPS ? warp_smem[lane] : 0.0f;
        total = warp_reduce_max_f32(total);
        if (lane == 0) {
            *block_smem = total;
        }
    }
    __syncthreads();
    return *block_smem;
}

template<int BLOCK_THREADS>
__global__ void convrot_quantize_activation_fused_f32(
        const float * __restrict__ x,
        int8_t      * __restrict__ qx,
        float       * __restrict__ x_scale,
        int tokens,
        int k) {
    constexpr int group = 256;
    constexpr int groups_in_flight = BLOCK_THREADS / group;
    constexpr int num_warps = BLOCK_THREADS / 32;

    extern __shared__ float smem[];
    float * row = smem;
    float * tmp = smem + k;

    __shared__ float warp_smem[num_warps];
    __shared__ float block_smem;

    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    const int row_offset = token * k;

    for (int col = tid; col < k; col += BLOCK_THREADS) {
        row[col] = x[row_offset + col];
    }
    __syncthreads();

    const int n_groups = k / group;
    const int sub = tid / group;
    const int lane = tid % group;
    float * buf0 = tmp + sub * (2 * group);
    float * buf1 = buf0 + group;
    const int iters = (n_groups + groups_in_flight - 1) / groups_in_flight;
    for (int it = 0; it < iters; ++it) {
        const int g = it * groups_in_flight + sub;
        const bool active = g < n_groups;
        float * src = active ? row + g * group : buf0;
        float * dst = active ? buf0 : buf1;
        for (int stage = 0; stage < 4; ++stage) {
            const int stride = stage == 0 ? 1 : stage == 1 ? 4 : stage == 2 ? 16 : 64;
            const int d = (lane / stride) & 3;
            const int local_base = lane - d * stride;
            dst[lane] = 0.5f * h4_row_dot(
                d,
                src[local_base],
                src[local_base + stride],
                src[local_base + 2 * stride],
                src[local_base + 3 * stride]);
            __syncthreads();
            float * next = src;
            src = dst;
            dst = next;
        }
    }

    float max_abs = 0.0f;
    for (int col = tid; col < k; col += BLOCK_THREADS) {
        max_abs = fmaxf(max_abs, fabsf(row[col]));
    }
    const float scale = fmaxf(block_reduce_max_f32<num_warps>(max_abs, warp_smem, &block_smem) / 127.0f, 1.0e-8f);
    if (tid == 0) {
        x_scale[token] = scale;
    }
    for (int col = tid; col < k; col += BLOCK_THREADS) {
        const int q = max(-128, min(127, __float2int_rn(row[col] / scale)));
        qx[row_offset + col] = static_cast<int8_t>(q);
    }
}

__global__ void convrot_scale_i32_output(
        const int32_t * __restrict__ acc,
        const float   * __restrict__ weight_scale,
        const float   * __restrict__ x_scale,
        const float   * __restrict__ bias,
        float         * __restrict__ out,
        int chunk_n,
        int dst_n,
        int row_offset,
        int tokens) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = chunk_n * tokens;
    if (i >= total) {
        return;
    }

    const int row = i % chunk_n;
    const int token = i / chunk_n;
    const int dst_row = row_offset + row;
    float value = static_cast<float>(acc[i]) * weight_scale[dst_row] * x_scale[token];
    if (bias != nullptr) {
        value += bias[dst_row];
    }
    out[token * dst_n + dst_row] = value;
}

} // namespace

void ggml_cuda_convrot_linear(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * weight_i8 = dst->src[0];
    const ggml_tensor * input = dst->src[1];
    const ggml_tensor * weight_scale = dst->src[2];
    const ggml_tensor * bias = dst->src[3];

    const int group_size = ((const int32_t *) dst->op_params)[0];
    GGML_ASSERT(group_size == 256);
    GGML_ASSERT(weight_i8->type == GGML_TYPE_I8);
    GGML_ASSERT(input->type == GGML_TYPE_F32);
    GGML_ASSERT(weight_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int k = static_cast<int>(weight_i8->ne[0]);
    const int n = static_cast<int>(weight_i8->ne[1]);
    const int tokens = static_cast<int>(input->ne[1] * input->ne[2] * input->ne[3]);
    GGML_ASSERT(k % group_size == 0);
    GGML_ASSERT(tokens > 0);

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int8_t> qx(ctx.pool(), static_cast<size_t>(tokens) * k);
    ggml_cuda_pool_alloc<float> x_scale(ctx.pool(), static_cast<size_t>(tokens));

    const bool wide = k > 5120;
    const int block_threads = wide ? 1024 : 256;
    const int groups_in_flight = block_threads / group_size;
    const size_t smem_bytes =
        (static_cast<size_t>(k) + static_cast<size_t>(groups_in_flight) * 2 * group_size) * sizeof(float);
    if (wide) {
        CUDA_CHECK(cudaFuncSetAttribute(
            convrot_quantize_activation_fused_f32<1024>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));
        convrot_quantize_activation_fused_f32<1024><<<tokens, 1024, smem_bytes, stream>>>(
            static_cast<const float *>(input->data),
            qx.get(),
            x_scale.get(),
            tokens,
            k);
    } else {
        CUDA_CHECK(cudaFuncSetAttribute(
            convrot_quantize_activation_fused_f32<256>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));
        convrot_quantize_activation_fused_f32<256><<<tokens, 256, smem_bytes, stream>>>(
            static_cast<const float *>(input->data),
            qx.get(),
            x_scale.get(),
            tokens,
            k);
    }
    CUDA_CHECK(cudaGetLastError());

    const int32_t alpha = 1;
    const int32_t beta = 0;
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
    constexpr int threads = 256;
    constexpr int max_chunk_n = 16384;
    constexpr int chunk_multiple = 256;
    constexpr size_t max_acc_bytes = 512ull * 1024ull * 1024ull;
    size_t max_rows_by_workspace = max_acc_bytes / (static_cast<size_t>(tokens) * sizeof(int32_t));
    if (max_rows_by_workspace == 0) {
        max_rows_by_workspace = 1;
    }
    int chunk_n = std::min<int>(max_chunk_n, std::min<int>(n, static_cast<int>(max_rows_by_workspace)));
    if (chunk_n > chunk_multiple) {
        chunk_n = (chunk_n / chunk_multiple) * chunk_multiple;
    }
    chunk_n = std::max(1, chunk_n);
    ggml_cuda_pool_alloc<int32_t> acc(ctx.pool(), static_cast<size_t>(tokens) * chunk_n);
    for (int row_offset = 0; row_offset < n; row_offset += chunk_n) {
        const int rows = std::min(chunk_n, n - row_offset);
        CUBLAS_CHECK(cublasGemmEx(
            ctx.cublas_handle(),
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            rows,
            tokens,
            k,
            &alpha,
            static_cast<const int8_t *>(weight_i8->data) + static_cast<size_t>(row_offset) * k,
            CUDA_R_8I,
            k,
            qx.get(),
            CUDA_R_8I,
            k,
            &beta,
            acc.get(),
            CUDA_R_32I,
            rows,
            CUBLAS_COMPUTE_32I,
            CUBLAS_GEMM_DEFAULT));

        convrot_scale_i32_output<<<(tokens * rows + threads - 1) / threads, threads, 0, stream>>>(
            acc.get(),
            static_cast<const float *>(weight_scale->data),
            x_scale.get(),
            bias == nullptr ? nullptr : static_cast<const float *>(bias->data),
            static_cast<float *>(dst->data),
            rows,
            n,
            row_offset,
            tokens);
        CUDA_CHECK(cudaGetLastError());
    }
}

bool ggml_cuda_convrot_linear_supported(int device, const ggml_tensor * dst) {
    const ggml_tensor * weight_i8 = dst->src[0];
    const ggml_tensor * input = dst->src[1];
    const ggml_tensor * weight_scale = dst->src[2];
    const ggml_tensor * bias = dst->src[3];
    const int group_size = ((const int32_t *) dst->op_params)[0];
    const int cc = ggml_cuda_info().devices[device].cc;
    return GGML_CUDA_CC_IS_NVIDIA(cc) &&
           cc >= GGML_CUDA_CC_TURING &&
           group_size == 256 &&
           dst->type == GGML_TYPE_F32 &&
           weight_i8 != nullptr &&
           input != nullptr &&
           weight_scale != nullptr &&
           weight_i8->type == GGML_TYPE_I8 &&
           input->type == GGML_TYPE_F32 &&
           weight_scale->type == GGML_TYPE_F32 &&
           (bias == nullptr || bias->type == GGML_TYPE_F32) &&
           ggml_is_contiguous(weight_i8) &&
           ggml_is_contiguous(input) &&
           ggml_is_contiguous(weight_scale) &&
           (bias == nullptr || ggml_is_contiguous(bias)) &&
           weight_i8->ne[0] == input->ne[0] &&
           weight_i8->ne[0] % group_size == 0 &&
           ggml_nelements(weight_scale) == weight_i8->ne[1] &&
           (bias == nullptr || bias->ne[0] == weight_i8->ne[1]);
}

#endif
