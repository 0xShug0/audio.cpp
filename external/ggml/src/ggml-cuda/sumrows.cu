#include "reduce_rows.cuh"
#include "sumrows.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// Short rows otherwise launch an entire block for each handful of values.
// Pack multiple rows into each warp while preserving the XOR reduction tree.
template <int columns>
static __global__ void sum_rows_short_f32(const float * __restrict__ x, float * __restrict__ dst,
                                        const int nrows) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t row = index / columns;
    const int column = index % columns;
    float sum = row < nrows ? __fadd_rn(0.0f, x[index]) : 0.0f;
#pragma unroll
    for (int offset = columns / 2; offset > 0; offset /= 2) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset, columns);
    }
    if (column == 0 && row < nrows) dst[row] = sum;
}

static bool sum_rows_short(const float * x, float * dst, int ncols, int nrows, cudaStream_t stream) {
    if (nrows < 16384) return false;
    constexpr int threads = 256;
    const int blocks = (static_cast<int64_t>(ncols) * nrows + threads - 1) / threads;
    switch (ncols) {
        case 2: sum_rows_short_f32<2><<<blocks, threads, 0, stream>>>(x, dst, nrows); break;
        case 4: sum_rows_short_f32<4><<<blocks, threads, 0, stream>>>(x, dst, nrows); break;
        case 8: sum_rows_short_f32<8><<<blocks, threads, 0, stream>>>(x, dst, nrows); break;
        case 16: sum_rows_short_f32<16><<<blocks, threads, 0, stream>>>(x, dst, nrows); break;
        case 32: sum_rows_short_f32<32><<<blocks, threads, 0, stream>>>(x, dst, nrows); break;
        default: return false;
    }
    return true;
}
#endif

void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (sum_rows_short(x, dst, ncols, nrows, stream)) return;
#endif
    const int  id  = ggml_cuda_get_device();
    const int  nsm = ggml_cuda_info().devices[id].nsm;
    const dim3 block_nums(nrows, 1, 1);
    if ((nrows / nsm) < 2) {
        const dim3 block_dims(512, 1, 1);
        reduce_rows_f32</*norm=*/false><<<block_nums, block_dims, 0, stream>>>(x, dst, ncols);
    } else {
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        reduce_rows_f32</*norm=*/false><<<block_nums, block_dims, 0, stream>>>(x, dst, ncols);
    }
}

void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (sum_rows_short(src0_d, dst_d, ncols, nrows, stream)) return;
#endif

    const dim3 block_nums(nrows, 1, 1);

    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;
    if ((nrows / nsm) < 2) {
        // Increase num threads to 512 for small nrows to better hide the latency
        const dim3 block_dims(512, 1, 1);
        reduce_rows_f32</*norm=*/false><<<block_nums, block_dims, 0, stream>>>(src0_d, dst_d, ncols);
    } else {
        // Enough active SMs to hide latency, use smaller blocks to allow better scheduling
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        reduce_rows_f32</*norm=*/false><<<block_nums, block_dims, 0, stream>>>(src0_d, dst_d, ncols);
    }
}
