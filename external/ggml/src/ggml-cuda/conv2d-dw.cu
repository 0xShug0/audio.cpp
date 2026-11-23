#include "conv2d-dw.cuh"

struct conv_params {
    int in_w, in_h;
    int out_w, out_h;
    int kernel_w, kernel_h;
    int stride_x, stride_y;
    int padding_x, padding_y;
    int dilation_x, dilation_y;
    int channels, batches;
};

struct kernel_bounds {
    int y_min, y_max;
    int x_min, x_max;
};

__device__ __forceinline__ kernel_bounds calculate_kernel_bounds(int out_x, int out_y, const conv_params & params) {
    kernel_bounds bounds;
    bounds.y_min = max(0, (params.padding_y - out_y * params.stride_y + params.dilation_y - 1) / params.dilation_y);
    bounds.y_max =
        min(params.kernel_h,
            (params.in_h + params.padding_y - out_y * params.stride_y + params.dilation_y - 1) / params.dilation_y);
    bounds.x_min = max(0, (params.padding_x - out_x * params.stride_x + params.dilation_x - 1) / params.dilation_x);
    bounds.x_max =
        min(params.kernel_w,
            (params.in_w + params.padding_x - out_x * params.stride_x + params.dilation_x - 1) / params.dilation_x);
    return bounds;
}

__device__ __forceinline__ int calculate_input_coord(int out_coord, int kern_coord, int stride, int dilation, int padding) {
    return out_coord * stride + kern_coord * dilation - padding;
}

struct whcn_layout {
    __device__ static int input_index(int n, int c, int y, int x, const conv_params & params) {
        return n * (params.channels * params.in_w * params.in_h) + c * params.in_w * params.in_h + y * params.in_w + x;
    }

    __device__ static int kernel_index(int c, int ky, int kx, const conv_params & params) {
        return c * params.kernel_h * params.kernel_w + ky * params.kernel_w + kx;
    }

    __device__ static int output_index(int n, int c, int y, int x, const conv_params & params) {
        return n * (params.channels * params.out_w * params.out_h) + c * params.out_w * params.out_h +
               y * params.out_w + x;
    }

    __device__ static void unpack_indices(int global_idx, const conv_params & params, int & n, int & c, int & out_y,
                                          int & out_x) {
        out_x = global_idx % params.out_w;
        out_y = (global_idx / params.out_w) % params.out_h;
        c     = (global_idx / (params.out_w * params.out_h)) % params.channels;
        n     = global_idx / (params.out_w * params.out_h * params.channels);
    }
};

struct cwhn_layout {
    __device__ static int input_index(int n, int c, int y, int x, const conv_params & params) {
        return n * (params.channels * params.in_w * params.in_h) + (y * params.in_w + x) * params.channels + c;
    }

    __device__ static int kernel_index(int c, int ky, int kx, const conv_params & params) {
        return (ky * params.kernel_w + kx) * params.channels + c;
    }

    __device__ static int output_index(int n, int c, int y, int x, const conv_params & params) {
        return n * (params.channels * params.out_w * params.out_h) + y * (params.out_w * params.channels) +
               x * params.channels + c;
    }

    __device__ static void unpack_indices(int global_idx, const conv_params & params, int & n, int & c, int & out_y,
                                          int & out_x) {
        c     = global_idx % params.channels;
        out_x = (global_idx / params.channels) % params.out_w;
        out_y = (global_idx / (params.channels * params.out_w)) % params.out_h;
        n     = global_idx / (params.channels * params.out_w * params.out_h);
    }
};

template <typename T, typename Layout>
__global__ void conv2d_dw_kernel(const T * __restrict__ input, const T * __restrict__ kernel, T * __restrict__ output,
                                 const int in_w, const int in_h, const int out_w, const int out_h,
                                 const int kernel_w, const int kernel_h, const int stride_x, const int stride_y,
                                 const int padding_x, const int padding_y, const int dilation_x, const int dilation_y,
                                 const int channels, const int batches) {
    const int global_idx     = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batches * channels * out_h * out_w;

    if (global_idx >= total_elements) {
        return;
    }

    conv_params params = { in_w,     in_h,      out_w,     out_h,      kernel_w,   kernel_h, stride_x,
                           stride_y, padding_x, padding_y, dilation_x, dilation_y, channels, batches };

    int batch_idx, channel_idx, out_y_idx, out_x_idx;
    Layout::unpack_indices(global_idx, params, batch_idx, channel_idx, out_y_idx, out_x_idx);

    T accumulator = 0;
    kernel_bounds bounds = calculate_kernel_bounds(out_x_idx, out_y_idx, params);

    for (int kern_y = bounds.y_min; kern_y < bounds.y_max; ++kern_y) {
        int in_y_idx = calculate_input_coord(out_y_idx, kern_y, params.stride_y, params.dilation_y, params.padding_y);

        for (int kern_x = bounds.x_min; kern_x < bounds.x_max; ++kern_x) {
            int in_x_idx = calculate_input_coord(out_x_idx, kern_x, params.stride_x, params.dilation_x, params.padding_x);

            const T input_val  = input[Layout::input_index(batch_idx, channel_idx, in_y_idx, in_x_idx, params)];
            const T kernel_val = kernel[Layout::kernel_index(channel_idx, kern_y, kern_x, params)];

            accumulator += input_val * kernel_val;
        }
    }

    output[Layout::output_index(batch_idx, channel_idx, out_y_idx, out_x_idx, params)] = accumulator;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// A warp owns a temporal row. Four adjacent columns reuse each tap and its
// row address; accumulation order remains the ordinary direct convolution's.
template<bool vectorized>
static __global__ void conv2d_dw_rows(const float *input, const float *weight, float *output,
        int width, int height, int out_height, int channels, int batches,
        int taps, int stride, int padding, int dilation) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int64_t rows = int64_t(out_height) * channels * batches;
    for (int64_t row = int64_t(blockIdx.x) * 4 + warp; row < rows; row += int64_t(gridDim.x) * 4) {
        const int y = row % out_height, channel = row / out_height % channels;
        const int64_t plane = row / out_height;
        const int first = max(0, (padding - y * stride + dilation - 1) / dilation);
        const int last = min(taps, (height + padding - y * stride + dilation - 1) / dilation);
        for (int x = lane * 4; x < width; x += 128) {
            float4 sum = make_float4(0, 0, 0, 0);
            for (int k = first; k < last; ++k) {
                const auto *source = input + (plane * height + y * stride + k * dilation - padding) * width + x;
                float4 v;
                if constexpr (vectorized) v = *reinterpret_cast<const float4 *>(source);
                else v = make_float4(source[0], x + 1 < width ? source[1] : 0,
                    x + 2 < width ? source[2] : 0, x + 3 < width ? source[3] : 0);
                const float w = weight[channel * taps + k];
                sum.x += v.x * w; sum.y += v.y * w; sum.z += v.z * w; sum.w += v.w * w;
            }
            auto *target = output + row * width + x;
            if constexpr (vectorized) *reinterpret_cast<float4 *>(target) = sum;
            else {
                target[0] = sum.x;
                if (x + 1 < width) target[1] = sum.y;
                if (x + 2 < width) target[2] = sum.z;
                if (x + 3 < width) target[3] = sum.w;
            }
        }
    }
}
#endif

void ggml_cuda_op_conv2d_dw(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F32 && input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    const float * w_d = (const float *) kernel->data;
    const float * x_d = (const float *) input->data;
    float *       y_d = (float *) dst->data;

    const int32_t * p          = (const int32_t *) dst->op_params;
    const int       stride_x   = p[0];
    const int       stride_y   = p[1];
    const int       padding_x  = p[2];
    const int       padding_y  = p[3];
    const int       dilation_x = p[4];
    const int       dilation_y = p[5];

    const int in_w     = input->ne[0];
    const int in_h     = input->ne[1];
    const int kernel_w = kernel->ne[0];
    const int kernel_h = kernel->ne[1];
    const int out_w    = dst->ne[0];
    const int out_h    = dst->ne[1];
    const int channels = dst->ne[2];
    const int batches  = dst->ne[3];

    cudaStream_t st = ctx.stream();

    const int total  = batches * channels * out_h * out_w;
    const int blocks = (total + CUDA_CONV2D_DW_BLOCK_SIZE - 1) / CUDA_CONV2D_DW_BLOCK_SIZE;

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (ggml_get_op_params_i32(dst, 6) == GGML_CONV_2D_DW_LOWERING_CUDA_ROWS) {
        GGML_ASSERT(ggml_is_contiguous(input) && ggml_is_contiguous(kernel) && ggml_is_contiguous(dst));
        GGML_ASSERT(kernel_w == 1 && stride_x == 1 && padding_x == 0 && in_w == out_w);
        const int64_t rows = int64_t(out_h) * channels * batches;
        const int row_blocks = int(std::min<int64_t>((rows + 3) / 4, 65535));
        const bool vectorized = in_w % 4 == 0 && reinterpret_cast<uintptr_t>(x_d) % 16 == 0 && reinterpret_cast<uintptr_t>(y_d) % 16 == 0;
        if (vectorized)
            conv2d_dw_rows<true><<<row_blocks, 128, 0, st>>>(x_d, w_d, y_d, in_w, in_h, out_h, channels, batches, kernel_h, stride_y, padding_y, dilation_y);
        else
            conv2d_dw_rows<false><<<row_blocks, 128, 0, st>>>(x_d, w_d, y_d, in_w, in_h, out_h, channels, batches, kernel_h, stride_y, padding_y, dilation_y);
        return;
    }
#endif

    if (ggml_is_contiguous(input)) {
        conv2d_dw_kernel<float, whcn_layout><<<blocks, CUDA_CONV2D_DW_BLOCK_SIZE, 0, st>>>(
            x_d, w_d, y_d, in_w, in_h, out_w, out_h, kernel_w, kernel_h, stride_x, stride_y, padding_x, padding_y,
            dilation_x, dilation_y, channels, batches);
    } else if (ggml_is_contiguous_channels(input)) {
        conv2d_dw_kernel<float, cwhn_layout><<<blocks, CUDA_CONV2D_DW_BLOCK_SIZE, 0, st>>>(
            x_d, w_d, y_d, in_w, in_h, out_w, out_h, kernel_w, kernel_h, stride_x, stride_y, padding_x, padding_y,
            dilation_x, dilation_y, channels, batches);
    } else {
        GGML_ABORT("Unsupported memory layout for conv_2d_dw");
    }
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// A warp owns a row; taps are supplied data, not model-specific kernel shapes.
template<bool vectorized>
static __global__ void conv2d_dw_sparse(const float *input, const float *weights,
        const int32_t *offsets, float *output, int64_t width, int64_t height,
        int64_t channels, int64_t rows, int64_t taps) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int64_t row = int64_t(blockIdx.x) * 4 + warp; row < rows; row += int64_t(gridDim.x) * 4) {
        const int64_t y = row % height, plane = row / height, channel = plane % channels;
        for (int64_t x = lane * 4; x < width; x += 128) {
            float4 sum = make_float4(0, 0, 0, 0);
            for (int64_t k = 0; k < taps; ++k) {
                const int64_t sy = y + offsets[2*k+1], sx = x + offsets[2*k];
                if (sy < 0 || sy >= height) continue;
                const float w = weights[channel * taps + k];
                const float *source = input + (plane * height + sy) * width;
                if (sx >= 0 && sx < width) sum.x = __fmaf_rn(source[sx], w, sum.x);
                if (sx+1 >= 0 && sx+1 < width) sum.y = __fmaf_rn(source[sx+1], w, sum.y);
                if (sx+2 >= 0 && sx+2 < width) sum.z = __fmaf_rn(source[sx+2], w, sum.z);
                if (sx+3 >= 0 && sx+3 < width) sum.w = __fmaf_rn(source[sx+3], w, sum.w);
            }
            float *target = output + row * width + x;
            if constexpr (vectorized) *reinterpret_cast<float4 *>(target) = sum;
            else {
                target[0] = sum.x;
                if (x+1 < width) target[1] = sum.y;
                if (x+2 < width) target[2] = sum.z;
                if (x+3 < width) target[3] = sum.w;
            }
        }
    }
}

void ggml_cuda_op_conv2d_dw_sparse(ggml_backend_cuda_context &ctx, ggml_tensor *dst) {
    const auto *input = static_cast<const float *>(dst->src[1]->data);
    const auto *weights = static_cast<const float *>(dst->src[0]->data);
    const auto *offsets = static_cast<const int32_t *>(dst->src[2]->data);
    auto *output = static_cast<float *>(dst->data);
    const int64_t width = dst->ne[0], height = dst->ne[1], channels = dst->ne[2];
    const int64_t rows = height * channels * dst->ne[3], taps = dst->src[0]->ne[0];
    const int blocks = int(std::min<int64_t>((rows + 3) / 4, 65535));
    if (width % 4 == 0 && reinterpret_cast<uintptr_t>(output) % 16 == 0)
        conv2d_dw_sparse<true><<<blocks, 128, 0, ctx.stream()>>>(input, weights, offsets, output, width, height, channels, rows, taps);
    else
        conv2d_dw_sparse<false><<<blocks, 128, 0, ctx.stream()>>>(input, weights, offsets, output, width, height, channels, rows, taps);
}
#endif
