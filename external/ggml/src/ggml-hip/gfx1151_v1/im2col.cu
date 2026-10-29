#include "gfx1151_v1.cuh"

#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <climits>

namespace {

// Wave32 reads adjacent time samples. A padded LDS transpose then gives each
// wave adjacent patch-column stores, without the generic kernel's channel jumps.
template<typename T>
__global__ void hip_gfx1151_v1_im2col_1d(
        const float * __restrict__ input, T * __restrict__ output,
        int64_t width, int64_t positions, int64_t kernel, int64_t columns, int64_t batches,
        int64_t channel_stride, int64_t batch_stride, int stride, int padding, int dilation) {
    __shared__ float tile[32][33];
    const int64_t column_base = int64_t(blockIdx.x) * 32;
    for (int64_t batch = blockIdx.z; batch < batches; batch += gridDim.z) {
        for (int64_t base = int64_t(blockIdx.y) * 32; base < positions; base += int64_t(gridDim.y) * 32) {
            const int64_t position = base + threadIdx.x;
#pragma unroll
            for (int j = 0; j < 32; j += 8) {
                const int64_t column = column_base + threadIdx.y + j;
                const int64_t channel = column / kernel;
                const int64_t tap = column % kernel;
                const int64_t source = position * stride + tap * dilation - padding;
                float value = 0;
                if (position < positions && column < columns && source >= 0 && source < width) {
                    value = input[batch * batch_stride + channel * channel_stride + source];
                }
                tile[threadIdx.y + j][threadIdx.x] = value;
            }
            __syncthreads();
#pragma unroll
            for (int j = 0; j < 32; j += 8) {
                const int64_t out_position = base + threadIdx.y + j;
                const int64_t out_column = column_base + threadIdx.x;
                if (out_position < positions && out_column < columns) {
                    output[(batch * positions + out_position) * columns + out_column] =
                        T(tile[threadIdx.x][threadIdx.y + j]);
                }
            }
            __syncthreads();
        }
    }
}

template<typename T>
void launch(const float * input, void * output,
        int64_t width, int64_t positions, int64_t kernel, int64_t columns, int64_t batches,
        int64_t channel_stride, int64_t batch_stride,
        int stride, int padding, int dilation, hipStream_t stream) {
    const dim3 grid(unsigned((columns + 31) / 32),
                    unsigned(std::min<int64_t>((positions + 31) / 32, 65535)),
                    unsigned(std::min<int64_t>(batches, 65535)));
    hip_gfx1151_v1_im2col_1d<T><<<grid, dim3(32, 8), 0, stream>>>(input, static_cast<T *>(output),
        width, positions, kernel, columns, batches, channel_stride, batch_stride,
        stride, padding, dilation);
}

} // namespace

bool ggml_hip_gfx1151_v1_im2col(const ggml_tensor * input, ggml_tensor * out, hipStream_t stream) {
    const auto * params = reinterpret_cast<const int32_t *>(out->op_params);
    const auto * kernel_tensor = out->src[0];
    if (params[6] != 0 || params[7] != 0 || input->type != GGML_TYPE_F32 ||
        input->nb[0] != sizeof(float) || !ggml_is_contiguous(out)) {
        return false;
    }
    const int64_t width = input->ne[0], positions = out->ne[1], kernel = kernel_tensor->ne[0];
    const int64_t channels = input->ne[1], batches = input->ne[2];
    if (width <= 0 || positions < 32 || kernel <= 0 || channels <= 0 || batches <= 0 ||
        kernel > INT_MAX || channels > INT_MAX / kernel || positions > INT_MAX ||
        input->nb[1] % sizeof(float) || input->nb[2] % sizeof(float)) {
        return false;
    }
    const auto columns = channels * kernel;
    if (columns < 32 || positions < ((int64_t{1} << 18) + columns - 1) / columns) {
        return false;
    }
    const auto channel_stride = input->nb[1] / sizeof(float);
    const auto batch_stride = input->nb[2] / sizeof(float);
    const auto * data = static_cast<const float *>(input->data);
    switch (out->type) {
        case GGML_TYPE_F32:
            launch<float>(data, out->data, width, positions, kernel, columns, batches,
                channel_stride, batch_stride, params[0], params[2], params[4], stream);
            break;
        case GGML_TYPE_F16:
            launch<__half>(data, out->data, width, positions, kernel, columns, batches,
                channel_stride, batch_stride, params[0], params[2], params[4], stream);
            break;
        case GGML_TYPE_BF16:
            launch<hip_bfloat16>(data, out->data, width, positions, kernel, columns, batches,
                channel_stride, batch_stride, params[0], params[2], params[4], stream);
            break;
        default:
            return false;
    }
    return true;
}
