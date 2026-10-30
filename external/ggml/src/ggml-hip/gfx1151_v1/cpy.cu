#include "gfx1151_v1.cuh"

#include <algorithm>
#include <cstdint>

namespace {

struct row_layout {
    int64_t rows, channels;
    size_t row_stride, channel_stride, sample_stride;
};

__global__ void gfx1151_v1_copy_rows(const char * src, char * dst, int64_t width, int64_t rows,
                              row_layout a, row_layout b) {
    for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
        const int64_t ar = row % a.rows;
        const int64_t ac = row / a.rows % a.channels;
        const int64_t as = row / (a.rows * a.channels);
        const int64_t br = row % b.rows;
        const int64_t bc = row / b.rows % b.channels;
        const int64_t bs = row / (b.rows * b.channels);
        const auto * x = reinterpret_cast<const int4 *>(src + ar * a.row_stride +
            ac * a.channel_stride + as * a.sample_stride);
        auto * y = reinterpret_cast<int4 *>(dst + br * b.row_stride +
            bc * b.channel_stride + bs * b.sample_stride);
        const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (col < width) {
            y[col] = x[col];
        }
    }
}

} // namespace

bool ggml_hip_gfx1151_v1_cpy(const ggml_tensor * src, ggml_tensor * dst, hipStream_t stream) {
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        src->ne[0] != dst->ne[0] || ggml_nelements(src) == 0 ||
        src->ne[0] % 4 != 0 ||
        src->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    for (const auto * tensor : {src, static_cast<const ggml_tensor *>(dst)}) {
        if (reinterpret_cast<uintptr_t>(tensor->data) % 16 != 0 ||
            (tensor->ne[1] > 1 && tensor->nb[1] % 16 != 0) ||
            (tensor->ne[2] > 1 && tensor->nb[2] % 16 != 0) ||
            (tensor->ne[3] > 1 && tensor->nb[3] % 16 != 0)) {
            return false;
        }
    }
    const int64_t width = src->ne[0] / 4;
    const int64_t rows = ggml_nelements(src) / src->ne[0];
    const int threads = width <= 32 ? 32 : 256;
    const dim3 grid((width + threads - 1) / threads, std::min<int64_t>(rows, 65535));
    const row_layout a{src->ne[1], src->ne[2], src->nb[1], src->nb[2], src->nb[3]};
    const row_layout b{dst->ne[1], dst->ne[2], dst->nb[1], dst->nb[2], dst->nb[3]};
    gfx1151_v1_copy_rows<<<grid, threads, 0, stream>>>(static_cast<const char *>(src->data),
        static_cast<char *>(dst->data), width, rows, a, b);
    return true;
}
