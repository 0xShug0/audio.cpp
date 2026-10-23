#pragma once

#include "ggml.h"
#include <hip/hip_runtime.h>

bool ggml_hip_halo_im2col_1d(
    const float * input, void * output, ggml_type type,
    int64_t width, int64_t positions, int64_t kernel, int64_t channels, int64_t batches,
    int64_t channel_stride, int64_t batch_stride,
    int stride, int padding, int dilation, hipStream_t stream);
