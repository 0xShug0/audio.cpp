#pragma once
#include "../../ggml-cuda/common.cuh"

void ggml_hip_halo_mmvq_q8_0(const void * weights, const void * input,
    float * output, int columns, int rows, int row_stride,
    ggml_cuda_mm_fusion_args_device fusion, hipStream_t stream);
