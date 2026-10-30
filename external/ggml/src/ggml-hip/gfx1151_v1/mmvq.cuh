#pragma once
#include "../impl.cuh"

template<int mode, int rows, int unroll = 0>
bool ggml_hip_ar_mmvq(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const void *,
                     const ggml_cuda_mm_fusion_args_device &, ggml_tensor *, hipStream_t);
