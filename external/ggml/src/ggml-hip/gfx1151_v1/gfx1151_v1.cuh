#pragma once

#include "../impl.cuh"

bool ggml_hip_gfx1151_v1_mul_mat(const ggml_tensor *, const ggml_tensor *, ggml_tensor *, hipStream_t);
bool ggml_hip_gfx1151_v1_mul_mat_typed(const ggml_tensor *, const ggml_tensor *, ggml_tensor *, hipStream_t);
bool ggml_hip_gfx1151_v1_im2col(const ggml_tensor *, ggml_tensor *, hipStream_t);

bool ggml_hip_gfx1151_v1_cpy(const ggml_tensor *, ggml_tensor *, hipStream_t);
