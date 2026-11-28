#include "affine.cuh"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// Remove an intermediate activation pass, preserving separate MUL/ADD rounding.
static __global__ void affine_f32(const float *x, const float *scale,
    const float *bias, float *out, int64_t count, int64_t stride, int64_t channels) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += int64_t(gridDim.x) * blockDim.x) {
        const int64_t c = i / stride % channels;
        out[i] = __fadd_rn(__fmul_rn(x[i], scale[c]), bias[c]);
    }
}

template<bool feature_vectors>
static __global__ void affine_f32x4(const float4 *x, const float *scale,
    const float *bias, float4 *out, int64_t count, int64_t stride, int64_t channels) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += int64_t(gridDim.x) * blockDim.x) {
        float4 gamma, beta;
        if constexpr (feature_vectors) {
            const int64_t c = i % channels;
            gamma = reinterpret_cast<const float4 *>(scale)[c];
            beta = reinterpret_cast<const float4 *>(bias)[c];
        } else {
            const int64_t c = i / stride % channels;
            gamma = make_float4(scale[c], scale[c], scale[c], scale[c]);
            beta = make_float4(bias[c], bias[c], bias[c], bias[c]);
        }
        const auto v = x[i];
        out[i] = make_float4(__fadd_rn(__fmul_rn(v.x, gamma.x), beta.x),
                            __fadd_rn(__fmul_rn(v.y, gamma.y), beta.y),
                            __fadd_rn(__fmul_rn(v.z, gamma.z), beta.z),
                            __fadd_rn(__fmul_rn(v.w, gamma.w), beta.w));
    }
}

void ggml_cuda_op_affine(ggml_backend_cuda_context &ctx, ggml_tensor *dst) {
    const int axis = ggml_get_op_params_i32(dst, 0);
    int64_t stride = 1;
    for (int d = 0; d < axis; ++d) stride *= dst->ne[d];
    const int64_t count = ggml_nelements(dst);
    const auto aligned = [](const void *pointer) { return reinterpret_cast<uintptr_t>(pointer) % 16 == 0; };
    const bool vectors = count % 4 == 0 && aligned(dst->src[0]->data) && aligned(dst->data);
    const bool feature_vectors = vectors && axis == 0 && dst->ne[0] % 4 == 0 &&
        aligned(dst->src[1]->data) && aligned(dst->src[2]->data);
    const bool channel_vectors = vectors && stride % 4 == 0;
    if (feature_vectors || channel_vectors) {
        const int blocks = int(std::min<int64_t>((count / 4 + 255) / 256, 65535));
        const auto *x = static_cast<const float4 *>(dst->src[0]->data);
        const auto *scale = static_cast<const float *>(dst->src[1]->data);
        const auto *bias = static_cast<const float *>(dst->src[2]->data);
        auto *out = static_cast<float4 *>(dst->data);
        if (feature_vectors)
            affine_f32x4<true><<<blocks, 256, 0, ctx.stream()>>>(x, scale, bias, out, count / 4, 1, dst->ne[0] / 4);
        else
            affine_f32x4<false><<<blocks, 256, 0, ctx.stream()>>>(x, scale, bias, out, count / 4, stride / 4, dst->ne[axis]);
        return;
    }
    const int blocks = int(std::min<int64_t>((count + 255) / 256, 65535));
    affine_f32<<<blocks, 256, 0, ctx.stream()>>>(
        static_cast<const float *>(dst->src[0]->data),
        static_cast<const float *>(dst->src[1]->data),
        static_cast<const float *>(dst->src[2]->data),
        static_cast<float *>(dst->data), count, stride, dst->ne[axis]);
}
#endif
