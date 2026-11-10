#include "pointwise.cuh"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "unary.cuh"
#include <climits>
#include <type_traits>

namespace {
template<int block_size, typename Output>
__global__ void norm_affine(const float * x, const float * shift, const float * scale,
        Output * output, int cols, float eps, float factor, float offset, bool transform) {
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    x += row * cols;
    output += row * cols;
    float2 sum = make_float2(0.0f, 0.0f);
    for (int col = tid; col < cols; col += block_size) {
        const float v = x[col];
        sum.x += v;
        sum.y += v * v;
    }
    extern __shared__ float2 scratch[];
    sum = block_reduce<block_reduce_method::SUM, block_size>(sum, scratch);
    const float mean = sum.x / cols;
    const float inv_std = rsqrtf(sum.y / cols - mean * mean + eps);
    for (int col = tid; col < cols; col += block_size) {
        const float normalized = (x[col] - mean) * inv_std;
        const float gamma = transform ? factor * scale[col] + offset : scale[col];
        const float value = __fadd_rn(__fmul_rn(normalized, gamma), shift[col]);
        if constexpr (std::is_same_v<Output, half>) output[col] = __float2half_rn(value);
        else output[col] = value;
    }
}

__global__ void add_gelu(const half * x, const half * bias, half * out, int64_t n, int64_t cols) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const half sum = __float2half_rn(__fadd_rn(__half2float(x[i]), __half2float(bias[i % cols])));
    out[i] = __float2half_rn(ggml_cuda_op_gelu_single(__half2float(sum)));
}

template<typename Input>
__global__ void add_gate_residual(const Input * x, const Input * bias, const float * gate,
        const float * residual, float * out, int64_t n, int64_t cols) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sum;
    if constexpr (std::is_same_v<Input, half>)
        sum = __half2float(__float2half_rn(__fadd_rn(__half2float(x[i]), __half2float(bias[i % cols]))));
    else sum = __fadd_rn(x[i], bias[i % cols]);
    out[i] = __fadd_rn(__fmul_rn(sum, gate[i % cols]), residual[i]);
}


}

void ggml_cuda_op_norm_affine(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int64_t cols = dst->ne[0], rows = ggml_nrows(dst);
    GGML_ASSERT(cols <= INT_MAX && rows <= INT_MAX);
    float params[4]; memcpy(params, dst->op_params, sizeof(params));
    const auto launch = [&](auto * out) {
        const auto * x = static_cast<const float *>(dst->src[0]->data);
        const auto * shift = static_cast<const float *>(dst->src[1]->data);
        const auto * scale = static_cast<const float *>(dst->src[2]->data);
        if (cols < 1024)
            norm_affine<WARP_SIZE><<<rows, WARP_SIZE, 0, ctx.stream()>>>(x, shift, scale, out, int(cols), params[0], params[1], params[2], params[3] != 0.0F);
        else
            norm_affine<1024><<<rows, 1024, 32*sizeof(float2), ctx.stream()>>>(x, shift, scale, out, int(cols), params[0], params[1], params[2], params[3] != 0.0F);
    };
    if (dst->type == GGML_TYPE_F16) launch(static_cast<half *>(dst->data));
    else launch(static_cast<float *>(dst->data));
}

void ggml_cuda_op_add_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int64_t n = ggml_nelements(dst), blocks = (n + 255) / 256;
    GGML_ASSERT(blocks <= INT_MAX);
    add_gelu<<<blocks, 256, 0, ctx.stream()>>>(static_cast<const half *>(dst->src[0]->data),
        static_cast<const half *>(dst->src[1]->data), static_cast<half *>(dst->data), n, dst->ne[0]);
}

void ggml_cuda_op_add_gate_residual(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int64_t n = ggml_nelements(dst), blocks = (n + 255) / 256;
    GGML_ASSERT(blocks <= INT_MAX);
    const auto launch = [&](auto * x, auto * bias) {
        add_gate_residual<<<blocks, 256, 0, ctx.stream()>>>(x, bias,
            static_cast<const float *>(dst->src[2]->data), static_cast<const float *>(dst->src[3]->data),
            static_cast<float *>(dst->data), n, dst->ne[0]);
    };
    if (dst->src[0]->type == GGML_TYPE_F16)
        launch(static_cast<const half *>(dst->src[0]->data), static_cast<const half *>(dst->src[1]->data));
    else launch(static_cast<const float *>(dst->src[0]->data), static_cast<const float *>(dst->src[1]->data));
}

#endif
