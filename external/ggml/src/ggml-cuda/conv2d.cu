#include "conv2d.cuh"
#include "convert.cuh"

#ifdef GGML_CUDA_CUDNN_CONV2D
#include <cudnn.h>
#endif

struct conv_params {
    const int64_t IW, IH;
    const int64_t OW, OH;
    const int64_t KW, KH;
    const int64_t ST_X, ST_Y;
    const int64_t PD_X, PD_Y;
    const int64_t DL_X, DL_Y;
    const int64_t IC, OC;
    const int64_t B;
    const int64_t TOTAL;
};

struct kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

__device__ __forceinline__ int64_t max64(int64_t a, int64_t b) {
    return (a > b) ? a : b;
}

__device__ __forceinline__ int64_t min64(int64_t a, int64_t b) {
    return (a < b) ? a : b;
}

__device__ __forceinline__ kernel_bounds calculate_kernel_bounds(int64_t out_x, int64_t out_y, const conv_params & P) {
    kernel_bounds bounds;
    bounds.y_min = max64(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = min64(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = max64(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = min64(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    return bounds;
}

__device__ __forceinline__ int calculate_input_coord(int64_t out_coord,
                                                     int64_t kern_coord,
                                                     int64_t stride,
                                                     int64_t dilation,
                                                     int64_t padding) {
    return out_coord * stride + kern_coord * dilation - padding;
}

struct whcn_layout {
    __device__ static int64_t input_index(int64_t n, int64_t c, int64_t y, int64_t x, const conv_params & P) {
        return n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
    }

    __device__ static int64_t kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, const conv_params & P) {
        return c_out * (P.IC * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
    }

    __device__ static int64_t output_index(int64_t n, int64_t c, int64_t y, int64_t x, const conv_params & P) {
        return n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
    }

    __device__ static void unpack_indices(int64_t             global_idx,
                                          const conv_params & P,
                                          int64_t &           n,
                                          int64_t &           c,
                                          int64_t &           out_y,
                                          int64_t &           out_x) {
        out_x = global_idx % P.OW;
        out_y = (global_idx / P.OW) % P.OH;
        c     = (global_idx / (P.OW * P.OH)) % P.OC;
        n     = global_idx / (P.OW * P.OH * P.OC);
    }
};

template <typename T, typename Layout>
static __global__ void conv2d_kernel(const float * __restrict__ input,
                                     const T * __restrict__ kernel,
                                     float * __restrict__ output,
                                     const conv_params P) {
    const int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx >= P.TOTAL) {
        return;
    }

    int64_t n, c_out, out_y, out_x;
    Layout::unpack_indices(global_idx, P, n, c_out, out_y, out_x);

    float acc = 0.0f;

    for (int64_t c_in = 0; c_in < P.IC; ++c_in) {
        kernel_bounds bounds = calculate_kernel_bounds(out_x, out_y, P);

        for (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            const int64_t in_y = calculate_input_coord(out_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);

            for (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                const int64_t in_x = calculate_input_coord(out_x, kx, P.ST_X, P.DL_X, P.PD_X);

                const float input_val = input[Layout::input_index(n, c_in, in_y, in_x, P)];
                const T kernel_val = kernel[Layout::kernel_index(c_out, c_in, ky, kx, P)];
                acc += (input_val * ggml_cuda_cast<float>(kernel_val));
            }
        }
    }

    // [N, OC, OH, OW]
    output[Layout::output_index(n, c_out, out_y, out_x, P)] = acc;
}

template <typename T>
static void conv2d_cuda(const float * X_D, const T * K_D, float * Y_D, const conv_params P, cudaStream_t st) {
    const int blocks = (P.TOTAL + CUDA_CONV2D_BLOCK_SIZE - 1) / CUDA_CONV2D_BLOCK_SIZE;
    conv2d_kernel<T, whcn_layout><<<blocks, CUDA_CONV2D_BLOCK_SIZE, 0, st>>>(X_D, K_D, Y_D, P);
}

static void conv2d_cuda_f16(const float * X_D, const half * K_D, float * Y_D, const conv_params P, cudaStream_t st) {
    conv2d_cuda<half>(X_D, K_D, Y_D, P, st);
}

static void conv2d_cuda_f32(const float * X_D, const float * K_D, float * Y_D, const conv_params P, cudaStream_t st) {
    conv2d_cuda<float>(X_D, K_D, Y_D, P, st);
}

#ifdef GGML_CUDA_CUDNN_CONV2D
static __global__ void conv2d_oihw_to_ohwi(const half * src, half * dst,
    int64_t out_channels, int64_t in_channels, int64_t height, int64_t width) {
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t count = out_channels * in_channels * height * width;
    if (index >= count) return;
    const int64_t ic = index % in_channels;
    const int64_t kw = (index / in_channels) % width;
    const int64_t kh = (index / (in_channels * width)) % height;
    const int64_t oc = index / (in_channels * width * height);
    dst[index] = src[((oc * in_channels + ic) * height + kh) * width + kw];
}

static void conv2d_cudnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];
    const ggml_tensor * bias   = dst->src[2];
    GGML_ASSERT(ggml_is_contiguous(kernel) && ggml_is_contiguous(input));
    GGML_ASSERT(kernel->type == input->type && input->type == dst->type);
    GGML_ASSERT(dst->type == GGML_TYPE_F16);
    GGML_ASSERT(!bias || (bias->type == GGML_TYPE_F16 && ggml_is_contiguous(bias) &&
        ggml_nelements(bias) == dst->ne[2]));

    const auto checked = [](cudnnStatus_t status) {
        if (status != CUDNN_STATUS_SUCCESS) {
            GGML_ABORT("cuDNN Conv2D failed: %s", cudnnGetErrorString(status));
        }
    };
    thread_local cudnnHandle_t handle = nullptr;
    if (!handle) checked(cudnnCreate(&handle));
    checked(cudnnSetStream(handle, ctx.stream()));

    cudnnTensorDescriptor_t x_desc, y_desc, x_nhwc_desc, y_nhwc_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    checked(cudnnCreateTensorDescriptor(&x_desc));
    checked(cudnnCreateTensorDescriptor(&y_desc));
    checked(cudnnCreateTensorDescriptor(&x_nhwc_desc));
    checked(cudnnCreateTensorDescriptor(&y_nhwc_desc));
    checked(cudnnCreateFilterDescriptor(&w_desc));
    checked(cudnnCreateConvolutionDescriptor(&conv_desc));
    const cudnnDataType_t dtype = CUDNN_DATA_HALF;
    checked(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, dtype,
        input->ne[3], input->ne[2], input->ne[1], input->ne[0]));
    checked(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, dtype,
        dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]));
    checked(cudnnSetTensor4dDescriptor(x_nhwc_desc, CUDNN_TENSOR_NHWC, dtype,
        input->ne[3], input->ne[2], input->ne[1], input->ne[0]));
    checked(cudnnSetTensor4dDescriptor(y_nhwc_desc, CUDNN_TENSOR_NHWC, dtype,
        dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]));
    checked(cudnnSetFilter4dDescriptor(w_desc, dtype, CUDNN_TENSOR_NHWC,
        kernel->ne[3], kernel->ne[2], kernel->ne[1], kernel->ne[0]));
    ggml_cuda_pool_alloc<uint8_t> x_nhwc(ctx.pool(), ggml_nbytes(input));
    ggml_cuda_pool_alloc<uint8_t> y_nhwc(ctx.pool(), ggml_nbytes(dst));
    ggml_cuda_pool_alloc<uint8_t> w_nhwc(ctx.pool(), ggml_nbytes(kernel));
    const float alpha = 1.0f, beta = 0.0f;
    checked(cudnnTransformTensor(handle, &alpha, x_desc, input->data, &beta, x_nhwc_desc, x_nhwc.get()));
    const int64_t weight_count = ggml_nelements(kernel);
    conv2d_oihw_to_ohwi<<<(weight_count + 255) / 256, 256, 0, ctx.stream()>>>(
        (const half *) kernel->data, (half *) w_nhwc.get(),
        kernel->ne[3], kernel->ne[2], kernel->ne[1], kernel->ne[0]);
    const int32_t * p = (const int32_t *) dst->op_params;
    checked(cudnnSetConvolution2dDescriptor(conv_desc, p[3], p[2], p[1], p[0], p[5], p[4],
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    checked(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
    constexpr size_t max_workspace_bytes = 256ull * 1024 * 1024;
    size_t workspace_bytes = 0;
    cudnnConvolutionFwdAlgo_t algorithm;
    if (bias) {
        algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        checked(cudnnGetConvolutionForwardWorkspaceSize(handle, x_nhwc_desc, w_desc, conv_desc, y_nhwc_desc,
            algorithm, &workspace_bytes));
    } else {
        cudnnConvolutionFwdAlgoPerf_t candidates[8];
        int candidate_count = 0;
        checked(cudnnGetConvolutionForwardAlgorithm_v7(handle, x_nhwc_desc, w_desc, conv_desc, y_nhwc_desc,
            8, &candidate_count, candidates));
        int selected = -1;
        for (int i = 0; i < candidate_count; ++i) {
            if (candidates[i].status != CUDNN_STATUS_SUCCESS) continue;
            size_t required_bytes = 0;
            if (cudnnGetConvolutionForwardWorkspaceSize(handle, x_nhwc_desc, w_desc, conv_desc, y_nhwc_desc,
                    candidates[i].algo, &required_bytes) != CUDNN_STATUS_SUCCESS) continue;
            if (required_bytes <= max_workspace_bytes) {
                selected = i;
                workspace_bytes = required_bytes;
                break;
            }
        }
        if (selected < 0) GGML_ABORT("cuDNN has no Conv2D algorithm within the workspace limit");
        algorithm = candidates[selected].algo;
    }
    if (workspace_bytes > max_workspace_bytes) GGML_ABORT("cuDNN Conv2D workspace exceeds limit");
    ggml_cuda_pool_alloc<uint8_t> workspace(ctx.pool());
    if (workspace_bytes) workspace.alloc(workspace_bytes);
    cudnnStatus_t convolution_status;
    if (bias) {
        cudnnTensorDescriptor_t bias_desc;
        cudnnActivationDescriptor_t activation_desc;
        checked(cudnnCreateTensorDescriptor(&bias_desc));
        checked(cudnnCreateActivationDescriptor(&activation_desc));
        checked(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, dtype, 1, dst->ne[2], 1, 1));
        checked(cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_IDENTITY,
            CUDNN_NOT_PROPAGATE_NAN, 0.0));
        convolution_status = cudnnConvolutionBiasActivationForward(handle, &alpha,
            x_nhwc_desc, x_nhwc.get(), w_desc, w_nhwc.get(), conv_desc, algorithm,
            workspace.get(), workspace_bytes, &beta, y_nhwc_desc, y_nhwc.get(),
            bias_desc, bias->data, activation_desc, y_nhwc_desc, y_nhwc.get());
        checked(cudnnDestroyActivationDescriptor(activation_desc));
        checked(cudnnDestroyTensorDescriptor(bias_desc));
    } else {
        convolution_status = cudnnConvolutionForward(handle, &alpha, x_nhwc_desc, x_nhwc.get(),
            w_desc, w_nhwc.get(), conv_desc, algorithm, workspace.get(), workspace_bytes,
            &beta, y_nhwc_desc, y_nhwc.get());
    }
    if (convolution_status != CUDNN_STATUS_SUCCESS) {
        GGML_ABORT("cuDNN Conv2D algorithm %d workspace %zu failed: %s", (int) algorithm,
            workspace_bytes, cudnnGetErrorString(convolution_status));
    }
    checked(cudnnTransformTensor(handle, &alpha, y_nhwc_desc, y_nhwc.get(), &beta, y_desc, dst->data));
    checked(cudnnDestroyConvolutionDescriptor(conv_desc));
    checked(cudnnDestroyFilterDescriptor(w_desc));
    checked(cudnnDestroyTensorDescriptor(y_desc));
    checked(cudnnDestroyTensorDescriptor(x_desc));
    checked(cudnnDestroyTensorDescriptor(y_nhwc_desc));
    checked(cudnnDestroyTensorDescriptor(x_nhwc_desc));
}
#endif

void ggml_cuda_op_conv2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#ifdef GGML_CUDA_CUDNN_CONV2D
    if (((const int32_t *) dst->op_params)[7] == 1) {
        conv2d_cudnn(ctx, dst);
        return;
    }
#else
    GGML_ASSERT(((const int32_t *) dst->op_params)[7] == 0);
#endif
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];
    float *             K_D    = (float *) kernel->data;
    const float *       X_D    = (const float *) input->data;
    float *             Y_D    = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(input->ne[2] == kernel->ne[2]);

    cudaStream_t st = ctx.stream();

    const int32_t * p    = (const int32_t *) dst->op_params;
    const int       ST_X = p[0];  // stride_x
    const int       ST_Y = p[1];  // stride_y
    const int       PD_X = p[2];  // padding_x
    const int       PD_Y = p[3];  // padding_y
    const int       DL_X = p[4];  // dilation_x
    const int       DL_Y = p[5];  // dilation_y

    // No cwhn
    GGML_ASSERT(p[6] == false);

    const int IW = input->ne[0];   // input_w
    const int IH = input->ne[1];   // input_h
    const int OW = dst->ne[0];     // output_w
    const int OH = dst->ne[1];     // output_h
    const int KW = kernel->ne[0];  // kernel_w
    const int KH = kernel->ne[1];  // kernel_h
    const int IC = input->ne[2];   // input_channels
    const int OC = kernel->ne[3];  // ouptut_chanles
    const int B  = input->ne[3];   // n_batches

    const int64_t total  = B * OC * OH * OW;
    conv_params   params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, total };

    if (kernel->type == GGML_TYPE_F16) {
        conv2d_cuda_f16(X_D, (half *) K_D, Y_D, params, st);
    } else {
        conv2d_cuda_f32(X_D, K_D, Y_D, params, st);
    }
}
