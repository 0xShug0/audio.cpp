#include "conv3d.cuh"
#include "convert.cuh"

#include <algorithm>
#include <climits>
#include <cstdio>

#ifdef GGML_CUDA_CUDNN_CONV3D
// GGML's cpy_scalar_transpose tile pattern: each batch's contiguous [rows, cols]
// matrix becomes [cols, rows], combining layout conversion and the scalar cast.
template <typename Src, typename Dst, bool replicate_depth = false, bool scatter_output = false>
static __global__ void conv3d_transpose_cast(
        const Src * __restrict__ src, Dst * __restrict__ dst,
        int64_t rows, int64_t cols, int64_t batches, const float * bias, bool round_bf16,
        int64_t source_depth = 0, int64_t plane = 0, int depth_left = 0, int64_t source_offset = 0,
        int64_t output_stride = 0, int64_t output_offset = 0, int64_t valid_rows = 0) {
    __shared__ Src tile[32][33];
    const int64_t col_base = int64_t(blockIdx.x) * 32;
    for (int64_t batch = blockIdx.z; batch < batches; batch += gridDim.z) {
        for (int64_t row_base = int64_t(blockIdx.y) * 32;
             row_base < rows; row_base += int64_t(gridDim.y) * 32) {
            const int64_t col = col_base + threadIdx.x;
#pragma unroll
            for (int offset = 0; offset < 32; offset += 8) {
                const int64_t row = row_base + threadIdx.y + offset;
                if (row < rows && col < cols) {
                    if constexpr (replicate_depth) {
                        const int64_t position = col / plane + source_offset - depth_left;
                        const int64_t depth = position < 0 ? 0 : (position < source_depth ? position : source_depth - 1);
                        tile[threadIdx.y + offset][threadIdx.x] =
                            src[((batch * rows + row) * source_depth + depth) * plane + col % plane];
                    } else {
                        tile[threadIdx.y + offset][threadIdx.x] = src[(batch * rows + row) * cols + col];
                    }
                }
            }
            __syncthreads();
            const int64_t out_col = row_base + threadIdx.x;
#pragma unroll
            for (int offset = 0; offset < 32; offset += 8) {
                const int64_t out_row = col_base + threadIdx.y + offset;
                if (out_row < cols && out_col < rows && (!scatter_output || out_col < valid_rows)) {
                    const int64_t output_index = scatter_output
                        ? (batch * cols + out_row) * output_stride + output_offset + out_col
                        : (batch * cols + out_row) * rows + out_col;
                    const Src value = tile[threadIdx.x][threadIdx.y + offset];
                    if constexpr (std::is_same_v<Src, float> && std::is_same_v<Dst, half>) {
                        // Match the former explicit F32 -> F16 cast, including ties.
                        dst[output_index] = __float2half_rn(value);
                    } else if constexpr (std::is_same_v<Dst, float>) {
                        float result = ggml_cuda_cast<float>(value);
                        if (bias != nullptr) {
                            result = result + bias[out_row]; // Same F32 addition as the standalone ADD.
                        }
                        if (round_bf16) {
                            // Existing conversion utility uses exactly the intrinsics
                            // in op_round_bf16: __float2bfloat16 then __bfloat162float.
                            result = ggml_cuda_cast<float>(ggml_cuda_cast<nv_bfloat16>(result));
                        }
                        dst[output_index] = result;
                    } else if constexpr (std::is_same_v<Src, half> && std::is_same_v<Dst, half>) {
                        // Input packing passes no bias and remains an exact half copy.
                        // Output bias uses F32 ADD followed by one round-to-nearest-even F16 cast.
                        if (bias != nullptr) {
                            const float result = ggml_cuda_cast<float>(value) + bias[out_row];
                            dst[output_index] = __float2half_rn(result);
                        } else {
                            dst[output_index] = value;
                        }
                    } else {
                        dst[output_index] = ggml_cuda_cast<Dst>(value);
                    }
                }
            }
            __syncthreads();
        }
    }
}
#endif

bool ggml_cuda_supports_conv3d(const ggml_tensor * op) {
#ifdef GGML_CUDA_CUDNN_CONV3D
    if (op->op != GGML_OP_CONV_3D || op->op_params[12] != 1 ||
        !op->src[0] || !op->src[1] ||
        op->src[0]->type != GGML_TYPE_F16 ||
        (op->src[1]->type != GGML_TYPE_F16 && op->src[1]->type != GGML_TYPE_F32) ||
        (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) || !ggml_is_contiguous(op->src[0]) ||
        !ggml_is_contiguous(op->src[1]) || !ggml_is_contiguous(op)) {
        return false;
    }
    if ((op->op_params[13] != 0 && op->op_params[13] != 1) ||
        (op->op_params[14] != 0 && op->op_params[14] != 1) ||
        (op->type == GGML_TYPE_F16 && op->op_params[13] != 0) ||
        (op->src[2] && (op->src[2]->type != GGML_TYPE_F32 || !ggml_is_contiguous(op->src[2]) ||
            ggml_nelements(op->src[2]) != op->op_params[11]))) {
        return false;
    }
    for (int i = 3; i < GGML_MAX_SRC; ++i) {
        if (op->src[i]) {
            return false;
        }
    }
    const int32_t * p = op->op_params;
    if (p[15] < 0 || p[15] > 2LL * p[5] + 1) {
        return false;
    }
    if (p[9] <= 0 || p[10] <= 0 || p[11] <= 0 ||
        op->src[0]->ne[3] != int64_t(p[9]) * p[11] ||
        op->src[1]->ne[3] != int64_t(p[9]) * p[10] ||
        op->ne[3] != int64_t(p[11]) * p[10]) {
        return false;
    }
    // cuDNN Nd descriptors use int dimensions/strides and at most 2G elements.
    const ggml_tensor * tensors[] = {op->src[0], op->src[1], op};
    for (const ggml_tensor * t : tensors) {
        int64_t elements = 1;
        for (int axis = 0; axis < 4; ++axis) {
            if (t->ne[axis] <= 0 || t->ne[axis] > INT_MAX / elements) {
                return false;
            }
            elements *= t->ne[axis];
        }
    }
    if (p[15] && (op->src[1]->ne[2] + 2LL * p[5] > INT_MAX ||
        ggml_nelements(op->src[1]) / op->src[1]->ne[2] >
            INT_MAX / (op->src[1]->ne[2] + 2LL * p[5]))) {
        return false;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (p[axis] <= 0 || p[axis + 3] < 0 || p[axis + 6] <= 0) {
            return false;
        }
        const int64_t extent = op->src[1]->ne[axis] + 2LL * p[axis + 3] -
            int64_t(p[axis + 6]) * (op->src[0]->ne[axis] - 1) - 1;
        if (extent < 0 || op->ne[axis] != extent / p[axis] + 1) {
            return false;
        }
    }
    return true;
#else
    GGML_UNUSED(op);
    return false;
#endif
}

void ggml_cuda_op_conv3d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#ifdef GGML_CUDA_CUDNN_CONV3D
    GGML_ASSERT(ggml_cuda_supports_conv3d(dst));
    const auto checked = [](cudnnStatus_t status) {
        if (status != CUDNN_STATUS_SUCCESS) {
            GGML_ABORT("cuDNN Conv3D failed: %s", cudnnGetErrorString(status));
        }
    };
    if (ctx.conv3d_cudnn_handle == nullptr) {
        checked(cudnnCreate(&ctx.conv3d_cudnn_handle));
    }
    const cudnnHandle_t handle = ctx.conv3d_cudnn_handle;
    checked(cudnnSetStream(handle, ctx.stream()));

    // No shape/pointer caches: descriptors and heuristic results are bounded
    // invocation-local metadata. Only the backend-owned handle persists.
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const int32_t * p = dst->op_params;
    const bool replicate_depth = p[15] != 0;
    int xd[] = {p[10], p[9], int(x->ne[2] + (replicate_depth ? 2LL * p[5] : 0)),
                     int(x->ne[1]), int(x->ne[0])};
    int yd[] = {p[10], p[11], int(dst->ne[2]), int(dst->ne[1]), int(dst->ne[0])};
    const int wd[] = {p[11], p[9], int(w->ne[2]), int(w->ne[1]), int(w->ne[0])};
    const int padding[] = {replicate_depth ? 0 : p[5], p[4], p[3]};
    const int stride[] = {p[2], p[1], p[0]};
    const int dilation[] = {p[8], p[7], p[6]};
    cudnnTensorDescriptor_t x_ndhwc_desc, y_desc, w_source_desc, w_packed_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    checked(cudnnCreateTensorDescriptor(&x_ndhwc_desc));
    checked(cudnnCreateTensorDescriptor(&y_desc));
    checked(cudnnCreateTensorDescriptor(&w_source_desc));
    checked(cudnnCreateTensorDescriptor(&w_packed_desc));
    checked(cudnnCreateFilterDescriptor(&w_desc));
    checked(cudnnCreateConvolutionDescriptor(&conv_desc));
    // Descriptor dimensions stay N,C,D,H,W (O,I,KD,KH,KW for weights).
    // NHWC on a five-dimensional descriptor selects physical NDHWC/ODHWI.
    checked(cudnnSetTensorNdDescriptorEx(x_ndhwc_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 5, xd));
    checked(cudnnSetTensorNdDescriptorEx(y_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 5, yd));
    // Treat the filter as a tensor solely for the existing cuDNN layout transform.
    checked(cudnnSetTensorNdDescriptorEx(w_source_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 5, wd));
    checked(cudnnSetTensorNdDescriptorEx(w_packed_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 5, wd));
    checked(cudnnSetFilterNdDescriptor(w_desc, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, 5, wd));
    checked(cudnnSetConvolutionNdDescriptor(conv_desc, 3, padding, stride, dilation,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    checked(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));

    constexpr size_t max_workspace_bytes = 256ull * 1024 * 1024;
    // Each legacy algorithm can have default-math and tensor-op variants.
    constexpr int max_candidates = 2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    cudnnConvolutionFwdAlgoPerf_t candidates[max_candidates];
    const int full_input_depth = xd[2], full_output_depth = yd[2];
    int tile_depth = full_output_depth;
    cudnnConvolutionFwdAlgoPerf_t chosen{}, untiled{};
    size_t workspace_bytes = 0, untiled_workspace = 0;
    for (;;) {
        checked(cudnnSetTensorNdDescriptorEx(x_ndhwc_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 5, xd));
        checked(cudnnSetTensorNdDescriptorEx(y_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 5, yd));
        checked(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
        int candidate_count = 0, selected = -1;
        bool tensor_core_exceeds_budget = false;
        checked(cudnnGetConvolutionForwardAlgorithm_v7(handle, x_ndhwc_desc, w_desc, conv_desc, y_desc,
            max_candidates, &candidate_count, candidates));
        for (int i = 0; i < candidate_count; ++i) {
            if (candidates[i].status != CUDNN_STATUS_SUCCESS) continue;
            checked(cudnnSetConvolutionMathType(conv_desc, candidates[i].mathType));
            size_t required_bytes = 0;
            checked(cudnnGetConvolutionForwardWorkspaceSize(handle, x_ndhwc_desc, w_desc, conv_desc, y_desc,
                candidates[i].algo, &required_bytes));
            if (candidates[i].mathType == CUDNN_TENSOR_OP_MATH && required_bytes > max_workspace_bytes)
                tensor_core_exceeds_budget = true;
            if (required_bytes <= max_workspace_bytes) {
                selected = i;
                workspace_bytes = required_bytes;
                break;
            }
        }
        if (selected < 0) GGML_ABORT("cuDNN has no Conv3D algorithm within the 256 MiB workspace limit");
        chosen = candidates[selected];
        if (tile_depth == full_output_depth) {
            untiled = chosen;
            untiled_workspace = workspace_bytes;
        }
        if (chosen.mathType == CUDNN_TENSOR_OP_MATH || !replicate_depth ||
            (!tensor_core_exceeds_budget && tile_depth == full_output_depth)) break;
        if (!tensor_core_exceeds_budget || tile_depth == 1) {
            // No tensor-core tile fits: preserve the original untiled algorithm.
            tile_depth = full_output_depth;
            xd[2] = full_input_depth;
            yd[2] = full_output_depth;
            chosen = untiled;
            workspace_bytes = untiled_workspace;
            checked(cudnnSetTensorNdDescriptorEx(x_ndhwc_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 5, xd));
            checked(cudnnSetTensorNdDescriptorEx(y_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 5, yd));
            break;
        }
        tile_depth = std::max(1, tile_depth / 2);
        yd[2] = tile_depth;
        xd[2] = (tile_depth - 1) * stride[0] + dilation[0] * (wd[2] - 1) + 1;
    }
    checked(cudnnSetConvolutionMathType(conv_desc, chosen.mathType));
    // Pool-owned scratch is released at return (the pool may retain capacity).
    // No im2col or padding expansion: peak requested scratch is exactly the
    // F16 input + optional filter packing + F16 output + <=256 MiB workspace.
    const size_t input_elements = size_t(xd[0]) * xd[1] * xd[2] * xd[3] * xd[4];
    const size_t input_bytes = input_elements * sizeof(half);
    const size_t weight_bytes = p[14] ? 0 : ggml_nbytes(w);
    const size_t output_elements = size_t(yd[0]) * yd[1] * yd[2] * yd[3] * yd[4];
    const size_t output_bytes = output_elements * sizeof(half);
    ggml_cuda_pool_alloc<half> input(ctx.pool(), input_elements);
    ggml_cuda_pool_alloc<half> weight(ctx.pool());
    if (!p[14]) weight.alloc(size_t(ggml_nelements(w)));
    ggml_cuda_pool_alloc<half> output(ctx.pool(), output_elements);
    ggml_cuda_pool_alloc<uint8_t> workspace(ctx.pool());
    if (workspace_bytes != 0) {
        workspace.alloc(workspace_bytes);
    }
    GGML_LOG_DEBUG("cuDNN Conv3D NDHWC: input %s, output %s, bias %s, bf16_round %d, algorithm %d, math %d, workspace %zu bytes (pool %zu), "
        "input scratch %zu bytes (pool %zu), filter scratch %zu bytes (pool %zu), "
        "output scratch %zu bytes (pool %zu), total scratch %zu bytes (pool %zu)\n",
        ggml_type_name(x->type), ggml_type_name(dst->type), dst->src[2] ? "F32" : "none", p[13],
        int(chosen.algo), int(chosen.mathType), workspace_bytes, workspace.actual_size,
        input_bytes, input.actual_size, weight_bytes, weight.actual_size, output_bytes, output.actual_size,
        input_bytes + weight_bytes + output_bytes + workspace_bytes,
        input.actual_size + weight.actual_size + output.actual_size + workspace.actual_size);
    const float alpha = 1.0f, beta = 0.0f;
    const void * filter = w->data;
    if (!p[14]) {
        checked(cudnnTransformTensor(handle, &alpha, w_source_desc, w->data, &beta, w_packed_desc, weight.get()));
        filter = weight.get();
    }
    const int64_t input_spatial = int64_t(xd[2]) * xd[3] * xd[4];
    const int64_t spatial = int64_t(yd[2]) * yd[3] * yd[4];
    const int64_t output_plane = dst->ne[0] * dst->ne[1];
    const int64_t output_stride = output_plane * full_output_depth;
    const dim3 input_blocks(unsigned((input_spatial + 31) / 32),
        unsigned(std::min<int64_t>((p[9] + 31LL) / 32, 65535)), unsigned(std::min(p[10], 65535)));
    const dim3 output_blocks(unsigned((p[11] + 31LL) / 32),
        unsigned(std::min<int64_t>((spatial + 31) / 32, 65535)), unsigned(std::min(p[10], 65535)));
    GGML_LOG_DEBUG("cuDNN Conv3D depth tile %d/%d, input depth %d, workspace %zu\n",
        tile_depth, full_output_depth, xd[2], workspace_bytes);
    for (int64_t start = 0; start < full_output_depth; start += tile_depth) {
        if (replicate_depth && x->type == GGML_TYPE_F32) {
            conv3d_transpose_cast<float, half, true><<<input_blocks, dim3(32, 8), 0, ctx.stream()>>>(
                static_cast<const float *>(x->data), input.get(), p[9], input_spatial, p[10], nullptr, false,
                x->ne[2], x->ne[0] * x->ne[1], p[15] - 1, int64_t(start) * stride[0]);
        } else if (replicate_depth) {
            conv3d_transpose_cast<half, half, true><<<input_blocks, dim3(32, 8), 0, ctx.stream()>>>(
                static_cast<const half *>(x->data), input.get(), p[9], input_spatial, p[10], nullptr, false,
                x->ne[2], x->ne[0] * x->ne[1], p[15] - 1, int64_t(start) * stride[0]);
        } else if (x->type == GGML_TYPE_F32) {
            conv3d_transpose_cast<float, half><<<input_blocks, dim3(32, 8), 0, ctx.stream()>>>(
                static_cast<const float *>(x->data), input.get(), p[9], input_spatial, p[10], nullptr, false);
        } else {
            conv3d_transpose_cast<half, half><<<input_blocks, dim3(32, 8), 0, ctx.stream()>>>(
                static_cast<const half *>(x->data), input.get(), p[9], input_spatial, p[10], nullptr, false);
        }
        CUDA_CHECK(cudaGetLastError());
        checked(cudnnConvolutionForward(handle, &alpha, x_ndhwc_desc, input.get(), w_desc, filter,
            conv_desc, chosen.algo, workspace.get(), workspace_bytes, &beta, y_desc, output.get()));
        const float * bias = dst->src[2] ? static_cast<const float *>(dst->src[2]->data) : nullptr;
        if (tile_depth != full_output_depth) {
            const int64_t offset = int64_t(start) * output_plane;
            const int64_t valid = std::min<int64_t>(tile_depth, full_output_depth - start) * output_plane;
            if (dst->type == GGML_TYPE_F16) {
                conv3d_transpose_cast<half, half, false, true><<<output_blocks, dim3(32, 8), 0, ctx.stream()>>>(
                    output.get(), static_cast<half *>(dst->data), spatial, p[11], p[10], bias, false,
                    0, 0, 0, 0, output_stride, offset, valid);
            } else {
                conv3d_transpose_cast<half, float, false, true><<<output_blocks, dim3(32, 8), 0, ctx.stream()>>>(
                    output.get(), static_cast<float *>(dst->data), spatial, p[11], p[10], bias, p[13] != 0,
                    0, 0, 0, 0, output_stride, offset, valid);
            }
        } else if (dst->type == GGML_TYPE_F16) {
            conv3d_transpose_cast<half, half><<<output_blocks, dim3(32, 8), 0, ctx.stream()>>>(
                output.get(), static_cast<half *>(dst->data), spatial, p[11], p[10], bias, false);
        } else {
            conv3d_transpose_cast<half, float><<<output_blocks, dim3(32, 8), 0, ctx.stream()>>>(
                output.get(), static_cast<float *>(dst->data), spatial, p[11], p[10], bias, p[13] != 0);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    checked(cudnnDestroyConvolutionDescriptor(conv_desc));
    checked(cudnnDestroyFilterDescriptor(w_desc));
    checked(cudnnDestroyTensorDescriptor(w_packed_desc));
    checked(cudnnDestroyTensorDescriptor(w_source_desc));
    checked(cudnnDestroyTensorDescriptor(y_desc));
    checked(cudnnDestroyTensorDescriptor(x_ndhwc_desc));
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("cuDNN Conv3D requires GGML_CUDA_CUDNN_CONV3D");
#endif
}

#define CUDA_CONV3D_BLOCK_SIZE 256
#define CUDA_CONV3D_SPATIAL_IM2COL_BLOCK_SIZE 256

template <typename T>
static __device__ float conv3d_load_as_float(const T * ptr) {
    return static_cast<float>(*ptr);
}

template <>
__device__ float conv3d_load_as_float<half>(const half * ptr) {
    return __half2float(*ptr);
}

template <typename TA, typename TB, int X_TILE, int C_TILE>
static __global__ void conv3d_concat_pad_spatial_im2col_kernel(
        const TA * __restrict__ a,
        const TB * __restrict__ b,
        half * __restrict__ dst,
        int IC, int AT, int BT, int H, int W, int OD, int OH, int OW,
        int lp0, int lp1, int lp2,
        int src_t_offset,
        int spatial_y_offset,
        int tile_oh,
        int a_stride_q, int a_stride_z, int a_stride_y, int a_stride_x,
        int b_stride_q, int b_stride_z, int b_stride_y, int b_stride_x) {
    const int local = threadIdx.x;
    if (local >= C_TILE * 9) {
        return;
    }

    const int groups = (IC + C_TILE - 1) / C_TILE;
    const int group = blockIdx.x % groups;
    const int local_src_t = blockIdx.x / groups;
    const int src_t_pad = local_src_t + src_t_offset;
    const int base_iow = blockIdx.y * X_TILE;
    const int local_ioh = blockIdx.z;
    const int ioh = spatial_y_offset + local_ioh;

    const int local_channel = local / 9;
    const int iic = group * C_TILE + local_channel;
    if (iic >= IC) {
        return;
    }

    const int rem0 = local - local_channel * 9;
    const int ikh = rem0 / 3;
    const int ikw = rem0 - ikh * 3;
    const int src_y_pad = ioh + ikh;
    const bool y_ok = src_y_pad >= lp1 && src_y_pad < lp1 + H;
    const int src_y = src_y_pad - lp1;

    const bool t_in_a = src_t_pad >= lp2 && src_t_pad < lp2 + AT;
    const bool t_in_b = src_t_pad >= lp2 + AT && src_t_pad < lp2 + AT + BT;
    const int src_t_a = src_t_pad - lp2;
    const int src_t_b = src_t_pad - lp2 - AT;

    const int K2 = IC * 9;
    const int M = tile_oh * OW;
    const int64_t dst_base =
        (static_cast<int64_t>(local_src_t) * M + static_cast<int64_t>(local_ioh) * OW + base_iow) * K2 +
        static_cast<int64_t>(iic) * 9 + rem0;

#pragma unroll
    for (int dx = 0; dx < X_TILE; ++dx) {
        const int iow = base_iow + dx;
        if (iow >= OW) {
            return;
        }

        const int src_x_pad = iow + ikw;
        float value = 0.0f;
        if (y_ok && src_x_pad >= lp0 && src_x_pad < lp0 + W) {
            const int src_x = src_x_pad - lp0;
            if (t_in_a) {
                value = conv3d_load_as_float(a + iic * a_stride_q + src_t_a * a_stride_z + src_y * a_stride_y + src_x * a_stride_x);
            } else if (t_in_b) {
                value = conv3d_load_as_float(b + iic * b_stride_q + src_t_b * b_stride_z + src_y * b_stride_y + src_x * b_stride_x);
            }
        }
        dst[dst_base + dx * K2] = __float2half(value);
    }

    GGML_UNUSED(OD);
    GGML_UNUSED(OH);
}

template <typename TA, typename TB, int X_TILE, int C_TILE>
static void launch_conv3d_concat_pad_spatial_im2col(
        cudaStream_t stream,
        dim3 grid,
        int threads,
        const ggml_tensor * a,
        const ggml_tensor * b,
        half * im2col,
        int IC, int AT, int BT, int H, int W, int OD, int OH, int OW,
        int lp0, int lp1, int lp2, int src_t_offset, int spatial_y_offset, int tile_oh,
        int a_type_size, int b_type_size) {
    conv3d_concat_pad_spatial_im2col_kernel<TA, TB, X_TILE, C_TILE><<<grid, threads, 0, stream>>>(
        reinterpret_cast<const TA *>(a->data),
        reinterpret_cast<const TB *>(b->data),
        im2col,
        IC, AT, BT, H, W, OD, OH, OW,
        lp0, lp1, lp2, src_t_offset, spatial_y_offset, tile_oh,
        static_cast<int>(a->nb[3] / a_type_size), static_cast<int>(a->nb[2] / a_type_size),
        static_cast<int>(a->nb[1] / a_type_size), static_cast<int>(a->nb[0] / a_type_size),
        static_cast<int>(b->nb[3] / b_type_size), static_cast<int>(b->nb[2] / b_type_size),
        static_cast<int>(b->nb[1] / b_type_size), static_cast<int>(b->nb[0] / b_type_size));
}

template <int X_TILE, int C_TILE>
static void dispatch_conv3d_concat_pad_spatial_im2col(
        cudaStream_t stream,
        dim3 grid,
        int threads,
        const ggml_tensor * a,
        const ggml_tensor * b,
        half * im2col,
        int IC, int AT, int BT, int H, int W, int OD, int OH, int OW,
        int lp0, int lp1, int lp2, int src_t_offset, int spatial_y_offset, int tile_oh) {
    const int a_type_size = ggml_type_size(a->type);
    const int b_type_size = ggml_type_size(b->type);
    if (a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
        launch_conv3d_concat_pad_spatial_im2col<half, half, X_TILE, C_TILE>(
            stream, grid, threads, a, b, im2col, IC, AT, BT, H, W, OD, OH, OW, lp0, lp1, lp2, src_t_offset,
            spatial_y_offset, tile_oh, a_type_size, b_type_size);
    } else if (a->type == GGML_TYPE_F16) {
        launch_conv3d_concat_pad_spatial_im2col<half, float, X_TILE, C_TILE>(
            stream, grid, threads, a, b, im2col, IC, AT, BT, H, W, OD, OH, OW, lp0, lp1, lp2, src_t_offset,
            spatial_y_offset, tile_oh, a_type_size, b_type_size);
    } else if (b->type == GGML_TYPE_F16) {
        launch_conv3d_concat_pad_spatial_im2col<float, half, X_TILE, C_TILE>(
            stream, grid, threads, a, b, im2col, IC, AT, BT, H, W, OD, OH, OW, lp0, lp1, lp2, src_t_offset,
            spatial_y_offset, tile_oh, a_type_size, b_type_size);
    } else {
        launch_conv3d_concat_pad_spatial_im2col<float, float, X_TILE, C_TILE>(
            stream, grid, threads, a, b, im2col, IC, AT, BT, H, W, OD, OH, OW, lp0, lp1, lp2, src_t_offset,
            spatial_y_offset, tile_oh, a_type_size, b_type_size);
    }
}

static __global__ void conv3d_zero_spatial_prefix_kernel(
        half * __restrict__ dst,
        int rows,
        int ld,
        int OC) {
    const int64_t n = int64_t(rows) * OC;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(blockDim.x) * gridDim.x) {
        const int row = i % rows;
        const int oc = i / rows;
        dst[row + oc * ld] = __float2half(0.0f);
    }
}

static __global__ void conv3d_k3_weight_f32_to_f16_spatial_kernel(
        const float * __restrict__ w,
        half * __restrict__ dst,
        int IC, int OC) {
    const int K2 = IC * 9;
    const int total = 3 * K2 * OC;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }

    const int rem_oc = i % (K2 * OC);
    const int kd = i / (K2 * OC);
    const int oc = rem_oc / K2;
    const int k2 = rem_oc - oc * K2;
    const int ic = k2 / 9;
    const int rem = k2 - ic * 9;
    const int kh = rem / 3;
    const int kw = rem - kh * 3;
    const int src = kw + kh * 3 + kd * 9 + (oc * IC + ic) * 27;
    dst[i] = __float2half(w[src]);
}

void ggml_cuda_op_conv3d_concat_pad_spatial_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];
    const ggml_tensor * w = dst->src[2];

    GGML_ASSERT(a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F16 || b->type == GGML_TYPE_F32);
    GGML_ASSERT(w->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);
    GGML_ASSERT(w->ne[0] == 3 && w->ne[1] == 3 && w->ne[2] == 3);

    const int32_t * params = reinterpret_cast<const int32_t *>(dst->op_params);
    const int lp0 = params[0];
    const int lp1 = params[2];
    const int lp2 = params[4];

    const int64_t IC = a->ne[3];
    const int64_t AT = a->ne[2];
    const int64_t BT = b->ne[2];
    const int64_t H  = a->ne[1];
    const int64_t W  = a->ne[0];
    const int64_t OD = dst->ne[2];
    const int64_t OH = dst->ne[1];
    const int64_t OW = dst->ne[0];
    const int64_t OC = dst->ne[3];
    const int64_t K2 = IC * 9;
    const int64_t M  = OH * OW;
    const int64_t source_frames = OD + 2;

    GGML_ASSERT(w->ne[3] == IC * OC);
    GGML_ASSERT(IC <= INT_MAX && AT <= INT_MAX && BT <= INT_MAX && H <= INT_MAX && W <= INT_MAX);
    GGML_ASSERT(OD <= INT_MAX && OH <= INT_MAX && OW <= INT_MAX && OC <= INT_MAX);
    const int a_type_size = ggml_type_size(a->type);
    const int b_type_size = ggml_type_size(b->type);
    GGML_ASSERT(a->nb[0] / a_type_size <= INT_MAX && a->nb[1] / a_type_size <= INT_MAX &&
                a->nb[2] / a_type_size <= INT_MAX && a->nb[3] / a_type_size <= INT_MAX);
    GGML_ASSERT(b->nb[0] / b_type_size <= INT_MAX && b->nb[1] / b_type_size <= INT_MAX &&
                b->nb[2] / b_type_size <= INT_MAX && b->nb[3] / b_type_size <= INT_MAX);

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<half> weight(ctx.pool(), static_cast<size_t>(3 * K2 * OC));
    ggml_cuda_pool_alloc<half> out;
    half * out_data = nullptr;
    if (dst->type == GGML_TYPE_F16) {
        out_data = reinterpret_cast<half *>(dst->data);
    } else {
        out.alloc(ctx.pool(), static_cast<size_t>(OD * M * OC));
        out_data = out.get();
    }

    const bool skip_front_zero = false;
    const bool use_skip_front_zero = skip_front_zero && lp2 == 1 && source_frames == OD + 2 && OD > 1;

    const int groups = (static_cast<int>(IC) + 27) / 28;
    const int src_t_offset = use_skip_front_zero ? 1 : 0;
    const int weight_blocks = (static_cast<int>(3 * K2 * OC) + CUDA_CONV3D_BLOCK_SIZE - 1) / CUDA_CONV3D_BLOCK_SIZE;
    conv3d_k3_weight_f32_to_f16_spatial_kernel<<<weight_blocks, CUDA_CONV3D_BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const float *>(w->data), weight.get(), static_cast<int>(IC), static_cast<int>(OC));

    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));

    const half alpha = __float2half(1.0f);
    const half beta_zero = __float2half(0.0f);
    const half beta_one = __float2half(1.0f);
    if (use_skip_front_zero) {
        const int64_t zero_rows = M * OC;
        const int zero_blocks = (static_cast<int>(zero_rows) + CUDA_CONV3D_BLOCK_SIZE - 1) / CUDA_CONV3D_BLOCK_SIZE;
        conv3d_zero_spatial_prefix_kernel<<<zero_blocks, CUDA_CONV3D_BLOCK_SIZE, 0, stream>>>(
            out_data, static_cast<int>(M), static_cast<int>(OD * M), static_cast<int>(OC));
    }

    const int32_t conv_lowering = ggml_get_op_params_i32(dst, 6);
    const bool use_tiled_f16_path =
        conv_lowering == GGML_CONV_3D_CONCAT_PAD_SPATIAL_GEMM_LOWERING_CUDA_TILED_C48;
    if (!use_tiled_f16_path) {
        ggml_cuda_pool_alloc<half> im2col(ctx.pool(), static_cast<size_t>(source_frames * M * K2));
        const int active_source_frames = static_cast<int>(source_frames) - src_t_offset;
        if (conv_lowering == GGML_CONV_3D_CONCAT_PAD_SPATIAL_GEMM_LOWERING_CUDA_C48) {
            const int groups_c48 = (static_cast<int>(IC) + 47) / 48;
            dim3 im2col_grid(groups_c48 * active_source_frames, (OW + 7) / 8, OH);
            dispatch_conv3d_concat_pad_spatial_im2col<8, 48>(
                stream, im2col_grid, 48 * 9, a, b, im2col.get(),
                static_cast<int>(IC), static_cast<int>(AT), static_cast<int>(BT),
                static_cast<int>(H), static_cast<int>(W), static_cast<int>(OD), static_cast<int>(OH), static_cast<int>(OW),
                lp0, lp1, lp2, src_t_offset, 0, static_cast<int>(OH));
        } else {
            dim3 im2col_grid(groups * active_source_frames, (OW + 7) / 8, OH);
            dispatch_conv3d_concat_pad_spatial_im2col<8, 28>(
                stream, im2col_grid, CUDA_CONV3D_SPATIAL_IM2COL_BLOCK_SIZE,
                a, b, im2col.get(), static_cast<int>(IC), static_cast<int>(AT), static_cast<int>(BT),
                static_cast<int>(H), static_cast<int>(W), static_cast<int>(OD), static_cast<int>(OH), static_cast<int>(OW),
                lp0, lp1, lp2, src_t_offset, 0, static_cast<int>(OH));
        }
        for (int kd = 0; kd < 3; ++kd) {
            const int row_offset = use_skip_front_zero && kd == 0 ? static_cast<int>(M) : 0;
            const int gemm_rows = static_cast<int>(OD * M) - row_offset;
            const half * a_ptr = im2col.get() + (static_cast<int64_t>(kd) * M + row_offset) * K2;
            const half * b_ptr = weight.get() + static_cast<int64_t>(kd) * K2 * OC;
            half * c_ptr = out_data + row_offset;
            const half * beta = kd == 0 ? &beta_zero : &beta_one;
            CUBLAS_CHECK(cublasGemmEx(
                ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                gemm_rows, static_cast<int>(OC), static_cast<int>(K2),
                &alpha,
                a_ptr, CUDA_R_16F, static_cast<int>(K2),
                b_ptr, CUDA_R_16F, static_cast<int>(K2),
                beta,
                c_ptr, CUDA_R_16F, static_cast<int>(OD * M),
                CUBLAS_COMPUTE_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    } else {
        constexpr size_t kMaxIm2ColBytes = 128ull * 1024ull * 1024ull;
        const int64_t max_source_frames =
            std::max<int64_t>(3, static_cast<int64_t>(kMaxIm2ColBytes / (sizeof(half) * static_cast<size_t>(M) * static_cast<size_t>(K2))));
        const int64_t max_tile_frames = std::max<int64_t>(1, max_source_frames - 2);
        const int64_t tile_frames = std::min<int64_t>(OD, max_tile_frames);
        for (int64_t tile_start = 0; tile_start < OD; tile_start += tile_frames) {
            const int64_t current_frames = std::min<int64_t>(tile_frames, OD - tile_start);
            const int current_source_frames = static_cast<int>(current_frames + 2);
            const int64_t max_tile_rows =
                std::max<int64_t>(1, static_cast<int64_t>(kMaxIm2ColBytes / (sizeof(half) * static_cast<size_t>(current_source_frames) * static_cast<size_t>(OW) * static_cast<size_t>(K2))));
            const int64_t tile_oh = std::min<int64_t>(OH, max_tile_rows);
            for (int64_t y_start = 0; y_start < OH; y_start += tile_oh) {
                const int current_tile_oh = static_cast<int>(std::min<int64_t>(tile_oh, OH - y_start));
                const int64_t tile_m = static_cast<int64_t>(current_tile_oh) * OW;
                ggml_cuda_pool_alloc<half> im2col(ctx.pool(), static_cast<size_t>(current_source_frames * tile_m * K2));
                if (conv_lowering == GGML_CONV_3D_CONCAT_PAD_SPATIAL_GEMM_LOWERING_CUDA_TILED_C48) {
                    const int groups_c48 = (static_cast<int>(IC) + 47) / 48;
                    dim3 im2col_grid(groups_c48 * current_source_frames, (OW + 7) / 8, current_tile_oh);
                    dispatch_conv3d_concat_pad_spatial_im2col<8, 48>(
                        stream, im2col_grid, 48 * 9, a, b, im2col.get(),
                        static_cast<int>(IC), static_cast<int>(AT), static_cast<int>(BT),
                        static_cast<int>(H), static_cast<int>(W), static_cast<int>(current_frames), static_cast<int>(OH), static_cast<int>(OW),
                        lp0, lp1, lp2, src_t_offset + static_cast<int>(tile_start), static_cast<int>(y_start), current_tile_oh);
                } else {
                    dim3 im2col_grid(groups * current_source_frames, (OW + 7) / 8, current_tile_oh);
                    dispatch_conv3d_concat_pad_spatial_im2col<8, 28>(
                        stream, im2col_grid, CUDA_CONV3D_SPATIAL_IM2COL_BLOCK_SIZE,
                        a, b, im2col.get(), static_cast<int>(IC), static_cast<int>(AT), static_cast<int>(BT),
                        static_cast<int>(H), static_cast<int>(W), static_cast<int>(current_frames), static_cast<int>(OH), static_cast<int>(OW),
                        lp0, lp1, lp2, src_t_offset + static_cast<int>(tile_start), static_cast<int>(y_start), current_tile_oh);
                }
                for (int64_t frame = 0; frame < current_frames; ++frame) {
                    for (int kd = 0; kd < 3; ++kd) {
                        const half * a_ptr = im2col.get() + (static_cast<int64_t>(kd) + frame) * tile_m * K2;
                        const half * b_ptr = weight.get() + static_cast<int64_t>(kd) * K2 * OC;
                        half * c_ptr = out_data + (tile_start + frame) * M + y_start * OW;
                        const half * beta = kd == 0 ? &beta_zero : &beta_one;
                        CUBLAS_CHECK(cublasGemmEx(
                            ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                            static_cast<int>(tile_m), static_cast<int>(OC), static_cast<int>(K2),
                            &alpha,
                            a_ptr, CUDA_R_16F, static_cast<int>(K2),
                            b_ptr, CUDA_R_16F, static_cast<int>(K2),
                            beta,
                            c_ptr, CUDA_R_16F, static_cast<int>(OD * M),
                            CUBLAS_COMPUTE_16F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                    }
                }
            }
        }
    }

    if (dst->type == GGML_TYPE_F32) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
        to_fp32_cuda(out_data, reinterpret_cast<float *>(dst->data), OD * M * OC, stream);
    }
}
