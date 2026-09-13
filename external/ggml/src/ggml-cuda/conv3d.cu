#include "conv3d.cuh"
#include "convert.cuh"

#include <algorithm>
#include <cstdio>

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
