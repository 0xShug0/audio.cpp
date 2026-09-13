#include "im2col.cuh"

#define MAX_GRIDDIM_Y 65535
#define MAX_GRIDDIM_Z 65535

template <typename T>
static  __global__ void im2col_kernel(
        const float * x, T * dst,
        int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW, int64_t KW, int64_t KH,
        int64_t IC_IH_IW, int64_t IH_IW, int64_t N_OH, int64_t KH_KW, int64_t IC_KH_KW,
        int s0, int s1, int p0, int p1, int d0, int d1) {
    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= IC_KH_KW) {
        return;
    }

    const int64_t iic = i / (KH_KW);
    const int64_t rem = i - iic * KH_KW;
    const int64_t ikh = rem / KW;
    const int64_t ikw = rem - ikh * KW;

    for (int64_t iow = blockIdx.y; iow < OW; iow += MAX_GRIDDIM_Y) {
        for (int64_t iz = blockIdx.z; iz < N_OH; iz += MAX_GRIDDIM_Z) {
            const int64_t  in = iz / OH;
            const int64_t  ioh = iz - in * OH;

            const int64_t iiw = iow * s0 + ikw * d0 - p0;
            const int64_t iih = ioh * s1 + ikh * d1 - p1;

            const int64_t offset_dst =
                ((in * OH + ioh) * OW + iow) * IC_KH_KW + iic * KH_KW + ikh * KW + ikw;

            if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                dst[offset_dst] = 0.0f;
            } else {
                const int64_t offset_src = iic * IC_IH_IW + in * IH_IW;
                dst[offset_dst] = x[offset_src + iih * IW + iiw];
            }
        }
    }

    GGML_UNUSED(IC);
    GGML_UNUSED(KH);
}

template <typename T>
static __global__ void im2col_n_k3_pad1_kernel(
        const float * __restrict__ x, T * __restrict__ dst,
        int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW,
        int64_t groups, int64_t IC_KH_KW,
        int64_t IC_IH_IW, int64_t IH_IW) {
    const int64_t local = threadIdx.x;
    if (local >= 28 * 9) {
        return;
    }
    const int64_t group = blockIdx.x % groups;
    const int64_t in = blockIdx.x / groups;
    const int64_t iow = blockIdx.y;
    const int64_t ioh = blockIdx.z;

    const int64_t local_channel = local / 9;
    const int64_t iic = group * 28 + local_channel;
    if (iic >= IC) {
        return;
    }
    const int64_t rem0 = local - local_channel * 9;
    const int64_t ikh = rem0 / 3;
    const int64_t ikw = rem0 - ikh * 3;
    const int64_t iih = ioh + ikh - 1;
    const int64_t iiw = iow + ikw - 1;

    const int64_t offset_dst =
        ((in * OH + ioh) * OW + iow) * IC_KH_KW +
        iic * 9 +
        rem0;
    if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
        dst[offset_dst] = T(0.0f);
    } else {
        dst[offset_dst] = T(x[iic * IC_IH_IW + in * IH_IW + iih * IW + iiw]);
    }
}

template <typename T, int X_TILE>
static __global__ void im2col_n_k3_pad1_xtile_kernel(
        const float * __restrict__ x, T * __restrict__ dst,
        int IC, int IW, int IH, int OH, int OW,
        int groups, int IC_KH_KW,
        int IC_IH_IW, int IH_IW) {
    const int local = threadIdx.x;
    if (local >= 28 * 9) {
        return;
    }
    const int group = blockIdx.x % groups;
    const int in = blockIdx.x / groups;
    const int base_iow = blockIdx.y * X_TILE;
    const int ioh = blockIdx.z;

    const int local_channel = local / 9;
    const int iic = group * 28 + local_channel;
    if (iic >= IC) {
        return;
    }
    const int rem0 = local - local_channel * 9;
    const int ikh = rem0 / 3;
    const int ikw = rem0 - ikh * 3;
    const int iih = ioh + ikh - 1;

#pragma unroll
    for (int dx = 0; dx < X_TILE; ++dx) {
        const int iow = base_iow + dx;
        if (iow >= OW) {
            return;
        }
        const int iiw = iow + ikw - 1;
        const int offset_dst =
            ((in * OH + ioh) * OW + iow) * IC_KH_KW +
            iic * 9 +
            rem0;
        if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
            dst[offset_dst] = T(0.0f);
        } else {
            dst[offset_dst] = T(x[iic * IC_IH_IW + in * IH_IW + iih * IW + iiw]);
        }
    }
}

template <typename T, int X_TILE>
static __global__ void im2col_n_k3_nopad_xtile_kernel(
        const float * __restrict__ x, T * __restrict__ dst,
        int IC, int IW, int IH, int OH, int OW,
        int groups, int IC_KH_KW,
        int IC_IH_IW, int IH_IW) {
    const int local = threadIdx.x;
    if (local >= 28 * 9) {
        return;
    }
    const int group = blockIdx.x % groups;
    const int in = blockIdx.x / groups;
    const int base_iow = blockIdx.y * X_TILE;
    const int ioh = blockIdx.z;

    const int local_channel = local / 9;
    const int iic = group * 28 + local_channel;
    if (iic >= IC) {
        return;
    }
    const int rem0 = local - local_channel * 9;
    const int ikh = rem0 / 3;
    const int ikw = rem0 - ikh * 3;
    const int iih = ioh + ikh;

#pragma unroll
    for (int dx = 0; dx < X_TILE; ++dx) {
        const int iow = base_iow + dx;
        if (iow >= OW) {
            return;
        }
        const int iiw = iow + ikw;
        const int offset_dst =
            ((in * OH + ioh) * OW + iow) * IC_KH_KW +
            iic * 9 +
            rem0;
        dst[offset_dst] = T(x[iic * IC_IH_IW + in * IH_IW + iih * IW + iiw]);
    }

    GGML_UNUSED(IH);
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
template <typename T>
static void im2col_cuda(const float * x, T* dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N, int64_t IC_IH_IW, int64_t IH_IW,
    int s0,int s1,int p0,int p1,int d0,int d1, int32_t lowering, cudaStream_t stream) {
    const int64_t IC_KH_KW = IC * KH * KW;
    const int64_t num_blocks = (IC_KH_KW + CUDA_IM2COL_BLOCK_SIZE - 1) / CUDA_IM2COL_BLOCK_SIZE;
    const int64_t N_OH = N * OH;
    const int64_t KH_KW = KW*KH;
    const bool use_n_k3_pad1_kernel = lowering == GGML_IM2COL_2D_LOWERING_CUDA_N_K3_PAD1_X8;
    const bool use_n_k3_pad1_x8_kernel = lowering == GGML_IM2COL_2D_LOWERING_CUDA_N_K3_PAD1_X8;
    const bool use_n_k3_nopad_x8_kernel = lowering == GGML_IM2COL_2D_LOWERING_CUDA_N_K3_NOPAD_X8;
    if (use_n_k3_nopad_x8_kernel &&
        KW == 3 && KH == 3 &&
        s0 == 1 && s1 == 1 &&
        p0 == 0 && p1 == 0 &&
        d0 == 1 && d1 == 1 &&
        IC <= INT_MAX && IW <= INT_MAX && IH <= INT_MAX &&
        OH <= INT_MAX && OW <= INT_MAX &&
        IC_KH_KW <= INT_MAX && IC_IH_IW <= INT_MAX && IH_IW <= INT_MAX) {
        const int64_t channel_groups = (IC + 27) / 28;
        if (channel_groups <= INT_MAX) {
            dim3 block_nums(channel_groups * N, MIN((OW + 7) / 8, MAX_GRIDDIM_Y), MIN(OH, MAX_GRIDDIM_Z));
            im2col_n_k3_nopad_xtile_kernel<T, 8><<<block_nums, 256, 0, stream>>>(
                x, dst,
                static_cast<int>(IC), static_cast<int>(IW), static_cast<int>(IH),
                static_cast<int>(OH), static_cast<int>(OW), static_cast<int>(channel_groups),
                static_cast<int>(IC_KH_KW), static_cast<int>(IC_IH_IW), static_cast<int>(IH_IW));
            return;
        }
    }
    if (use_n_k3_pad1_kernel &&
        KW == 3 && KH == 3 &&
        s0 == 1 && s1 == 1 &&
        p0 == 1 && p1 == 1 &&
        d0 == 1 && d1 == 1) {
        const int64_t channel_groups = (IC + 27) / 28;
        if (use_n_k3_pad1_x8_kernel &&
            IC <= INT_MAX && IW <= INT_MAX && IH <= INT_MAX &&
            OH <= INT_MAX && OW <= INT_MAX && channel_groups <= INT_MAX &&
            IC_KH_KW <= INT_MAX && IC_IH_IW <= INT_MAX && IH_IW <= INT_MAX) {
            dim3 block_nums(channel_groups * N, MIN((OW + 7) / 8, MAX_GRIDDIM_Y), MIN(OH, MAX_GRIDDIM_Z));
            im2col_n_k3_pad1_xtile_kernel<T, 8><<<block_nums, 256, 0, stream>>>(
                x, dst,
                static_cast<int>(IC), static_cast<int>(IW), static_cast<int>(IH),
                static_cast<int>(OH), static_cast<int>(OW), static_cast<int>(channel_groups),
                static_cast<int>(IC_KH_KW), static_cast<int>(IC_IH_IW), static_cast<int>(IH_IW));
        } else {
            dim3 block_nums(channel_groups * N, MIN(OW, MAX_GRIDDIM_Y), MIN(OH, MAX_GRIDDIM_Z));
            im2col_n_k3_pad1_kernel<<<block_nums, 256, 0, stream>>>(
                x, dst, IC, IW, IH, OH, OW, channel_groups, IC_KH_KW, IC_IH_IW, IH_IW);
        }
        return;
    }
    dim3 block_nums(num_blocks, MIN(OW, MAX_GRIDDIM_Y), MIN(N_OH, MAX_GRIDDIM_Z));
    im2col_kernel<<<block_nums, MIN(IC_KH_KW, CUDA_IM2COL_BLOCK_SIZE) , 0, stream>>>(x, dst, IC, IW, IH, OH, OW, KW, KH,
                                                                                     IC_IH_IW, IH_IW, N_OH, KH_KW, IC_KH_KW,
                                                                                     s0, s1, p0, p1, d0, d1);
}

static void im2col_cuda_f16(const float * x, half * dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N, int64_t IC_IH_IW, int64_t IH_IW,
    int s0,int s1,int p0,int p1,int d0,int d1, int32_t lowering, cudaStream_t stream) {

    im2col_cuda<half>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, lowering, stream);
}

static void im2col_cuda_f32(const float * x, float * dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N, int64_t IC_IH_IW, int64_t IH_IW,
    int s0,int s1,int p0,int p1,int d0,int d1, int32_t lowering, cudaStream_t stream) {

    im2col_cuda<float>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, lowering, stream);
}

void ggml_cuda_op_im2col(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t*)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t*)(dst->op_params))[5];

    const bool is_2D = ((const int32_t*)(dst->op_params))[6] == 1;

    const int64_t IC = src1->ne[is_2D ? 2 : 1];
    const int64_t IH = is_2D ? src1->ne[1] : 1;
    const int64_t IW =         src1->ne[0];

    const int64_t KH = is_2D ? src0->ne[1] : 1;
    const int64_t KW =         src0->ne[0];

    const int64_t OH = is_2D ? dst->ne[2] : 1;
    const int64_t OW =         dst->ne[1];

    const int64_t IC_IH_IW = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    const int64_t N        = src1->ne[is_2D ? 3 : 2];
    const int64_t IH_IW    = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32
    const int32_t lowering     = ggml_get_op_params_i32(dst, 7);

    if(dst->type == GGML_TYPE_F16) {
        im2col_cuda_f16(src1_d, (half *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, lowering, stream);
    } else {
        im2col_cuda_f32(src1_d, (float *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, lowering, stream);
    }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
template <typename T>
static  __global__ void im2col_3d_kernel(
        const float * src, T * dst,
        int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
        int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
        int64_t OH_OW, int64_t KD_KH_KW, int64_t ID_IH_IW, int64_t KH_KW, int64_t IH_IW, int64_t IC_ID_IH_IW,
        int64_t IC_KD_KH_KW, int64_t OW_KD_KH_KW, int64_t OD_OH_OW_IC_KD_KH_KW, int64_t OH_OW_IC_KD_KH_KW,
        int64_t OW_IC_KD_KH_KW, int64_t N_OD_OH, int64_t OD_OH,
        int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
        int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2) {
    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= IC_KD_KH_KW) {
        return;
    }
    GGML_UNUSED(N); GGML_UNUSED(OC); GGML_UNUSED(OH_OW); GGML_UNUSED(OD); GGML_UNUSED(OW); GGML_UNUSED(KD); GGML_UNUSED(KH);
    GGML_UNUSED(ID_IH_IW); GGML_UNUSED(IH_IW); GGML_UNUSED(IC_ID_IH_IW); GGML_UNUSED(OW_KD_KH_KW);

    const int64_t iic = i / KD_KH_KW;
    const int64_t ikd = (i - iic * KD_KH_KW) / KH_KW;
    const int64_t ikh = (i - iic * KD_KH_KW - ikd * KH_KW) / KW;
    const int64_t ikw = i % KW;

    for (int64_t iow = blockIdx.y; iow < OW; iow += MAX_GRIDDIM_Y) {
        for (int64_t iz = blockIdx.z; iz < N_OD_OH; iz += MAX_GRIDDIM_Z) {
            const int64_t in  = iz / OD_OH;
            const int64_t iod = (iz - in*OD_OH) / OH;
            const int64_t ioh = iz % OH;

            const int64_t iiw = iow * s0 + ikw * d0 - p0;
            const int64_t iih = ioh * s1 + ikh * d1 - p1;
            const int64_t iid = iod * s2 + ikd * d2 - p2;

            const int64_t offset_dst = in*OD_OH_OW_IC_KD_KH_KW + iod*OH_OW_IC_KD_KH_KW + ioh*OW_IC_KD_KH_KW + iow*IC_KD_KH_KW + iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw;

            if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                dst[offset_dst] = 0.0f;
            } else {
                const int64_t offset_src = ((in * IC + iic) * stride_q) + (iid * stride_z) + (iih * stride_y) + (iiw * stride_x);
                dst[offset_dst] = src[offset_src];
            }
        }
    }
}

template <typename T>
static __global__ void im2col_3d_n1_k3_nopad_kernel(
        const float * __restrict__ src, T * __restrict__ dst,
        int64_t IC, int64_t IH, int64_t IW, int64_t OH, int64_t OW,
        int64_t groups, int64_t IC_KD_KH_KW,
        int64_t OH_OW_IC_KD_KH_KW, int64_t OW_IC_KD_KH_KW,
        int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x) {
    const int64_t local = threadIdx.x;
    if (local >= 9 * 27) {
        return;
    }
    const int64_t group = blockIdx.x % groups;
    const int64_t iod = blockIdx.x / groups;
    const int64_t iow = blockIdx.y;
    const int64_t ioh = blockIdx.z;

    const int64_t local_channel = local / 27;
    const int64_t iic = group * 9 + local_channel;
    if (iic >= IC) {
        return;
    }
    const int64_t rem0 = local - local_channel * 27;
    const int64_t ikd = rem0 / 9;
    const int64_t rem1 = rem0 - ikd * 9;
    const int64_t ikh = rem1 / 3;
    const int64_t ikw = rem1 - ikh * 3;

    const int64_t offset_dst =
        iod * OH_OW_IC_KD_KH_KW +
        ioh * OW_IC_KD_KH_KW +
        iow * IC_KD_KH_KW +
        iic * 27 +
        rem0;
    const int64_t offset_src =
        iic * stride_q +
        (iod + ikd) * stride_z +
        (ioh + ikh) * stride_y +
        (iow + ikw) * stride_x;
    dst[offset_dst] = T(src[offset_src]);

    GGML_UNUSED(IH);
    GGML_UNUSED(IW);
}

template <typename T, int X_TILE>
static __global__ void im2col_3d_n1_k3_nopad_xtile_kernel(
        const float * __restrict__ src, T * __restrict__ dst,
        int IC, int OH, int OW,
        int groups, int IC_KD_KH_KW,
        int OH_OW_IC_KD_KH_KW, int OW_IC_KD_KH_KW,
        int stride_q, int stride_z, int stride_y, int stride_x) {
    const int local = threadIdx.x;
    if (local >= 9 * 27) {
        return;
    }
    const int group = blockIdx.x % groups;
    const int iod = blockIdx.x / groups;
    const int base_iow = blockIdx.y * X_TILE;
    const int ioh = blockIdx.z;

    const int local_channel = local / 27;
    const int iic = group * 9 + local_channel;
    if (iic >= IC) {
        return;
    }
    const int rem0 = local - local_channel * 27;
    const int ikd = rem0 / 9;
    const int rem1 = rem0 - ikd * 9;
    const int ikh = rem1 / 3;
    const int ikw = rem1 - ikh * 3;

#pragma unroll
    for (int dx = 0; dx < X_TILE; ++dx) {
        const int iow = base_iow + dx;
        if (iow >= OW) {
            return;
        }
        const int offset_dst =
            iod * OH_OW_IC_KD_KH_KW +
            ioh * OW_IC_KD_KH_KW +
            iow * IC_KD_KH_KW +
            iic * 27 +
            rem0;
        const int offset_src =
            iic * stride_q +
            (iod + ikd) * stride_z +
            (ioh + ikh) * stride_y +
            (iow + ikw) * stride_x;
        dst[offset_dst] = T(src[offset_src]);
    }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
template <typename T>
static void im2col_3d_cuda(const float * src, T* dst,
    int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
    int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
    int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
    int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, int32_t lowering, cudaStream_t stream) {
    const int64_t OH_OW = OH*OW;
    const int64_t KD_KH_KW = KD*KH*KW;
    const int64_t ID_IH_IW = ID*IH*IW;
    const int64_t KH_KW = KH*KW;
    const int64_t IH_IW = IH*IW;
    const int64_t IC_KD_KH_KW = IC*KD*KH*KW;
    const int64_t OW_KD_KH_KW = OW*KD*KH*KW;
    const int64_t N_OD_OH = N*OD*OH;
    const int64_t OD_OH = OD*OH;
    const int64_t IC_ID_IH_IW = IC*ID*IH*IW;
    const int64_t OD_OH_OW_IC_KD_KH_KW = OD*OH*OW*IC*KD*KH*KW;
    const int64_t OH_OW_IC_KD_KH_KW = OH*OW*IC*KD*KH*KW;
    const int64_t OW_IC_KD_KH_KW = OW*IC*KD*KH*KW;
    const int64_t num_blocks = (IC_KD_KH_KW + CUDA_IM2COL_BLOCK_SIZE - 1) / CUDA_IM2COL_BLOCK_SIZE;
    const int64_t N_OD_OH_OW_IC_KD_KH_KW = N*OD*OH*OW*IC*KD*KH*KW;
    const bool use_n1_k3_nopad_kernel = lowering == GGML_IM2COL_3D_LOWERING_CUDA_N1_K3_NOPAD_X8;
    if (use_n1_k3_nopad_kernel &&
        N == 1 && KD == 3 && KH == 3 && KW == 3 &&
        s0 == 1 && s1 == 1 && s2 == 1 &&
        p0 == 0 && p1 == 0 && p2 == 0 &&
        d0 == 1 && d1 == 1 && d2 == 1) {
        const bool dst_offsets_fit_i32 = N_OD_OH_OW_IC_KD_KH_KW <= INT_MAX;
        const int64_t channel_groups = (IC + 8) / 9;
        if (IC <= INT_MAX && OH <= INT_MAX && OW <= INT_MAX &&
            channel_groups <= INT_MAX && IC_KD_KH_KW <= INT_MAX &&
            dst_offsets_fit_i32 &&
            OH_OW_IC_KD_KH_KW <= INT_MAX && OW_IC_KD_KH_KW <= INT_MAX &&
            stride_q <= INT_MAX && stride_z <= INT_MAX && stride_y <= INT_MAX && stride_x <= INT_MAX) {
            dim3 block_nums(channel_groups * OD, MIN((OW + 7) / 8, MAX_GRIDDIM_Y), MIN(OH, MAX_GRIDDIM_Z));
            im2col_3d_n1_k3_nopad_xtile_kernel<T, 8><<<block_nums, 256, 0, stream>>>(
                src, dst,
                static_cast<int>(IC), static_cast<int>(OH), static_cast<int>(OW),
                static_cast<int>(channel_groups), static_cast<int>(IC_KD_KH_KW),
                static_cast<int>(OH_OW_IC_KD_KH_KW), static_cast<int>(OW_IC_KD_KH_KW),
                static_cast<int>(stride_q), static_cast<int>(stride_z), static_cast<int>(stride_y), static_cast<int>(stride_x));
            return;
        }
        dim3 block_nums(channel_groups * OD, MIN(OW, MAX_GRIDDIM_Y), MIN(OH, MAX_GRIDDIM_Z));
        im2col_3d_n1_k3_nopad_kernel<<<block_nums, 256, 0, stream>>>(
            src, dst, IC, IH, IW, OH, OW, channel_groups, IC_KD_KH_KW,
            OH_OW_IC_KD_KH_KW, OW_IC_KD_KH_KW,
            stride_q, stride_z, stride_y, stride_x);
        return;
    }
    dim3 block_nums(num_blocks, MIN(OW, MAX_GRIDDIM_Y), MIN(N_OD_OH, MAX_GRIDDIM_Z));
    im2col_3d_kernel<<<block_nums, MIN(IC_KD_KH_KW, CUDA_IM2COL_BLOCK_SIZE) , 0, stream>>>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                                                                                           OH_OW, KD_KH_KW, ID_IH_IW, KH_KW, IH_IW, IC_ID_IH_IW,
                                                                                           IC_KD_KH_KW, OW_KD_KH_KW, OD_OH_OW_IC_KD_KH_KW,
                                                                                           OH_OW_IC_KD_KH_KW, OW_IC_KD_KH_KW, N_OD_OH, OD_OH,
                                                                                           stride_q, stride_z, stride_y, stride_x,
                                                                                           s0, s1, s2, p0, p1, p2, d0, d1, d2);
}

static void im2col_3d_cuda_f16(const float * src, half * dst,
    int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
    int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
    int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
    int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, int32_t lowering, cudaStream_t stream) {

    im2col_3d_cuda<half>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                         stride_q, stride_z, stride_y, stride_x,
                         s0, s1, s2, p0, p1, p2, d0, d1, d2, lowering, stream);
}

static void im2col_3d_cuda_f32(const float * src, float * dst,
    int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
    int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
    int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
    int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, int32_t lowering, cudaStream_t stream) {

    im2col_3d_cuda<float>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                          stride_q, stride_z, stride_y, stride_x,
                          s0, s1, s2, p0, p1, p2, d0, d1, d2, lowering, stream);
}

void ggml_cuda_op_im2col_3d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t s2 = ((const int32_t *)(dst->op_params))[2];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[3];
    const int32_t p1 = ((const int32_t *)(dst->op_params))[4];
    const int32_t p2 = ((const int32_t *)(dst->op_params))[5];
    const int32_t d0 = ((const int32_t *)(dst->op_params))[6];
    const int32_t d1 = ((const int32_t *)(dst->op_params))[7];
    const int32_t d2 = ((const int32_t *)(dst->op_params))[8];
    const int32_t IC = ((const int32_t *)(dst->op_params))[9];

    const int64_t N  = ne13 / IC;
    const int64_t ID = ne12;
    const int64_t IH = ne11;
    const int64_t IW = ne10;

    const int64_t OC = ne03 / IC;
    const int64_t KD = ne02;
    const int64_t KH = ne01;
    const int64_t KW = ne00;

    const int64_t OD = ne3 / N;
    const int64_t OH = ne2;
    const int64_t OW = ne1;

    const size_t  es       = ggml_element_size(src1);
    const int64_t stride_x = src1->nb[0] / es;
    const int64_t stride_y = src1->nb[1] / es;
    const int64_t stride_z = src1->nb[2] / es;
    const int64_t stride_q = src1->nb[3] / es;
    const int32_t lowering     = ggml_get_op_params_i32(dst, 10);

    if(dst->type == GGML_TYPE_F16) {
        im2col_3d_cuda_f16(src1_d, (half *) dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                           stride_q, stride_z, stride_y, stride_x,
                           s0, s1, s2, p0, p1, p2, d0, d1, d2, lowering, stream);
    } else {
        im2col_3d_cuda_f32(src1_d, (float *) dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                           stride_q, stride_z, stride_y, stride_x,
                           s0, s1, s2, p0, p1, p2, d0, d1, d2, lowering, stream);
    }
}
