//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "conv.hpp"

static  void conv_transpose_1d_kernel(
        const int s0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2,
        const int src1_ne0, const int dst_ne0,
        const float * src0, const float * src1,  float * dst,
        const sycl::nd_item<3> &item_ct1) {
    int global_index = item_ct1.get_local_id(2) +
                       item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (global_index >= output_size) {
        return;
    }

    int out_index = global_index / dst_ne0;

    float accumulator = 0;

    const int idx = global_index % dst_ne0;

    // only inputs with i*s0 <= idx < i*s0 + src0_ne0 contribute
    const int i_begin = idx >= src0_ne0 ? (idx - src0_ne0) / s0 + 1 : 0;
    const int i_end   = sycl::min(src1_ne0, idx / s0 + 1);

    for (int c = 0; c < src0_ne2; c++) {
        int kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        int input_offset = src1_ne0 * c;

        for (int i = i_begin; i < i_end; i++) {
            int weight_idx = idx - i*s0;

            float kernel_weight = src0[kernel_offset + weight_idx];
            float input_value =  src1[input_offset+i];

            accumulator += kernel_weight * input_value;
        }
    }
    dst[global_index] = accumulator;
}

static void conv_transpose_1d_f32_f32_sycl(
    const int s0, const int output_size,
    const int src0_ne0, const int src0_ne1, const int src0_ne2,
    const int src1_ne0, const int dst_ne0,
    const float *src0, const float *src1, float *dst,
    const queue_ptr& stream) {

    const int num_blocks = (output_size + SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE;
    const sycl::range<3> block_dims(1, 1, SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, 1, num_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(
            block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv_transpose_1d_kernel(
                s0, output_size,
                src0_ne0, src0_ne1, src0_ne2,
                src1_ne0, dst_ne0,
                src0, src1, dst, item_ct1);
        });
}

void ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];
    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;

    float * dst_d = (float *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const int32_t * opts = (const int32_t *)dst->op_params;

    const int s0 = opts[0];
    GGML_ASSERT(s0 > 0);

    const int64_t output_size = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_sycl(s0, output_size,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

// Depthwise 2D convolution. Each output sample sums the kernel window over the input, skipping
// positions outside it. CHANNELS_FIRST selects the two memory layouts ggml uses:
//   false: [W, H, C, N] contiguous, each channel a dense plane
//   true:  channels contiguous at every (x, y), kernel laid out the same way
template <bool CHANNELS_FIRST>
static void conv_2d_dw_f32_sycl(const float * knl, const float * src, float * dst, const int64_t channels, const int64_t batch,
                                const int64_t src_w, const int64_t src_h, const int64_t dst_w, const int64_t dst_h,
                                const int64_t knl_w, const int64_t knl_h, const int stride_x, const int stride_y,
                                const int pad_x, const int pad_y, const int dilation_x, const int dilation_y,
                                dpct::queue_ptr stream) {
    stream->parallel_for(sycl::range<1>(dst_w * dst_h * channels * batch), [=](sycl::id<1> id) {
        int64_t gid = id[0];
        int64_t dst_x, dst_y, channel, plane;
        if constexpr (CHANNELS_FIRST) {
            channel = gid % channels; gid /= channels;
            dst_x = gid % dst_w; gid /= dst_w;
            dst_y = gid % dst_h;
            plane = gid / dst_h;  // batch index
        } else {
            dst_x = gid % dst_w; gid /= dst_w;
            dst_y = gid % dst_h;
            plane = gid / dst_h;  // channel + batch * channels
            channel = plane % channels;
        }
        const float * knl_ptr = knl;
        const float * src_ptr = src;
        int64_t step = 1;
        if constexpr (CHANNELS_FIRST) {
            knl_ptr += channel;
            src_ptr += plane * src_w * src_h * channels + channel;
            step = channels;
        } else {
            knl_ptr += channel * knl_w * knl_h;
            src_ptr += plane * src_w * src_h;
        }
        float sum = 0.0f;
        for (int64_t knl_y = 0; knl_y < knl_h; ++knl_y) {
            const int64_t src_y = dst_y * stride_y + knl_y * dilation_y - pad_y;
            if (src_y < 0 || src_y >= src_h) {
                continue;
            }
            for (int64_t knl_x = 0; knl_x < knl_w; ++knl_x) {
                const int64_t src_x = dst_x * stride_x + knl_x * dilation_x - pad_x;
                if (src_x < 0 || src_x >= src_w) {
                    continue;
                }
                sum += knl_ptr[(knl_y * knl_w + knl_x) * step] * src_ptr[(src_y * src_w + src_x) * step];
            }
        }
        dst[id] = sum;
    });
}

void ggml_sycl_op_conv_2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * src = dst->src[1];
    GGML_ASSERT(kernel->type == GGML_TYPE_F32 && src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->ne[3] == src->ne[2] && dst->ne[3] == src->ne[3]);

    const int32_t * o = reinterpret_cast<const int32_t *>(dst->op_params);
    dpct::queue_ptr stream = ctx.stream();
    if (ggml_is_contiguous(src)) {
        conv_2d_dw_f32_sycl<false>(static_cast<const float *>(kernel->data), static_cast<const float *>(src->data),
                                   static_cast<float *>(dst->data), src->ne[2], src->ne[3], src->ne[0], src->ne[1],
                                   dst->ne[0], dst->ne[1], kernel->ne[0], kernel->ne[1], o[0], o[1], o[2], o[3], o[4], o[5],
                                   stream);
    } else {
        GGML_ASSERT(ggml_is_contiguous_channels(src));
        GGML_ASSERT(kernel->nb[0] >= kernel->nb[2] && kernel->nb[1] >= kernel->nb[0]);
        conv_2d_dw_f32_sycl<true>(static_cast<const float *>(kernel->data), static_cast<const float *>(src->data),
                                  static_cast<float *>(dst->data), src->ne[2], src->ne[3], src->ne[0], src->ne[1],
                                  dst->ne[0], dst->ne[1], kernel->ne[0], kernel->ne[1], o[0], o[1], o[2], o[3], o[4], o[5],
                                  stream);
    }
}
