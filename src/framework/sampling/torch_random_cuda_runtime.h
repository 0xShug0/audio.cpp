#pragma once

#include "engine/framework/sampling/torch_random.h"

#include <cstddef>
#include <cstdint>

namespace engine::sampling::detail {

void sample_topk_exponential_pairs_cuda(
    const void * device_logits_f32,
    int64_t songs,
    int64_t vocab,
    float guidance_scale,
    int64_t top_k,
    const uint64_t * seeds,
    const uint64_t * offset_blocks,
    uint64_t offset_step_blocks,
    const TorchCudaSamplingPolicy & policy,
    int32_t * out_codes);

void depth_frame_ensure_cuda(int64_t songs, int64_t levels, int64_t hidden_size, const TorchCudaSamplingPolicy & policy);
void depth_frame_begin_cuda(const uint64_t * seeds, const uint64_t * offset_blocks, int64_t songs, void * stream);
void depth_frame_sample_cuda(
    const void * device_logits_f32,
    int64_t level_index,
    int64_t songs,
    int64_t vocab,
    float guidance_scale,
    int64_t top_k,
    const TorchCudaSamplingPolicy & policy,
    void * stream);
void depth_frame_residual_fill_cuda(
    void * residual_ids_i32,
    int64_t previous_levels,
    int64_t songs,
    int64_t audio_vocab,
    void * stream);
void depth_frame_accumulate_hidden_cuda(
    const void * hidden_f32,
    int64_t level_index,
    int64_t songs,
    int64_t hidden_size,
    void * stream);
void depth_frame_end_cuda(
    int32_t * host_codes,
    float * host_hidden,
    int64_t levels,
    int64_t songs,
    int64_t hidden_size,
    void * stream);

void fill_torch_cuda_tensor_iterator_randn_cuda(
    float * output,
    size_t count,
    uint64_t seed,
    uint64_t offset_blocks,
    const TorchCudaSamplingPolicy & policy,
    TorchRandnPrecision precision);

}  // namespace engine::sampling::detail
