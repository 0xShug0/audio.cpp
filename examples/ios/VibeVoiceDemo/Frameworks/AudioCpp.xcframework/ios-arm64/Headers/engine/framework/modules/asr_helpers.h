#pragma once

#include <cstdint>
#include <vector>

namespace engine::modules {

std::vector<int32_t> make_asr_keep_mask(int64_t frames, int64_t valid_frames);

void fill_asr_keep_mask(std::vector<int32_t> & out, int64_t frames, int64_t valid_frames);

std::vector<float> make_asr_full_attention_bias(int64_t frames, int64_t valid_frames);

void fill_asr_full_attention_bias(std::vector<float> & out, int64_t frames, int64_t valid_frames);

void fill_asr_chunked_attention_bias(
    std::vector<float> & out,
    int64_t frames,
    int64_t valid_frames,
    int64_t left_context,
    int64_t right_context);

void fill_asr_stream_attention_bias(
    std::vector<float> & out,
    int64_t current_frames,
    int64_t key_frames,
    int64_t valid_cached_frames,
    int64_t q_offset,
    int64_t kv_offset,
    int64_t left_context,
    int64_t right_context);

}  // namespace engine::modules
