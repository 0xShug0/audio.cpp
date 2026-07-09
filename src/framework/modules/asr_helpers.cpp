#include "engine/framework/modules/asr_helpers.h"

#include <algorithm>
#include <stdexcept>

namespace engine::modules {

std::vector<int32_t> make_asr_keep_mask(int64_t frames, int64_t valid_frames) {
    std::vector<int32_t> mask;
    fill_asr_keep_mask(mask, frames, valid_frames);
    return mask;
}

void fill_asr_keep_mask(std::vector<int32_t> & out, int64_t frames, int64_t valid_frames) {
    if (frames < 0) {
        throw std::runtime_error("ASR keep mask frame count must be non-negative");
    }
    out.assign(static_cast<size_t>(frames), 0);
    const int64_t valid = std::clamp<int64_t>(valid_frames, 0, frames);
    for (int64_t i = 0; i < valid; ++i) {
        out[static_cast<size_t>(i)] = 1;
    }
}

std::vector<float> make_asr_full_attention_bias(int64_t frames, int64_t valid_frames) {
    std::vector<float> mask;
    fill_asr_full_attention_bias(mask, frames, valid_frames);
    return mask;
}

void fill_asr_full_attention_bias(std::vector<float> & out, int64_t frames, int64_t valid_frames) {
    if (frames < 0) {
        throw std::runtime_error("ASR attention bias frame count must be non-negative");
    }
    out.assign(static_cast<size_t>(frames * frames), -1.0e9f);
    const int64_t valid = std::clamp<int64_t>(valid_frames, 0, frames);
    for (int64_t q = 0; q < valid; ++q) {
        for (int64_t k = 0; k < valid; ++k) {
            out[static_cast<size_t>(q * frames + k)] = 0.0f;
        }
    }
}

void fill_asr_chunked_attention_bias(
    std::vector<float> & out,
    int64_t frames,
    int64_t valid_frames,
    int64_t left_context,
    int64_t right_context) {
    if (frames < 0 || right_context < 0) {
        throw std::runtime_error("ASR chunked attention bias received invalid dimensions");
    }
    out.assign(static_cast<size_t>(frames * frames), -1.0e9f);
    const int64_t valid = std::clamp<int64_t>(valid_frames, 0, frames);
    const int64_t chunk = right_context + 1;
    const int64_t left_chunks = left_context >= 0 ? left_context / chunk : 10000;
    for (int64_t q = 0; q < valid; ++q) {
        const int64_t q_chunk = q / chunk;
        for (int64_t k = 0; k < valid; ++k) {
            const int64_t k_chunk = k / chunk;
            const int64_t diff = q_chunk - k_chunk;
            if (diff >= 0 && diff <= left_chunks) {
                out[static_cast<size_t>(q * frames + k)] = 0.0f;
            }
        }
    }
}

void fill_asr_stream_attention_bias(
    std::vector<float> & out,
    int64_t current_frames,
    int64_t key_frames,
    int64_t valid_cached_frames,
    int64_t q_offset,
    int64_t kv_offset,
    int64_t left_context,
    int64_t right_context) {
    if (current_frames < 0 || key_frames < current_frames || right_context < 0) {
        throw std::runtime_error("ASR stream attention bias received invalid dimensions");
    }
    out.assign(static_cast<size_t>(current_frames * key_frames), -1.0e9f);
    const int64_t valid_cache = std::clamp<int64_t>(valid_cached_frames, 0, key_frames - current_frames);
    const int64_t valid_start = key_frames - current_frames - valid_cache;
    const int64_t valid_keys = valid_cache + current_frames;
    const int64_t chunk = right_context + 1;
    const int64_t left_chunks = left_context >= 0 ? left_context / chunk : 10000;
    for (int64_t q = 0; q < current_frames; ++q) {
        const int64_t global_q = q_offset + q;
        const int64_t q_chunk = global_q / chunk;
        for (int64_t k = 0; k < valid_keys; ++k) {
            const int64_t key_index = valid_start + k;
            const int64_t global_k = kv_offset + k;
            const int64_t k_chunk = global_k / chunk;
            const int64_t diff = q_chunk - k_chunk;
            if (diff >= 0 && diff <= left_chunks) {
                out[static_cast<size_t>(q * key_frames + key_index)] = 0.0f;
            }
        }
    }
}

}  // namespace engine::modules
