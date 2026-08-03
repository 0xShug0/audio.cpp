#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::qwen3_asr {

struct Qwen3ASRGenerationOptions {
    int64_t max_new_tokens = 512;
    bool return_timestamps = false;
};

struct Qwen3ASRRequest {
    runtime::AudioBuffer audio;
    std::string context;
    std::string language;
    Qwen3ASRGenerationOptions generation;
};

struct Qwen3ASRResult {
    std::string text;
    std::string language;
    std::vector<runtime::WordTimestamp> word_timestamps;
};

struct Qwen3ASRPrompt {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> audio_token_positions;
    std::vector<int32_t> attention_mask;
};

struct Qwen3ASRAudioFeatures {
    std::vector<float> values;
    std::vector<int32_t> attention_mask;
    int64_t mel_bins = 0;
    int64_t frames = 0;
    int64_t encoder_tokens = 0;
};

struct Qwen3ASRAudioEmbeddings {
    std::vector<float> values;
    int64_t tokens = 0;
    int64_t hidden_size = 0;
};

struct Qwen3ASRGeneratedTokens {
    std::vector<int32_t> token_ids;
};

inline int64_t qwen3_asr_floor_div(int64_t numerator, int64_t denominator) {
    int64_t quotient = numerator / denominator;
    const int64_t remainder = numerator % denominator;
    if (remainder != 0 && ((remainder < 0) != (denominator < 0))) {
        --quotient;
    }
    return quotient;
}

inline int64_t qwen3_asr_audio_encoder_token_count(int64_t input_frames) {
    if (input_frames <= 0) {
        throw std::runtime_error("Qwen3 ASR requires positive feature frame count");
    }
    const int64_t input_lengths_leave = input_frames % 100;
    const int64_t feat_lengths = qwen3_asr_floor_div(input_lengths_leave - 1, 2) + 1;
    return qwen3_asr_floor_div(qwen3_asr_floor_div(feat_lengths - 1, 2) + 1 - 1, 2) + 1 +
        (input_frames / 100) * 13;
}

}  // namespace engine::models::qwen3_asr
