#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::community_models::confucius4_r2t2 {

struct R2T2ASRGenerationOptions {
    int64_t max_new_tokens = 512;
    bool reuse_graphs = false;
    bool return_timestamps = false;
    bool clamp_timestamps_to_audio = false;
};

struct R2T2ASRRequest {
    runtime::AudioBuffer audio;
    std::string context;
    std::string language;
    R2T2ASRGenerationOptions generation;
};

struct R2T2ASRResult {
    std::string text;
    std::string language;
    std::vector<runtime::WordTimestamp> word_timestamps;
};

struct R2T2ASRPrompt {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> audio_token_positions;
    std::vector<int32_t> attention_mask;
};

struct R2T2ASRAudioFeatures {
    std::vector<float> values;
    std::vector<int32_t> attention_mask;
    int64_t mel_bins = 0;
    int64_t frames = 0;
    int64_t encoder_tokens = 0;
};

struct R2T2ASRAudioEmbeddings {
    std::vector<float> values;
    int64_t tokens = 0;
    int64_t hidden_size = 0;
};

struct R2T2ASRGeneratedTokens {
    std::vector<int32_t> token_ids;
};

inline int64_t confucius4_r2t2_floor_div(int64_t numerator, int64_t denominator) {
    int64_t quotient = numerator / denominator;
    const int64_t remainder = numerator % denominator;
    if (remainder != 0 && ((remainder < 0) != (denominator < 0))) {
        --quotient;
    }
    return quotient;
}

inline int64_t confucius4_r2t2_audio_encoder_token_count(int64_t input_frames) {
    if (input_frames <= 0) {
        throw std::runtime_error("R2T2 ASR requires positive feature frame count");
    }
    const int64_t input_lengths_leave = input_frames % 100;
    const int64_t feat_lengths = confucius4_r2t2_floor_div(input_lengths_leave - 1, 2) + 1;
    return confucius4_r2t2_floor_div(confucius4_r2t2_floor_div(feat_lengths - 1, 2) + 1 - 1, 2) + 1 +
        (input_frames / 100) * 13;
}

}  // namespace engine::community_models::confucius4_r2t2
