#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::higgs_audio_stt {

struct HiggsAudioSTTGenerationOptions {
    int64_t max_new_tokens = 1024;
    bool enable_thinking = true;
};

struct HiggsAudioSTTRequest {
    runtime::AudioBuffer audio;
    std::string context;
    std::string language;
    HiggsAudioSTTGenerationOptions generation;
};

struct HiggsAudioSTTResult {
    std::string text;
    std::string language;
};

struct HiggsAudioSTTPrompt {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> audio_token_positions;
    std::vector<int32_t> attention_mask;
};

struct HiggsAudioSTTAudioFeatures {
    std::vector<float> values;
    std::vector<int32_t> attention_mask;
    int64_t mel_bins = 0;
    int64_t chunks = 0;
    int64_t frames = 0;
    std::vector<int64_t> valid_frames;
    std::vector<int64_t> encoder_tokens_per_chunk;
    int64_t encoder_tokens = 0;
};

struct HiggsAudioSTTAudioEmbeddings {
    std::vector<float> values;
    int64_t tokens = 0;
    int64_t hidden_size = 0;
};

struct HiggsAudioSTTGeneratedTokens {
    std::vector<int32_t> token_ids;
};

inline int64_t higgs_audio_stt_floor_div(int64_t numerator, int64_t denominator) {
    int64_t quotient = numerator / denominator;
    const int64_t remainder = numerator % denominator;
    if (remainder != 0 && ((remainder < 0) != (denominator < 0))) {
        --quotient;
    }
    return quotient;
}

inline int64_t higgs_audio_stt_conv_frames(int64_t input_frames) {
    if (input_frames <= 0) {
        throw std::runtime_error("Higgs Audio STT requires positive feature frame count");
    }
    return higgs_audio_stt_floor_div(input_frames - 1, 2) + 1;
}

inline int64_t higgs_audio_stt_audio_tower_token_count(int64_t input_frames) {
    const int64_t conv = higgs_audio_stt_conv_frames(input_frames);
    return higgs_audio_stt_floor_div(conv - 2, 2) + 1;
}

inline int64_t higgs_audio_stt_audio_encoder_token_count(int64_t input_frames, int64_t projector_stride) {
    if (projector_stride <= 0) {
        throw std::runtime_error("Higgs Audio STT requires positive projector stride");
    }
    const int64_t tower_tokens = higgs_audio_stt_audio_tower_token_count(input_frames);
    return higgs_audio_stt_floor_div(tower_tokens - 1, projector_stride) + 1;
}

}  // namespace engine::models::higgs_audio_stt
