#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::vibevoice_asr {

struct VibeVoiceASRGenerationOptions {
    int64_t max_new_tokens = 32768;
    float temperature = 0.0f;
    float top_p = 1.0f;
    int64_t top_k = 50;
    float repetition_penalty = 1.0f;
    int num_beams = 1;
    uint64_t seed = 1234;
};

struct VibeVoiceASRRequest {
    runtime::AudioBuffer audio;
    std::string context;
    std::string language;
    VibeVoiceASRGenerationOptions generation;
};

struct VibeVoiceASRPrompt {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> attention_mask;
    std::vector<int32_t> speech_positions;
    int64_t speech_tokens = 0;
    double audio_seconds = 0.0;
};

struct VibeVoiceASRSpeechFeatures {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t hidden_size = 0;
    uint64_t next_rng_index = 0;
};

struct VibeVoiceASRGeneratedTokens {
    std::vector<int32_t> token_ids;
    std::string raw_text;
};

struct VibeVoiceASRSegment {
    double start_time = 0.0;
    double end_time = 0.0;
    std::string speaker_id;
    std::string text;
};

struct VibeVoiceASRDecoded {
    std::string text;
    std::string raw_text;
    std::vector<VibeVoiceASRSegment> segments;
};

}  // namespace engine::models::vibevoice_asr
