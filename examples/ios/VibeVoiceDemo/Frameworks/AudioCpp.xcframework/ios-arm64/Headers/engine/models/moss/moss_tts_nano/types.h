#pragma once

#include "engine/framework/runtime/model.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::moss_tts_nano {

struct MossTTSNanoSamplingOptions {
    bool do_sample = true;
    float text_temperature = 1.5F;
    float text_top_p = 1.0F;
    int text_top_k = 50;
    float audio_temperature = 1.7F;
    float audio_top_p = 0.8F;
    int audio_top_k = 25;
    float audio_repetition_penalty = 1.0F;
};

struct MossTTSNanoGenerationOptions {
    int64_t max_new_frames = 300;
    int64_t active_codebooks = 16;
    uint32_t seed = 0;
    MossTTSNanoSamplingOptions sampling;
};

struct MossTTSNanoAudioCodes {
    int64_t frames = 0;
    int64_t codebooks = 0;
    bool hit_max_new_frames = false;
    std::vector<int32_t> token_ids;
};

struct MossTTSNanoPrompt {
    std::vector<int32_t> input_ids;
    std::vector<uint8_t> attention_mask;
    int64_t rows = 0;
    int64_t row_width = 0;
};

struct MossTTSNanoRequest {
    std::string text;
    std::string prompt_text;
    runtime::AudioBuffer prompt_audio;
    bool has_prompt_audio = false;
    MossTTSNanoGenerationOptions generation;
};

}  // namespace engine::models::moss_tts_nano
