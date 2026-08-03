#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::voxtral_realtime {

struct VoxtralRealtimeFeatures {
    std::vector<float> values;
    int64_t mel_bins = 0;
    int64_t frames = 0;
};

struct VoxtralRealtimePrompt {
    std::vector<int32_t> input_ids;
    int64_t audio_tokens = 0;
    int64_t num_delay_tokens = 0;
};

struct VoxtralRealtimeAudioEmbeddings {
    std::vector<float> values;
    int64_t tokens = 0;
    int64_t hidden_size = 0;
};

struct VoxtralRealtimeAudioEncoderStreamState {
    bool first_chunk = true;
    int64_t seen_encoder_steps = 0;
    int64_t cached_encoder_steps = 0;
};

struct VoxtralRealtimeGenerationOptions {
    int64_t max_new_tokens = 0;
    bool max_new_tokens_set = false;
    bool do_sample = false;
    float temperature = 1.0F;
    float top_p = 1.0F;
    int64_t top_k = 50;
    uint64_t seed = 1234;
};

struct VoxtralRealtimeGeneratedTokens {
    std::vector<int32_t> token_ids;
};

struct VoxtralRealtimeRequest {
    runtime::AudioBuffer audio;
    VoxtralRealtimeGenerationOptions generation;
    bool streaming = false;
};

}  // namespace engine::models::voxtral_realtime
