#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::audio_flamingo {

struct AudioFlamingoGenerationOptions {
    int64_t max_new_tokens = 512;
    float repetition_penalty = 1.0F;
    bool do_sample = false;
    int64_t seed = 42;
    float temperature = 1.0F;
    int64_t top_k = 50;
    float top_p = 1.0F;
};

struct AudioFlamingoAudioFeatures {
    std::vector<float> values;
    std::vector<int32_t> attention_mask;
    std::vector<int64_t> post_lengths;
    int64_t batch = 0;
    int64_t mel_bins = 0;
    int64_t frames = 0;
    int64_t windows = 0;
};

struct AudioFlamingoPrompt {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> attention_mask;
    std::vector<int32_t> audio_token_positions;
};

struct AudioFlamingoAudioEncoderOutput {
    std::vector<float> values;
    int64_t batch = 0;
    int64_t tokens = 0;
    int64_t hidden_size = 0;
};

struct AudioFlamingoAudioProjectorOutput {
    std::vector<float> values;
    int64_t tokens = 0;
    int64_t hidden_size = 0;
};

struct AudioFlamingoGeneratedTokens {
    std::vector<int32_t> token_ids;
};

inline int64_t audio_flamingo_floor_div(int64_t numerator, int64_t denominator) {
    int64_t quotient = numerator / denominator;
    const int64_t remainder = numerator % denominator;
    if (remainder != 0 && ((remainder < 0) != (denominator < 0))) {
        --quotient;
    }
    return quotient;
}

inline int64_t audio_flamingo_conv2_length(int64_t input_frames) {
    return (input_frames - 1) / 2 + 1;
}

inline int64_t audio_flamingo_post_length(int64_t input_frames) {
    return audio_flamingo_floor_div(audio_flamingo_conv2_length(input_frames) - 2, 2) + 1;
}

}  // namespace engine::models::audio_flamingo
