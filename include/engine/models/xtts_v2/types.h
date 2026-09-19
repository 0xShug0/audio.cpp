#pragma once

#include "engine/framework/runtime/model.h"

#include <cstdint>
#include <string>

namespace engine::models::xtts_v2 {

struct XttsV2Config {
    int64_t sample_rate = 24000;
    int64_t conditioning_sample_rate = 22050;
    int64_t speaker_sample_rate = 16000;
    int64_t text_vocab = 6681;
    int64_t audio_vocab = 1026;
    int64_t model_dim = 1024;
    int64_t gpt_layers = 30;
    int64_t gpt_heads = 16;
    int64_t max_text_tokens = 402;
    int64_t max_audio_tokens = 605;
    int64_t start_text_token = 261;
    int64_t stop_text_token = 0;
    int64_t start_audio_token = 1024;
    int64_t stop_audio_token = 1025;
    int64_t conditioning_tokens = 32;
    int64_t speaker_dim = 512;
    int64_t code_stride = 1024;
    int64_t output_hop = 256;
};

struct XttsV2GenerationOptions {
    float temperature = 0.75F;
    float top_p = 0.85F;
    int64_t top_k = 50;
    float repetition_penalty = 5.0F;
    float speed = 1.0F;
    int64_t max_tokens = 603;
    uint32_t seed = 0;
};

struct XttsV2Request {
    std::string text;
    std::string language = "en";
    runtime::AudioBuffer speaker_audio;
    XttsV2GenerationOptions generation;
};

}  // namespace engine::models::xtts_v2
