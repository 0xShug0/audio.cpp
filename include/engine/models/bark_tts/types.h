#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::bark_tts {

struct BarkTransformerConfig {
    int64_t hidden = 768;
    int64_t layers = 12;
    int64_t heads = 12;
    int64_t block_size = 1024;
    int64_t input_vocab = 0;
    int64_t output_vocab = 0;
    bool bias = false;
};

struct BarkConfig {
    BarkTransformerConfig semantic;
    BarkTransformerConfig coarse;
    BarkTransformerConfig fine;
    int64_t codebook_size = 1024;
    int64_t codebook_dim = 128;
    int64_t sample_rate = 24000;
};

struct BarkSpeakerPreset {
    std::vector<int32_t> semantic;
    std::vector<std::vector<int32_t>> coarse;
    std::vector<std::vector<int32_t>> fine;
};

struct BarkGenerationOptions {
    std::string voice_id = "v2/en_speaker_6";
    float temperature = 0.7F;
    int top_k = 50;
    int64_t max_tokens = 768;
    uint64_t seed = 0;
};

}  // namespace engine::models::bark_tts
