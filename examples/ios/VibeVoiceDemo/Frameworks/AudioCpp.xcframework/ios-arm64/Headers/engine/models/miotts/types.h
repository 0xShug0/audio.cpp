#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::miotts {

struct MioTTSGenerationOptions {
    int64_t max_tokens = 700;
    int top_k = 50;
    float top_p = 1.0F;
    float temperature = 0.8F;
    float repetition_penalty = 1.0F;
    float presence_penalty = 0.0F;
    float frequency_penalty = 0.0F;
    uint32_t seed = 0;
    bool do_sample = true;
};

struct MioTTSPrompt {
    std::string text;
    std::vector<int32_t> input_ids;
};

struct MioTTSGeneratedTokens {
    std::vector<int32_t> token_ids;
    std::vector<int32_t> codec_tokens;
};

}  // namespace engine::models::miotts
