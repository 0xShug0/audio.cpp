#pragma once

#include "engine/framework/text/chunking.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::magpie_tts {

struct MagpieTTSGenerationOptions {
    std::string language = "en";
    int32_t speaker = 0;
    float temperature = 0.6F;
    int32_t top_k = 80;
    float guidance_scale = 2.5F;
    int64_t max_tokens = 500;
    int64_t text_chunk_size = 300;
    engine::text::TextChunkMode text_chunk_mode = engine::text::TextChunkMode::Default;
    uint64_t seed = 0;
};

struct MagpieTTSRequest {
    std::string text;
    MagpieTTSGenerationOptions generation;
};

struct MagpieTokenChunk {
    std::string text;
    std::vector<int32_t> tokens;
};

struct MagpieTokenizationResult {
    std::string language;
    std::string tokenizer_name;
    std::vector<int32_t> tokens;
    std::vector<MagpieTokenChunk> chunks;
};

}  // namespace engine::models::magpie_tts
