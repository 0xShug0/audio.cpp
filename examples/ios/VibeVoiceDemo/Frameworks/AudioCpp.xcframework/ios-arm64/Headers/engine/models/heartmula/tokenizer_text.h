#pragma once

#include "engine/models/heartmula/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::heartmula {

struct HeartMuLaGenerationOptions {
    float duration_seconds = 120.0F;
    float temperature = 1.0F;
    int64_t top_k = 50;
    float guidance_scale = 1.5F;
    float codec_duration = 29.76F;
    int64_t num_inference_steps = 10;
    float codec_guidance_scale = 1.25F;
    bool infinite_mode = false;
    int64_t text_chunk_size = 4096;
    int64_t infinite_chunk_audio_length_ms = 240000;
};

struct HeartMuLaPromptRequest {
    std::string tags;
    std::string lyrics;
    HeartMuLaGenerationOptions options;
};

struct HeartMuLaPromptEncoding {
    int64_t batch_size = 0;
    int64_t prompt_len = 0;
    int64_t parallel_number = 0;
    std::vector<int64_t> tokens;
    std::vector<uint8_t> tokens_mask;
    std::vector<float> muq_embed;
    std::vector<int64_t> muq_idx;
    std::vector<int64_t> pos;
    std::vector<int32_t> tags_ids;
    std::vector<int32_t> lyrics_ids;
};

class HeartMuLaTextTokenizer {
public:
    struct Impl;

    explicit HeartMuLaTextTokenizer(std::shared_ptr<const HeartMuLaAssets> assets);

    std::vector<int32_t> encode(const std::string & text) const;
    HeartMuLaPromptEncoding encode_prompt(const HeartMuLaPromptRequest & request) const;

private:
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::heartmula
