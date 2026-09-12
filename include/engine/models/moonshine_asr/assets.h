#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::moonshine_asr {

struct MoonshineSlidingWindow {
    int64_t past = 16;
    int64_t future = 4;
};

struct MoonshineEncoderConfig {
    std::string model_type = "moonshine_streaming_encoder";
    std::string hidden_act = "gelu";
    int64_t sample_rate = 16000;
    float frame_ms = 5.0F;
    int64_t frame_length = 80;
    int64_t hidden_size = 320;
    int64_t intermediate_size = 1280;
    int64_t layers = 6;
    int64_t heads = 8;
    int64_t kv_heads = 8;
    int64_t head_dim = 40;
    int64_t max_position_embeddings = 4096;
    bool attention_bias = false;
    std::vector<MoonshineSlidingWindow> sliding_windows;
};

struct MoonshineDecoderConfig {
    std::string hidden_act = "silu";
    int64_t hidden_size = 320;
    int64_t intermediate_size = 1280;
    int64_t layers = 6;
    int64_t heads = 8;
    int64_t kv_heads = 8;
    int64_t head_dim = 40;
    int64_t vocab_size = 32768;
    int64_t max_position_embeddings = 4096;
    int64_t bos_token_id = 1;
    int64_t eos_token_id = 2;
    int64_t pad_token_id = 0;
    int64_t decoder_start_token_id = 1;
    float rope_theta = 10000.0F;
    float partial_rotary_factor = 0.8F;
    int64_t rotary_dim = 32;
    bool attention_bias = false;
    bool tie_word_embeddings = false;
};

struct MoonshineConfig {
    std::string model_type = "moonshine_streaming";
    std::string variant;
    MoonshineEncoderConfig encoder;
    MoonshineDecoderConfig decoder;
    float max_tokens_per_second = 6.5F;
};

struct MoonshineToken {
    std::string text;
    bool special = false;
};

struct MoonshineAssets {
    assets::ResourceBundle resources;
    MoonshineConfig config;
    std::shared_ptr<const assets::TensorSource> source;
    std::vector<MoonshineToken> tokens;
};

std::shared_ptr<const MoonshineAssets> load_moonshine_asr_assets(const std::filesystem::path & model_path);

std::string decode_moonshine_tokens(
    const MoonshineAssets & assets,
    const std::vector<int32_t> & token_ids);

}  // namespace engine::models::moonshine_asr
