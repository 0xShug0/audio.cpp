#pragma once

#include "engine/framework/assets/resource_bundle.h"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace engine::models::hviske_asr_v6 {

struct HviskeV6WhisperConfig {
    int64_t mel_bins = 128;
    int64_t hidden_size = 1280;
    int64_t intermediate_size = 5120;
    int64_t layers = 32;
    int64_t heads = 20;
    int64_t rotary_dim = 56;
    float rope_theta = 10000.0f;
};

struct HviskeV6Qwen3Config {
    int64_t vocab_size = 151936;
    int64_t hidden_size = 1024;
    int64_t intermediate_size = 3072;
    int64_t layers = 28;
    int64_t heads = 16;
    int64_t kv_heads = 8;
    int64_t head_dim = 128;
    int64_t max_positions = 32768;
    float rms_norm_eps = 1.0e-6f;
    float rope_theta = 1000000.0f;
};

struct HviskeV6Config {
    HviskeV6WhisperConfig encoder;
    HviskeV6Qwen3Config decoder;
    int64_t frame_stack = 4;
    int32_t bos_token_id = 151644;
    int32_t eos_token_id = 151645;
    int32_t pad_token_id = 151643;
    int32_t cased_token_id = 151670;
    int32_t nocase_token_id = 151669;
    int32_t punctuation_token_id = 151672;
    int32_t no_punctuation_token_id = 151671;
};

struct HviskeV6Assets {
    assets::ResourceBundle resources;
    HviskeV6Config config;
    std::shared_ptr<const assets::TensorSource> weights;
};

std::shared_ptr<const HviskeV6Assets> load_hviske_v6_assets(const std::filesystem::path & path);

}  // namespace engine::models::hviske_asr_v6
