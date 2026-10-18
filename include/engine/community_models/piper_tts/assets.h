#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::piper_tts {

struct PiperTtsConfig {
    int64_t vocab_size = 0;
    int64_t sample_rate = 0;
    int64_t inter_channels = 192;
    int64_t hidden_channels = 192;
    int64_t filter_channels = 768;
    int64_t attention_heads = 2;
    int64_t encoder_layers = 6;
    int64_t flow_layers = 4;
    int64_t flow_count = 4;
    int64_t upsample_initial_channels = 256;
    std::vector<int64_t> upsample_rates{8, 8, 4};
    std::vector<int64_t> upsample_kernel_sizes{16, 16, 8};
    std::vector<int64_t> resblock_kernel_sizes{3, 5, 7};
    std::vector<std::vector<int64_t>> resblock_dilations{{1, 2}, {2, 6}, {3, 12}};
    std::string espeak_voice;
    std::unordered_map<std::string, int32_t> phoneme_id_map;
};

struct PiperTtsAssets {
    assets::ResourceBundle resources;
    PiperTtsConfig config;
    std::shared_ptr<const assets::TensorSource> weights;
};

std::shared_ptr<const PiperTtsAssets> load_piper_tts_assets(
    const std::filesystem::path & model_path);

}  // namespace engine::models::piper_tts
