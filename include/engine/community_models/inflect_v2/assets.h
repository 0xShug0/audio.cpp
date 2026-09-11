#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::inflect_v2 {

struct InflectV2Config {
    int64_t vocab_size = 178;
    int64_t sample_rate = 24000;
    int64_t hop_length = 256;
    int64_t inter_channels = 0;
    int64_t hidden_channels = 0;
    int64_t filter_channels = 0;
    int64_t duration_channels = 256;
    int64_t attention_heads = 2;
    int64_t encoder_layers = 3;
    int64_t flow_layers = 4;
    int64_t flow_count = 4;
    int64_t upsample_initial_channels = 0;
    std::vector<int64_t> upsample_rates;
    std::vector<int64_t> upsample_kernel_sizes;
    std::vector<int64_t> resblock_kernel_sizes;
    std::vector<std::vector<int64_t>> resblock_dilations;
    std::string variant;
};

struct InflectV2Assets {
    assets::ResourceBundle resources;
    InflectV2Config config;
    std::shared_ptr<const assets::TensorSource> weights;
};

std::shared_ptr<const InflectV2Assets> load_inflect_v2_assets(
    const std::filesystem::path & model_path);

}  // namespace engine::models::inflect_v2
