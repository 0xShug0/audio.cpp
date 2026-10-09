#pragma once

#include "engine/framework/assets/resource_bundle.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::ced {

struct CedConfig {
    int64_t hidden = 0;
    int64_t heads = 0;
    int64_t layers = 0;
    int64_t intermediate = 0;
    int64_t patch_size = 0;
    int64_t patch_stride = 0;
    int64_t mel_bins = 0;
    int64_t target_frames = 0;
    int64_t fft_size = 0;
    int64_t hop_size = 0;
    int64_t window_size = 0;
    int sample_rate = 0;
    bool center = true;
    bool pad_last = true;
    std::vector<std::string> labels;
};

struct CedAssets {
    assets::ResourceBundle resources;
    CedConfig config;
};

std::shared_ptr<CedAssets> load_assets(const std::filesystem::path & path);

}  // namespace engine::models::ced
