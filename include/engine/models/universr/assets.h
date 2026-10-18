#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <map>
#include <memory>
#include <vector>

namespace engine::models::universr {

struct UniverSRConfig {
    std::vector<int64_t> dims;
    std::vector<int64_t> depths;
    int time_dim = 0;
    int cond_dim = 0;
    int total_freq_bins = 0;
    int hr_freq_bins = 0;
    int feature_enc_layers = 0;
    std::map<int, int> sr_to_lr_bins;
    int sample_rate = 0;
    int n_fft = 0;
    int hop_length = 0;
    float alpha = 0;
    float beta = 0;
    float comp_eps = 0;
};

struct UniverSRAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> tensors;
    UniverSRConfig config;
    std::vector<float> window;
};

std::shared_ptr<const UniverSRAssets> load_universr_assets(const std::filesystem::path & path);

}  // namespace engine::models::universr
