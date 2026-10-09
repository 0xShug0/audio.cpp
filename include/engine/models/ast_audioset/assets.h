#pragma once

#include "engine/framework/assets/resource_bundle.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::ast_audioset {

struct Config {
    int64_t hidden_size = 0;
    int64_t layers = 0;
    int64_t heads = 0;
    int64_t intermediate_size = 0;
    int64_t patch_size = 0;
    int64_t frequency_stride = 0;
    int64_t time_stride = 0;
    int64_t max_length = 0;
    int64_t num_mel_bins = 0;
    float layer_norm_eps = 0.0f;
    int sample_rate = 0;
    float feature_mean = 0.0f;
    float feature_std = 0.0f;
    std::vector<std::string> labels;
};

struct Assets {
    assets::ResourceBundle resources;
    Config config;
};

std::shared_ptr<Assets> load_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::ast_audioset
