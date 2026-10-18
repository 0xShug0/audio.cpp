#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace engine::models::muscriptor {

struct MuScriptorConfig {
    std::string model_type;
    std::string variant;
    int64_t dim = 0;
    int64_t num_heads = 0;
    int64_t num_layers = 0;
    int64_t card = 0;
    int sample_rate = 16000;
    int segment_seconds = 5;
    int64_t n_fft = 2048;
    int64_t hop_length = 160;
    int64_t n_mels = 512;
};

struct MuScriptorAssets {
    assets::ResourceBundle resources;
    MuScriptorConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
};

std::shared_ptr<const MuScriptorAssets> load_muscriptor_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::muscriptor
