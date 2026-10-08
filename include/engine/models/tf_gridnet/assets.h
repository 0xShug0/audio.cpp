#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>
#include <vector>

namespace engine::models::tf_gridnet {

struct TFGridNetConfig {
    int sample_rate = 0;
    int speakers = 0;
    int microphones = 0;
    int n_fft = 0;
    int hop_length = 0;
    int layers = 0;
    int embedding_dim = 0;
    int embedding_kernel = 0;
    int embedding_stride = 0;
    int lstm_hidden = 0;
    int attention_heads = 0;
    int attention_qk_dim = 0;
    float epsilon = 0;
};

struct TFGridNetAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> tensors;
    TFGridNetConfig config;
    std::vector<float> window;
};

std::shared_ptr<const TFGridNetAssets> load_tf_gridnet_assets(const std::filesystem::path & path);

}  // namespace engine::models::tf_gridnet
