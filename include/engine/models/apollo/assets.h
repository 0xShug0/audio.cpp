#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>
#include <vector>

namespace engine::models::apollo {

struct ApolloConfig {
    int sample_rate = 44100;
    int n_fft = 882;
    int hop_length = 441;
    int dim = 256;
    int layers = 6;
    std::vector<int64_t> band_widths;
};

struct ApolloAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> tensors;
    ApolloConfig config;
    std::vector<float> window;
};

std::shared_ptr<const ApolloAssets> load_apollo_assets(const std::filesystem::path & path);

}  // namespace engine::models::apollo
