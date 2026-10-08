#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>

namespace engine::models::mossformer2 {

struct MossFormer2Config {
    int sample_rate = 16000;
    int speakers = 2;
    int kernel = 16;
    int channels = 512;
    int hidden = 512;
    int layers = 24;
};

struct MossFormer2Assets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> tensors;
    MossFormer2Config config;
};

std::shared_ptr<const MossFormer2Assets> load_mossformer2_assets(const std::filesystem::path & path);

}  // namespace engine::models::mossformer2
