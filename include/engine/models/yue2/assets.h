#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/models/yue2/types.h"

#include <filesystem>
#include <memory>

namespace engine::models::yue2 {

struct Yue2Assets {
    std::filesystem::path model_root;
    std::filesystem::path tiktoken_path;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> vae_weights;
    Yue2Config config;
};

std::shared_ptr<const Yue2Assets> load_yue2_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::yue2
