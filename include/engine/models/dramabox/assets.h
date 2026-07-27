#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/models/dramabox/types.h"

#include <filesystem>
#include <memory>
#include <vector>

namespace engine::assets {
class TensorSource;
}

namespace engine::models::dramabox {

struct DramaBoxAssets {
    assets::ResourceBundle resources;
    DramaBoxConfig config;
    std::shared_ptr<const assets::TensorSource> dit_weights;
    std::shared_ptr<const assets::TensorSource> audio_weights;
    std::vector<std::shared_ptr<const assets::TensorSource>> gemma_weights;
    std::shared_ptr<const assets::TensorSource> silence_latent;
};

std::shared_ptr<const DramaBoxAssets> load_dramabox_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::dramabox
