#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/models/dots_tts/types.h"

#include <filesystem>
#include <memory>

namespace engine::assets {
class TensorSource;
}

namespace engine::models::dots_tts {

struct DotsAssets {
    assets::ResourceBundle resources;
    DotsConfig config;
    std::shared_ptr<const assets::TensorSource> core_weights;
    std::shared_ptr<const assets::TensorSource> vocoder_weights;
    std::shared_ptr<const assets::TensorSource> speaker_encoder_weights;
    std::shared_ptr<const assets::TensorSource> latent_stats;
};

std::shared_ptr<const DotsAssets> load_dots_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::dots_tts
