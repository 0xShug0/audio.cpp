#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/models/confucius4_tts/types.h"

#include <filesystem>
#include <memory>

namespace engine::assets {
class TensorSource;
}

namespace engine::models::confucius4_tts {

struct ConfuciusAssets {
    assets::ResourceBundle resources;
    ConfuciusConfig config;
    std::shared_ptr<const assets::TensorSource> t2s_weights;
    std::shared_ptr<const assets::TensorSource> s2a_weights;
    std::shared_ptr<const assets::TensorSource> semantic_encoder_weights;
    std::shared_ptr<const assets::TensorSource> semantic_encoder_shaw_weights;
    std::shared_ptr<const assets::TensorSource> semantic_stats;
    std::shared_ptr<const assets::TensorSource> style_encoder_weights;
    std::shared_ptr<const assets::TensorSource> vocoder_weights;
};

std::shared_ptr<const ConfuciusAssets> load_confucius_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::confucius4_tts
