#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/models/bark_tts/types.h"

#include <filesystem>
#include <memory>
#include <unordered_map>

namespace engine::models::bark_tts {

struct BarkAssets {
    engine::assets::ResourceBundle resources;
    std::shared_ptr<const engine::assets::TensorSource> weights;
    BarkConfig config;
    std::unordered_map<std::string, BarkSpeakerPreset> presets;
};

std::shared_ptr<const BarkAssets> load_bark_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::bark_tts
