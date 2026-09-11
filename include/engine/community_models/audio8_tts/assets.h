#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/community_models/audio8_tts/types.h"

#include <filesystem>
#include <memory>

namespace engine::assets {
class TensorSource;
}

namespace engine::models::audio8_tts {

struct Audio8TtsAssets {
    assets::ResourceBundle resources;
    Audio8TtsConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> codec_weights;
};

std::shared_ptr<const Audio8TtsAssets> load_audio8_tts_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::audio8_tts
