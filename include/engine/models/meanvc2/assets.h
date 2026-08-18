#pragma once

#include "engine/framework/assets/resource_bundle.h"

#include <filesystem>
#include <memory>

namespace engine::models::meanvc2 {

struct MeanVC2Assets {
    engine::assets::ResourceBundle resources;
    std::shared_ptr<const engine::assets::TensorSource> flow_weights;
    std::shared_ptr<const engine::assets::TensorSource> vocos_weights;
    std::shared_ptr<const engine::assets::TensorSource> asr_weights;
    std::shared_ptr<const engine::assets::TensorSource> speaker_wavlm_weights;
    std::shared_ptr<const engine::assets::TensorSource> speaker_ecapa_weights;
};

std::shared_ptr<const MeanVC2Assets> load_meanvc2_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::meanvc2
