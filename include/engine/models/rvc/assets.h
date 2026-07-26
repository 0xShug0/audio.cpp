#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>

namespace engine::models::rvc {

struct RvcVoiceModel {
    std::string id;
    std::string version;
    int sample_rate = 0;
    bool has_f0 = true;
    int speaker_count = 0;
    std::shared_ptr<const engine::assets::TensorSource> checkpoint;
    std::shared_ptr<const engine::assets::TensorSource> index_vectors;
    std::filesystem::path checkpoint_path;
    std::filesystem::path index_path;
};

struct RvcAssets {
    engine::assets::ResourceBundle resources;
    std::shared_ptr<const engine::assets::TensorSource> hubert;
    std::shared_ptr<const engine::assets::TensorSource> rmvpe;
    std::unordered_map<std::string, RvcVoiceModel> voices;
};

std::shared_ptr<const RvcAssets> load_rvc_assets(const std::filesystem::path & model_path);
RvcVoiceModel load_rvc_voice_model(const std::filesystem::path & checkpoint_path);

}  // namespace engine::models::rvc
