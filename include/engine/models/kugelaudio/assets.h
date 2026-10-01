#pragma once

#include "engine/models/kugelaudio/ar.h"
#include "engine/models/kugelaudio/codec.h"
#include "engine/models/kugelaudio/diffusion.h"
#include "engine/framework/model_spec/package.h"

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::kugelaudio {

struct ModelConfig {
    ArConfig ar;
    DiffusionConfig diffusion;
    CodecConfig codec;
    int inference_steps = 20;
};

struct VoiceFeatures {
    std::vector<float> mean;
    float std = 0.5F;
};

struct ModelAssets {
    assets::ResourceBundle resources;
    ModelConfig config;
    std::shared_ptr<const assets::TensorSource> weights;
    std::unordered_map<std::string, VoiceFeatures> voices;
};

std::shared_ptr<const ModelAssets> load_assets(const std::filesystem::path & path);

}  // namespace engine::models::kugelaudio
