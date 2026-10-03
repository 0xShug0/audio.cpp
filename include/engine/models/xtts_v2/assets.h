#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/models/xtts_v2/types.h"

#include <filesystem>
#include <memory>

namespace engine::models::xtts_v2 {

struct XttsV2Assets {
    assets::ResourceBundle resources;
    XttsV2Config config;
    std::shared_ptr<const assets::TensorSource> gpt;
    std::shared_ptr<const assets::TensorSource> decoder;
    std::shared_ptr<const assets::TensorSource> speaker_encoder;
    std::filesystem::path tokenizer_path;
};

std::shared_ptr<const XttsV2Assets> load_xtts_v2_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::xtts_v2
