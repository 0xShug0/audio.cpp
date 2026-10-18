#pragma once

#include "engine/models/moonshine_asr/config.h"

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::moonshine_asr {


struct MoonshineToken {
    std::string text;
    bool special = false;
};

struct MoonshineAssets {
    assets::ResourceBundle resources;
    MoonshineConfig config;
    std::shared_ptr<const assets::TensorSource> source;
    std::vector<MoonshineToken> tokens;
};

std::shared_ptr<const MoonshineAssets> load_moonshine_asr_assets(const std::filesystem::path & model_path);

std::string decode_moonshine_tokens(
    const MoonshineAssets & assets,
    const std::vector<int32_t> & token_ids);

}  // namespace engine::models::moonshine_asr
