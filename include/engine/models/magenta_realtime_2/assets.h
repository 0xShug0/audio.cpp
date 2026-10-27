#pragma once

#include "engine/framework/assets/resource_bundle.h"

#include <filesystem>
#include <memory>
#include <string>

namespace engine::models::magenta_realtime_2 {

struct MagentaConfig {
    std::string variant;
    int64_t temporal_layers = 0;
    int64_t temporal_dim = 0;
    int64_t temporal_heads = 0;
    int64_t temporal_ffn_dim = 0;
    int64_t condition_dim = 0;
    int64_t window = 0;
    int64_t depth_dim = 0;
    int64_t depth_heads = 0;
    int64_t depth_layers = 0;
    int64_t depth_ffn_dim = 0;
};

struct MagentaAssets {
    MagentaConfig config;
    engine::assets::ResourceBundle resources;
    std::shared_ptr<const engine::assets::TensorSource> weights;
};

std::shared_ptr<const MagentaAssets> load_magenta_realtime_2_assets(const std::filesystem::path & path);

}  // namespace engine::models::magenta_realtime_2
