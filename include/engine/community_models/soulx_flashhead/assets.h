#pragma once
#include "engine/framework/model_spec/package.h"
#include <array>
#include <memory>

namespace engine::community_models::soulx_flashhead {

struct Config {
    std::string variant;
    int64_t dim = 0, ffn_dim = 0, freq_dim = 0, heads = 0, layers = 0;
    int64_t in_channels = 0, latent_channels = 0;
    std::array<int64_t, 3> stride{}, patch{};
    float eps = 0;
    static Config from_json(const io::json::Value & json, const std::string & variant);
};

struct Assets {
    assets::ResourceBundle resources;
    Config config;
    std::shared_ptr<const assets::TensorSource> dit, vae, audio;
};
std::shared_ptr<const Assets> load_assets(const std::filesystem::path & path);
} // namespace engine::community_models::soulx_flashhead
