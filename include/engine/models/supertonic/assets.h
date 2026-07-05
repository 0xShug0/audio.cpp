#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/json.h"

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>

namespace engine::models::supertonic {

struct SupertonicConfig {
    int sample_rate = 44100;
    int64_t base_chunk_size = 512;
    int64_t chunk_compress_factor = 6;
    int64_t latent_dim = 24;
};

struct SupertonicAssetPaths {
    std::filesystem::path model_root;
    std::filesystem::path tts_config_path;
    std::filesystem::path unicode_indexer_path;
    std::filesystem::path weights_path;
    std::filesystem::path voice_styles_dir;
};

struct SupertonicAssets {
    SupertonicAssetPaths paths;
    SupertonicConfig config;
    std::shared_ptr<const assets::TensorSource> weights;
    std::unordered_map<uint32_t, int64_t> unicode_indexer;
};

SupertonicAssetPaths resolve_supertonic_assets(const std::filesystem::path & model_path);
std::shared_ptr<const SupertonicAssets> load_supertonic_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::supertonic
