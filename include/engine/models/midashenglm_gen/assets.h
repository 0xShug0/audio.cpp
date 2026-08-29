#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>
#include <string>

namespace engine::models::midashenglm_gen {

struct MiDashengLmGenConfig {
    int64_t vocab_size = 151697;
    int64_t hidden_size = 2048;
    int64_t audio_embedding_size = 768;
    int64_t target_embedding_size = 768;
    int64_t sequence_length = 500;
    int64_t patch_size = 5;
    int64_t block_size = 3200;
    int64_t flow_hidden_size = 2048;
    int64_t flow_depth = 16;
    int64_t flow_heads = 8;
    int64_t flow_intermediate_size = 8192;
    int64_t decoder_layers = 8;
    int64_t decoder_intermediate_size = 1536;
    int64_t istft_n_fft = 1280;
    int64_t istft_hop = 320;
    int64_t sample_rate = 16000;
};

struct MiDashengLmGenAssets {
    std::filesystem::path model_root;
    MiDashengLmGenConfig config;
    engine::assets::ResourceBundle resources;
    std::shared_ptr<const engine::assets::TensorSource> weights;
};

std::shared_ptr<const MiDashengLmGenAssets> load_midashenglm_gen_assets(
    const std::filesystem::path & model_path);

}  // namespace engine::models::midashenglm_gen
