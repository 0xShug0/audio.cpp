#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>
#include <vector>

namespace engine::models::controlfoley {

struct ControlFoleyConfig {
    int64_t sample_rate = 44100;
    int64_t spec_frame_frequency = 512;
    int64_t latent_reduction_factor = 2;
    int64_t latent_dim = 40;
    int64_t clip_dim = 1024;
    int64_t visual_dim = 768;
    int64_t sync_dim = 768;
    int64_t text_dim = 1024;
    int64_t audio_dim = 512;
    int64_t timbre_dim = 1536;
    int64_t text_seq_len = 77;
    int64_t audio_seq_len = 1;
    int64_t timbre_seq_len = 1;
};

struct ControlFoleyAssets {
    std::filesystem::path model_root;
    std::filesystem::path gguf_path;
    ControlFoleyConfig config;
    engine::assets::ResourceBundle resources;
    std::shared_ptr<const engine::assets::TensorSource> weights;
    std::shared_ptr<const engine::assets::TensorSource> flow_weights;
    std::shared_ptr<const engine::assets::TensorSource> vae_weights;
    std::shared_ptr<const engine::assets::TensorSource> bigvgan_weights;
    std::shared_ptr<const engine::assets::TensorSource> open_clip_weights;
    std::shared_ptr<const engine::assets::TensorSource> cav_mae_weights;
    std::shared_ptr<const engine::assets::TensorSource> synchformer_weights;
    std::shared_ptr<const engine::assets::TensorSource> clap_weights;
    std::shared_ptr<const engine::assets::TensorSource> mert_weights;
    std::shared_ptr<const engine::assets::TensorSource> musicgen_style_weights;
};

std::shared_ptr<const ControlFoleyAssets> load_controlfoley_assets(
    const std::filesystem::path & model_path);

}  // namespace engine::models::controlfoley
