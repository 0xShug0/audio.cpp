#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/tokenizers/sentencepiece.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>

namespace engine::community_models::liveavatar {

struct LiveAvatarConfig {
    int64_t text_len = 512;
    int64_t text_dim = 4096;
    int64_t latent_channels = 16;
    int64_t hidden_size = 5120;
    int64_t ffn_dim = 13824;
    int64_t num_heads = 40;
    int64_t num_layers = 40;
    int64_t audio_dim = 1024;
    int64_t audio_layers = 25;
    int64_t audio_tokens = 4;
    int64_t motion_frames = 73;
    int64_t latent_motion_frames = 19;
    int64_t fps = 16;
    int64_t sample_steps = 4;
    float sample_shift = 3.0F;
    float guidance_scale = 0.0F;
    int64_t height = 480;
    int64_t width = 832;
    int64_t default_frames = 48;
};

struct LiveAvatarAssets {
    std::filesystem::path model_root;
    std::filesystem::path tokenizer_model;
    LiveAvatarConfig config;
    std::vector<engine::tokenizers::SentencePiecePiece> tokenizer_pieces;
    std::shared_ptr<const engine::assets::TensorSource> support_weights;
    std::shared_ptr<const engine::assets::TensorSource> text_encoder_weights;
    std::shared_ptr<const engine::assets::TensorSource> denoiser_weights;
    std::shared_ptr<const engine::assets::TensorSource> audio_encoder_weights;
    std::shared_ptr<const engine::assets::TensorSource> vae_weights;
};

std::shared_ptr<const LiveAvatarAssets>
load_liveavatar_assets(
    const std::filesystem::path & model_path,
    const std::unordered_map<std::string, std::string> & options = {});

}  // namespace engine::community_models::liveavatar
