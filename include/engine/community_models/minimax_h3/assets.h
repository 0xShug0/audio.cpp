#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace engine::models::minimax_h3 {

struct MiniMaxH3Config {
    int64_t vocab_size = 151936;
    int64_t prompt_hidden = 384;
    int64_t prompt_layers = 4;
    int64_t prompt_heads = 8;
    int64_t prompt_kv_heads = 2;
    int64_t prompt_head_dim = 48;
    int64_t prompt_intermediate = 1024;
    float prompt_eps = 1.0e-6F;
    float prompt_rope_theta = 10000.0F;

    int64_t dit_layers = 6;
    int64_t token_refiner_layers = 2;
    int64_t hidden = 384;
    int64_t heads = 8;
    int64_t head_dim = 48;
    int64_t ffn = 1024;
    int64_t video_latents_dim = 8;
    int64_t audio_latents_dim = 16;
    int64_t text_dim = 384;
    int64_t timestep_input_dim = 64;
    int64_t time_embed_hidden = 384;
    int64_t time_embed_dim = 192;
    int64_t adaln_curve_grid = 0;
    int64_t rope_inv_freq_len = 4;
    float norm_eps = 1.0e-5F;
    float qk_norm_eps = 1.0e-5F;
    float final_norm_eps = 1.0e-5F;

    int64_t audio_vae_latent_channels = 16;
    int64_t audio_vae_latent_dim = 512;
    int64_t audio_vae_decoder_dim = 256;
    std::vector<int64_t> audio_vae_decoder_rates = {5, 4, 4, 2};
    std::vector<int64_t> audio_vae_decoder_kernels = {9, 8, 8, 4};
    bool audio_vae_bigvgan_weight_norm = true;
    int sample_rate = 32000;

    int64_t video_vae_latent_channels = 24;
    int64_t video_vae_hidden = 2048;
    int64_t video_vae_heads = 32;
    int64_t video_vae_head_dim = 64;
    int64_t video_vae_layers = 36;
    int64_t video_vae_register_tokens = 4;
    int64_t video_vae_patch_size = 16;
    int64_t video_vae_patch_size_t = 4;
    int64_t video_vae_token_drop = 3;
    int64_t video_vae_clip_length = 17;
    int64_t video_vae_tile_size = 256;
    int64_t video_vae_tile_overlap = 64;
    float video_vae_rope_theta = 100.0F;
    float video_vae_rope_dim_ratio = 0.75F;

    int64_t height = 32;
    int64_t width = 32;
    int64_t num_frames = 379;
    int64_t video_latent_t = 112;
    int64_t video_latent_h = 2;
    int64_t video_latent_w = 2;
    int64_t audio_channels = 1;
    int64_t audio_steps = 640;
    int64_t video_patches = 64;
    int64_t denoise_steps = 4;
    float flow_shift = 12.0F;
    float audio_flow_shift = 3.0F;
};

struct MiniMaxH3Assets {
    assets::ResourceBundle resources;
    MiniMaxH3Config config;
    std::shared_ptr<const assets::TensorSource> text_encoder_weights;
    std::shared_ptr<const assets::TensorSource> dit_weights;
    std::shared_ptr<const assets::TensorSource> audio_vae_weights;
    std::shared_ptr<const assets::TensorSource> video_vae_weights;
};

std::shared_ptr<const MiniMaxH3Assets> load_minimax_h3_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::minimax_h3
