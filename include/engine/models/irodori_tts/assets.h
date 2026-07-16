#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace engine::models::irodori_tts {

struct IrodoriModelConfig {
    int64_t latent_dim = 32;
    int64_t latent_patch_size = 1;
    int64_t model_dim = 1280;
    int64_t num_layers = 12;
    int64_t num_heads = 20;
    float mlp_ratio = 2.875F;
    float text_mlp_ratio = 2.6F;
    float speaker_mlp_ratio = 2.6F;
    float dropout = 0.0F;
    int64_t text_vocab_size = 99574;
    bool text_add_bos = true;
    int64_t text_dim = 512;
    int64_t text_layers = 10;
    int64_t text_heads = 8;
    bool use_caption_condition = false;
    bool use_speaker_condition = true;
    int64_t caption_vocab_size = 0;
    int64_t caption_dim = 0;
    int64_t caption_layers = 0;
    int64_t caption_heads = 0;
    float caption_mlp_ratio = 0.0F;
    int64_t speaker_dim = 768;
    int64_t speaker_layers = 8;
    int64_t speaker_heads = 12;
    int64_t speaker_patch_size = 1;
    int64_t timestep_embed_dim = 512;
    int64_t adaln_rank = 192;
    float norm_eps = 1.0e-5F;
    bool use_duration_predictor = true;
    int64_t duration_aux_dim = 14;
    int64_t duration_hidden_dim = 1024;
    int64_t duration_layers = 3;
    float duration_dropout = 0.1F;
    int64_t duration_attention_heads = 8;
    std::string duration_architecture = "token_sum_adarn_zero_no_aux";
    float duration_token_init_frames = 9.0F;
    std::string duration_speaker_fusion = "adarn_zero";
    std::string duration_caption_fusion = "adarn_zero";
    std::string duration_caption_pooling = "masked_mean";
    int64_t max_text_len = 256;
    int64_t max_caption_len = 256;

    int64_t patched_latent_dim() const noexcept;
    int64_t speaker_patched_latent_dim() const noexcept;
    int64_t caption_vocab_size_resolved() const noexcept;
    int64_t caption_dim_resolved() const noexcept;
    int64_t caption_layers_resolved() const noexcept;
    int64_t caption_heads_resolved() const noexcept;
    float caption_mlp_ratio_resolved() const noexcept;
};

struct IrodoriCodecConfig {
    int64_t encoder_dim = 64;
    int64_t latent_dim = 1024;
    int64_t decoder_dim = 1536;
    int64_t n_codebooks = 16;
    int64_t codebook_size = 1024;
    int64_t codebook_dim = 32;
    int sample_rate = 48000;
    int64_t hop_length = 1920;
};

struct IrodoriTTSAssets {
    assets::ResourceBundle resources;
    IrodoriModelConfig config;
    IrodoriCodecConfig codec;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> codec_weights;
};

std::shared_ptr<const IrodoriTTSAssets> load_irodori_tts_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::irodori_tts
