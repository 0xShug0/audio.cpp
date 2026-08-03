#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::stable_audio {

struct StableAudioConfig {
    int64_t sample_size = 0;
    int sample_rate = 0;
    int audio_channels = 0;
    int64_t latent_dim = 0;
    int64_t downsampling_ratio = 0;
    int64_t pretransform_patch_size = 256;
    int64_t pretransform_encoder_chunk_size = 32;
    int64_t pretransform_encoder_stride = 16;
    int64_t same_encoder_in_channels = 0;
    int64_t same_decoder_out_channels = 0;
    int64_t same_channels = 0;
    int64_t same_dim_heads = 64;
    std::vector<int64_t> same_c_mults;
    std::vector<int64_t> same_strides;
    std::vector<int64_t> same_encoder_transformer_depths;
    std::vector<int64_t> same_decoder_transformer_depths;
    std::vector<int64_t> same_decoder_sinusoidal_blocks;
    std::vector<int64_t> same_sliding_window;
    bool same_differential = true;
    bool same_variable_stride = true;
    bool same_encoder_conv_mapping = false;
    bool same_decoder_conv_mapping = false;
    bool same_chunk_midpoint_shift = false;
    float same_encoder_mask_noise = 0.0F;
    float same_decoder_mask_noise = 0.0F;
    bool same_bottleneck_noise_regularize = false;
    bool same_bottleneck_auto_scale = false;
    int64_t same_bottleneck_noise_augment_dim = 0;
    int64_t io_channels = 0;
    int64_t cond_dim = 0;
    int64_t prompt_max_length = 0;
    float seconds_min = 0.0F;
    float seconds_max = 0.0F;
    int64_t diffusion_io_channels = 0;
    int64_t embed_dim = 0;
    int64_t depth = 0;
    int64_t num_heads = 0;
    int64_t cond_token_dim = 0;
    int64_t global_cond_dim = 0;
    int64_t local_add_cond_dim = 0;
    int64_t num_memory_tokens = 0;
    bool differential_attention = false;
    bool force_fp32_norm = false;
    std::string diffusion_objective;
    std::string diffusion_type;
    std::string transformer_type;
    bool stable_audio_open_v1 = false;
    std::string distribution_shift_type = "logsnr";
    float distribution_shift_base_shift = 0.5F;
    float distribution_shift_max_shift = 1.15F;
    int64_t distribution_shift_min_length = 256;
    int64_t distribution_shift_max_length = 4096;
    bool distribution_shift_use_sine = false;
    std::string pretransform_type;
    std::string prompt_conditioner_type;
    int64_t t5_hidden_size = 768;
    int64_t t5_layers = 12;
    int64_t t5_attention_heads = 12;
    int64_t t5_kv_heads = 12;
    int64_t t5_head_dim = 64;
    int64_t t5_intermediate_size = 2048;
    int64_t t5_vocab_size = 256000;
    int64_t t5_pad_token_id = 0;
    int64_t t5_sliding_window = 4096;
    float t5_rope_theta = 10000.0F;
    float t5_rms_norm_eps = 1.0e-6F;
    float t5_attn_logit_softcap = 50.0F;
    float t5_query_pre_attn_scalar = 64.0F;
    bool medium_architecture = false;
};

struct StableAudioAssets {
    assets::ResourceBundle resources;
    std::filesystem::path t5_tokenizer_model_path;
    StableAudioConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> t5_weights;
};

std::shared_ptr<const StableAudioAssets> load_stable_audio_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::stable_audio
