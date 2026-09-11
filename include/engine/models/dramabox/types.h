#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::dramabox {

enum class DramaBoxPerfMode {
    Exact,
    FlashAttention,
};

struct DramaBoxTransformerConfig {
    int64_t num_layers = 48;
    int64_t hidden_size = 2048;
    int64_t in_channels = 128;
    int64_t out_channels = 128;
    int64_t num_attention_heads = 32;
    int64_t attention_head_dim = 64;
    int64_t cross_attention_dim = 2048;
    int64_t timestep_scale_multiplier = 1000;
    float positional_embedding_theta = 10000.0F;
    std::vector<int64_t> positional_embedding_max_pos;
    int64_t connector_num_layers = 8;
    int64_t connector_num_attention_heads = 32;
    int64_t connector_attention_head_dim = 64;
    std::vector<int64_t> connector_positional_embedding_max_pos;
    int64_t connector_num_learnable_registers = 128;
    bool apply_gated_attention = true;
    bool connector_apply_gated_attention = true;
    bool cross_attention_adaln = true;
    bool use_middle_indices_grid = true;
    bool double_precision_rope = true;
    std::string rope_type = "split";
};

struct DramaBoxAudioVaeConfig {
    int64_t latent_channels = 8;
    int64_t latent_mel_bins = 16;
    int64_t mel_bins = 64;
    int64_t ch = 128;
    std::vector<int64_t> ch_mult;
    int64_t num_res_blocks = 2;
    std::vector<int64_t> attn_resolutions;
    int64_t resolution = 256;
    int64_t out_channels = 2;
    int64_t sample_rate = 16000;
    int64_t hop_length = 160;
    int64_t n_fft = 1024;
    bool causal = true;
    std::string norm_type = "pixel";
    std::string causality_axis = "height";
};

struct DramaBoxVocoderConfig {
    int64_t input_sample_rate = 16000;
    int64_t output_sample_rate = 48000;
    int64_t hop_length = 80;
    int64_t n_fft = 512;
    int64_t num_mels = 64;
    int64_t vocoder_initial_channel = 1536;
    int64_t bwe_initial_channel = 512;
    std::vector<int64_t> vocoder_upsample_rates;
    std::vector<int64_t> vocoder_upsample_kernel_sizes;
    std::vector<int64_t> bwe_upsample_rates;
    std::vector<int64_t> bwe_upsample_kernel_sizes;
    std::vector<int64_t> resblock_kernel_sizes;
};

struct DramaBoxGemma3Config {
    int64_t hidden_size = 3840;
    int64_t intermediate_size = 15360;
    int64_t num_hidden_layers = 48;
    int64_t num_attention_heads = 16;
    int64_t num_key_value_heads = 8;
    int64_t head_dim = 256;
    int64_t vocab_size = 262208;
    int64_t max_position_embeddings = 131072;
    int64_t prompt_max_length = 1024;
    int64_t sliding_window = 1024;
    int64_t sliding_window_pattern = 6;
    float rope_theta = 1000000.0F;
    float rope_local_base_freq = 10000.0F;
    float rope_scaling_factor = 8.0F;
    float rms_norm_eps = 1.0e-6F;
    float query_pre_attn_scalar = 256.0F;
};

struct DramaBoxConfig {
    std::string model_type = "dramabox-tts";
    DramaBoxTransformerConfig transformer;
    DramaBoxAudioVaeConfig audio_vae;
    DramaBoxVocoderConfig vocoder;
    DramaBoxGemma3Config gemma;
    int64_t diffusion_steps = 30;
    float default_cfg_scale = 2.5F;
    float default_spatio_temporal_guidance_scale = 1.5F;
    float default_duration_scale = 1.1F;
    float default_reference_duration_sec = 10.0F;
    float fps = 25.0F;
};

struct DramaBoxRequest {
    std::string prompt;
    std::string negative_prompt;
    std::filesystem::path target_voice;
    bool has_inline_target_voice = false;
    int64_t seed = 42;
    int64_t steps = 30;
    float cfg_scale = 2.5F;
    float spatio_temporal_guidance_scale = 1.5F;
    float duration_scale = 1.1F;
    float duration_sec = 0.0F;
    float reference_duration_sec = 10.0F;
    float guidance_rescale = -1.0F;
    float audio_chunk_threshold_sec = 45.0F;
    float audio_chunk_duration_sec = 37.0F;
    float cross_fade_duration_sec = 0.05F;
};

}  // namespace engine::models::dramabox
