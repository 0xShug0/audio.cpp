#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace engine::models::minimax_music3 {

struct MiniMaxMusic3Config {
    // Global language model (Qwen3-8B shape, derived from the LM GGUF).
    int64_t lm_vocab_size = 200000;
    int64_t lm_hidden = 4096;
    int64_t lm_layers = 36;
    int64_t lm_heads = 32;
    int64_t lm_kv_heads = 8;
    int64_t lm_head_dim = 128;
    int64_t lm_intermediate = 12288;
    int64_t lm_logits = 16385;  // sliced head: [audio_end, 16384 semantic codes]
    float lm_rms_eps = 1.0e-6F;
    float lm_rope_theta = 1000000.0F;
    int64_t lm_max_context = 10240;

    // RVQ depth decoder.
    int64_t depth_hidden = 4096;
    int64_t depth_layers = 4;
    int64_t depth_heads = 16;
    int64_t depth_intermediate = 6144;
    int64_t depth_audio_vocab = 1024;
    int64_t depth_codebooks = 8;
    int64_t depth_max_positions = 16;
    float depth_rms_eps = 1.0e-6F;

    // Condition encoder.
    int64_t cond_hidden = 4096;
    int64_t cond_layers = 8;
    int64_t cond_out_dim = 2048;
    int64_t cond_input_sampling_rate = 24000;
    int64_t cond_input_hop = 960;
    int64_t cond_output_sampling_rate = 44100;
    int64_t cond_output_hop = 512;

    // Flow-matching transformer.
    int64_t dit_in_channels = 128;
    int64_t dit_condition_dim = 2048;
    int64_t dit_layers = 36;
    int64_t dit_heads = 32;
    int64_t dit_head_dim = 64;
    int64_t dit_ff_inner = 8192;
    int64_t dit_rotary_dim = 32;
    int64_t dit_fourier_dim = 256;
    float dit_rope_theta = 10000.0F;

    // Vocoder (DAC-style Flow-VAE decoder).
    int64_t vocoder_latent_channels = 128;
    int64_t vocoder_input_dim = 1024;
    int64_t vocoder_hidden_dim = 1536;
    std::vector<int64_t> vocoder_strides = {8, 8, 4, 2};
    int sample_rate = 44100;
};

struct MiniMaxMusic3Assets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> lm_weights;
    std::shared_ptr<const assets::TensorSource> depth_decoder_weights;
    std::shared_ptr<const assets::TensorSource> dit_weights;
    std::shared_ptr<const assets::TensorSource> condition_encoder_weights;
    std::shared_ptr<const assets::TensorSource> vocoder_weights;
    MiniMaxMusic3Config config;
};

std::shared_ptr<const MiniMaxMusic3Assets> load_minimax_music3_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::minimax_music3
