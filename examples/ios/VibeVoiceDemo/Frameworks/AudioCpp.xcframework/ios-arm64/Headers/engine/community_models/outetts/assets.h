#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace engine::models::qwen3_asr {
struct Qwen3ASRAssets;
}

namespace engine::models::outetts {

struct OuteTTSLlama3RopeConfig {
    float factor = 32.0F;
    float low_freq_factor = 1.0F;
    float high_freq_factor = 4.0F;
    int64_t original_max_position_embeddings = 8192;
};

struct OuteTTSConfig {
    int64_t bos_token_id = 0;
    int64_t eos_token_id = 0;
    int64_t pad_token_id = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t max_position_embeddings = 0;
    int64_t num_attention_heads = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t vocab_size = 0;
    float rms_norm_eps = 1.0e-5F;
    float rope_theta = 500000.0F;
    OuteTTSLlama3RopeConfig rope_scaling;
    int64_t sample_rate = 24000;
    int64_t hop_length = 320;
    int64_t codebook_size = 1024;
    int64_t codebooks = 2;
    int64_t dac_latent_dim = 1024;
    int64_t dac_decoder_dim = 1536;
};

struct OuteTTSGenerationConfig {
    float temperature = 0.4F;
    float repetition_penalty = 1.1F;
    int64_t repetition_window = 64;
    int64_t top_k = 40;
    float top_p = 0.9F;
    float min_p = 0.05F;
    int64_t max_length = 8192;
};

struct OuteTTSAssets {
    assets::ResourceBundle resources;
    OuteTTSConfig config;
    OuteTTSGenerationConfig generation;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> dac_weights;
    std::shared_ptr<const engine::models::qwen3_asr::Qwen3ASRAssets>
        embedded_aligner;
};

std::shared_ptr<const OuteTTSAssets> load_outetts_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::outetts
