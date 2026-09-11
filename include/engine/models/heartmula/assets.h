#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::heartmula {

struct HeartMuLaTransformerConfig {
    std::string flavor;
    int64_t num_layers = 0;
    int64_t num_heads = 0;
    int64_t num_kv_heads = 0;
    int64_t embed_dim = 0;
    int64_t max_seq_len = 0;
    int64_t intermediate_dim = 0;
    int64_t head_dim = 0;
    float norm_eps = 1.0e-5F;
    float rope_base = 500000.0F;
    float rope_scale_factor = 32.0F;
};

struct HeartMuLaConfig {
    std::string model_type;
    std::string torch_dtype;
    std::string backbone_flavor;
    std::string decoder_flavor;
    int64_t text_vocab_size = 0;
    int64_t audio_vocab_size = 0;
    int64_t audio_num_codebooks = 0;
    int64_t muq_dim = 0;
    HeartMuLaTransformerConfig backbone;
    HeartMuLaTransformerConfig decoder;
};

struct HeartMuLaGenerationConfig {
    int64_t text_bos_id = 0;
    int64_t text_eos_id = 0;
    int64_t audio_eos_id = 0;
    int64_t empty_id = 0;
};

struct HeartCodecConfig {
    std::string model_type;
    std::string torch_dtype;
    int64_t dim = 0;
    int64_t codebook_size = 0;
    float decay = 0.0F;
    float commitment_weight = 0.0F;
    int64_t threshold_ema_dead_code = 0;
    bool use_cosine_sim = false;
    int64_t codebook_dim = 0;
    int64_t num_quantizers = 0;
    int64_t attention_head_dim = 0;
    int64_t in_channels = 0;
    std::string norm_type;
    int64_t num_attention_heads = 0;
    int64_t num_layers = 0;
    int64_t num_layers_2 = 0;
    int64_t out_channels = 0;
    int64_t num_bands = 0;
    int sample_rate = 0;
    bool causal = true;
    int64_t num_samples = 0;
    std::vector<int64_t> downsample_factors;
    std::vector<int64_t> downsample_kernel_sizes;
    std::vector<int64_t> upsample_factors;
    std::vector<int64_t> upsample_kernel_sizes;
    int64_t latent_hidden_dim = 0;
    int64_t default_kernel_size = 0;
    int64_t delay_kernel_size = 0;
    int64_t init_channel = 0;
    int64_t res_kernel_size = 0;
};

struct HeartMuLaAssets {
    assets::ResourceBundle resources;
    HeartMuLaConfig mula_config;
    HeartMuLaGenerationConfig generation_config;
    HeartCodecConfig codec_config;
    std::shared_ptr<const assets::TensorSource> mula_weights;
    std::shared_ptr<const assets::TensorSource> codec_weights;
};

std::shared_ptr<const HeartMuLaAssets> load_heartmula_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::heartmula
