#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/codecs/heartcodec_runtime.h"

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

struct HeartMuLaAssets {
    assets::ResourceBundle resources;
    HeartMuLaConfig mula_config;
    HeartMuLaGenerationConfig generation_config;
    codecs::HeartCodecConfig codec_config;
    std::shared_ptr<const assets::TensorSource> mula_weights;
    std::shared_ptr<const assets::TensorSource> codec_weights;
};

std::shared_ptr<const HeartMuLaAssets> load_heartmula_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::heartmula
