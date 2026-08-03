#pragma once

#include "engine/framework/assets/resource_bundle.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace engine::assets {
class TensorSource;
}

namespace engine::models::higgs_audio_tts {

struct HiggsTextConfig {
    std::string model_type;
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t max_position_embeddings = 0;
    int64_t bos_token_id = 0;
    int64_t eos_token_id = 0;
    int64_t pad_token_id = -1;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1000000.0F;
    bool tie_word_embeddings = true;
};

struct HiggsAudioEncoderConfig {
    std::string model_type;
    std::string encoder_type;
    int64_t num_codebooks = 0;
    int64_t vocab_size = 0;
    int64_t out_dim = 0;
    int64_t mel_per_sample = 0;
    int64_t max_chunk_size = 0;
    bool tie_word_embeddings = true;
    bool use_delay_pattern = true;
};

struct HiggsConfig {
    std::string model_type;
    std::string architecture;
    int64_t hidden_size = 0;
    int64_t vocab_size = 0;
    int64_t audio_token_id = -100;
    int64_t ignore_index = -100;
    HiggsTextConfig text;
    HiggsAudioEncoderConfig audio;
};

struct HiggsAssets {
    assets::ResourceBundle resources;
    HiggsConfig config;
    std::shared_ptr<const assets::TensorSource> weights;
};

std::shared_ptr<const HiggsAssets> load_higgs_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::higgs_audio_tts
