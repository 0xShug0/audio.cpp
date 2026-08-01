#pragma once

#include "engine/framework/assets/resource_bundle.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::assets {
class TensorSource;
}

namespace engine::models::higgs_audio_stt {

struct HiggsAudioSTTAudioEncoderConfig {
    int64_t num_mel_bins = 128;
    int64_t encoder_layers = 0;
    int64_t encoder_attention_heads = 0;
    int64_t encoder_ffn_dim = 0;
    int64_t d_model = 0;
    int64_t max_source_positions = 0;
    std::string activation_function = "gelu";
};

struct HiggsAudioSTTTextDecoderConfig {
    int64_t vocab_size = 0;
    int64_t output_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t max_position_embeddings = 0;
    int64_t audio_in_token_id = 0;
    int64_t audio_out_token_id = 0;
    int64_t audio_bos_token_id = 0;
    int64_t audio_eos_token_id = 0;
    int64_t audio_out_bos_token_id = 0;
    int64_t pad_token_id = 0;
    std::vector<int64_t> eos_token_ids;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 5000000.0F;
    std::vector<int64_t> mrope_section = {24, 20, 20};
};

struct HiggsAudioSTTFrontendConfig {
    int sample_rate = 16000;
    int64_t feature_size = 128;
    int64_t hop_length = 160;
    int64_t n_fft = 400;
    double chunk_size_seconds = 4.0;
};

struct HiggsAudioSTTConfig {
    std::string model_type;
    std::string text_decoder_model_type;
    std::string model_size;
    int sample_rate = 16000;
    int64_t max_new_tokens = 1024;
    int64_t projector_temporal_downsample = 1;
    HiggsAudioSTTFrontendConfig frontend;
    HiggsAudioSTTAudioEncoderConfig audio_encoder;
    HiggsAudioSTTTextDecoderConfig text_decoder;
    std::vector<std::string> supported_languages;
};

struct HiggsAudioSTTAssets {
    assets::ResourceBundle resources;
    HiggsAudioSTTConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
};

std::shared_ptr<const HiggsAudioSTTAssets> load_higgs_audio_stt_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::higgs_audio_stt
