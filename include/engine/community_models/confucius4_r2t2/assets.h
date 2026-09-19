#pragma once

#include "engine/framework/assets/resource_bundle.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace engine::assets {
class TensorSource;
}

namespace engine::community_models::confucius4_r2t2 {

struct R2T2ASRAudioEncoderConfig {
    int64_t num_mel_bins = 128;
    int64_t encoder_layers = 0;
    int64_t encoder_attention_heads = 0;
    int64_t encoder_ffn_dim = 0;
    int64_t d_model = 0;
    int64_t max_source_positions = 0;
    int64_t n_window = 100;
    int64_t n_window_infer = 400;
    int64_t conv_chunksize = 500;
    int64_t downsample_hidden_size = 0;
    int64_t output_dim = 0;
    std::string activation_function = "gelu";
};

struct R2T2ASRTextDecoderConfig {
    int64_t vocab_size = 0;
    int64_t output_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t max_position_embeddings = 0;
    int64_t audio_token_id = 0;
    int64_t audio_start_token_id = 0;
    int64_t audio_end_token_id = 0;
    int64_t pad_token_id = 0;
    std::vector<int64_t> eos_token_ids;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 5000000.0F;
    bool attention_bias = false;
    std::vector<int64_t> mrope_section = {24, 20, 20};
};

struct R2T2ASRFrontendConfig {
    int sample_rate = 16000;
    int64_t feature_size = 128;
    int64_t hop_length = 160;
    int64_t n_fft = 400;
};

struct R2T2ASRConfig {
    std::string model_type;
    std::string thinker_model_type;
    std::string model_size;
    int sample_rate = 16000;
    int64_t max_new_tokens = 512;
    int64_t classify_num = 0;
    int64_t timestamp_token_id = 0;
    int64_t timestamp_segment_time_ms = 0;
    bool hf_transformers_layout = false;
    bool tie_word_embeddings = false;
    R2T2ASRFrontendConfig frontend;
    R2T2ASRAudioEncoderConfig audio_encoder;
    R2T2ASRTextDecoderConfig text_decoder;
    std::vector<std::string> supported_languages;
};

struct R2T2ASRAssets {
    assets::ResourceBundle resources;
    R2T2ASRConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
};

std::shared_ptr<const R2T2ASRAssets> load_confucius4_r2t2_assets(const std::filesystem::path & model_path);
std::shared_ptr<const R2T2ASRAssets> load_confucius4_r2t2_assets(
    const std::filesystem::path & model_path,
    std::string_view package_family);
std::shared_ptr<const R2T2ASRAssets> load_confucius4_r2t2_assets(
    assets::ResourceBundle resources);

}  // namespace engine::community_models::confucius4_r2t2
