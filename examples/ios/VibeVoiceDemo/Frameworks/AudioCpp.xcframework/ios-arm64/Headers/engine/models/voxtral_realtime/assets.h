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

namespace engine::models::voxtral_realtime {

struct VoxtralRealtimeFrontendConfig {
    int64_t sample_rate = 16000;
    int64_t feature_size = 128;
    int64_t n_fft = 400;
    int64_t win_length = 400;
    int64_t hop_length = 160;
    float global_log_mel_max = 1.5F;
    float padding_value = 0.0F;
    bool return_attention_mask = true;
    std::string padding_side = "right";
};

struct VoxtralRealtimeAudioConfig {
    std::string model_type;
    std::string activation_function = "gelu";
    std::string hidden_act = "silu";
    int64_t hidden_size = 1280;
    int64_t intermediate_size = 5120;
    int64_t num_hidden_layers = 32;
    int64_t num_attention_heads = 32;
    int64_t num_key_value_heads = 32;
    int64_t head_dim = 64;
    int64_t num_mel_bins = 128;
    int64_t max_position_embeddings = 1500;
    int64_t sliding_window = 750;
    int64_t vocab_size = 131072;
    float rms_norm_eps = 1.0e-5F;
    float rope_theta = 1000000.0F;
};

struct VoxtralRealtimeTextConfig {
    std::string model_type;
    std::string hidden_act = "silu";
    int64_t vocab_size = 131072;
    int64_t hidden_size = 3072;
    int64_t intermediate_size = 9216;
    int64_t num_hidden_layers = 26;
    int64_t num_attention_heads = 32;
    int64_t num_key_value_heads = 8;
    int64_t head_dim = 128;
    int64_t max_position_embeddings = 131072;
    int64_t sliding_window = 8192;
    int64_t bos_token_id = 1;
    int64_t eos_token_id = 2;
    int64_t pad_token_id = 11;
    float rms_norm_eps = 1.0e-5F;
    float rope_theta = 1000000.0F;
    bool tie_word_embeddings = true;
    bool use_cache = true;
};

struct VoxtralRealtimeConfig {
    std::string model_type;
    std::string dtype;
    std::string projector_hidden_act = "gelu";
    int64_t hidden_size = 3072;
    int64_t audio_length_per_tok = 8;
    int64_t default_num_delay_tokens = 6;
    int64_t downsample_factor = 4;
    VoxtralRealtimeFrontendConfig frontend;
    VoxtralRealtimeAudioConfig audio;
    VoxtralRealtimeTextConfig text;
    std::vector<std::string> supported_languages;
};

struct VoxtralRealtimeAssets {
    assets::ResourceBundle resources;
    VoxtralRealtimeConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
};

std::shared_ptr<const VoxtralRealtimeAssets> load_voxtral_realtime_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::voxtral_realtime
