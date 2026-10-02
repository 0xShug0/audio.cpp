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

namespace engine::models::audio_flamingo {

enum class AudioFlamingoVariant {
    V3,
    Next,
};

struct AudioFlamingoFrontendConfig {
    int sample_rate = 16000;
    int64_t feature_size = 128;
    int64_t hop_length = 160;
    int64_t n_fft = 400;
    int64_t chunk_length_sec = 30;
    int64_t max_audio_length_sec = 600;
};

struct AudioFlamingoAudioEncoderConfig {
    int64_t num_mel_bins = 128;
    int64_t num_hidden_layers = 32;
    int64_t num_attention_heads = 20;
    int64_t intermediate_size = 5120;
    int64_t hidden_size = 1280;
    int64_t max_source_positions = 1500;
    std::string activation_function = "gelu";
};

struct AudioFlamingoTextDecoderConfig {
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t max_position_embeddings = 0;
    int64_t audio_token_id = 0;
    int64_t audio_bos_token_id = 0;
    int64_t audio_eos_token_id = 0;
    int64_t pad_token_id = 0;
    std::vector<int64_t> eos_token_ids;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1000000.0F;
};

struct AudioFlamingoRoTEConfig {
    int64_t max_position_embeddings = 1200;
    float rope_theta = 1200.0F;
    float partial_rotary_factor = 0.2F;
    float audio_frame_step = 0.01F;
};

struct AudioFlamingoConfig {
    std::string model_type;
    AudioFlamingoVariant variant = AudioFlamingoVariant::V3;
    int64_t max_new_tokens = 512;
    bool projector_bias = true;
    std::string projector_hidden_act = "gelu";
    AudioFlamingoFrontendConfig frontend;
    AudioFlamingoAudioEncoderConfig audio_encoder;
    AudioFlamingoTextDecoderConfig text_decoder;
    AudioFlamingoRoTEConfig rote;
};

struct AudioFlamingoAssets {
    assets::ResourceBundle resources;
    AudioFlamingoConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
};

std::shared_ptr<const AudioFlamingoAssets> load_audio_flamingo_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::audio_flamingo
