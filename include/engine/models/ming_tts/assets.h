#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

namespace engine::models::ming_tts {

struct MingBackboneConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t layers = 0;
    int64_t attention_heads = 0;
    int64_t key_value_heads = 0;
    float rms_norm_eps = 0;
    float rope_theta = 0;
};

struct MingPatchNetworkConfig {
    int64_t hidden_size = 0;
    int64_t heads = 0;
    int64_t depth = 0;
    int64_t intermediate_size = 0;
};

struct MingFlowConfig {
    MingPatchNetworkConfig network;
    MingPatchNetworkConfig aggregator;
    int64_t latent_dim = 0;
    int64_t conditioning_dim = 0;
};

struct MingAudioEncoderConfig {
    MingBackboneConfig backbone;
    int64_t input_dim = 0;
    int64_t hop_size = 0;
    int64_t latent_dim = 0;
    int64_t sliding_window = 0;
};

struct MingAudioDecoderConfig {
    MingBackboneConfig backbone;
    int64_t latent_dim = 0;
    int64_t output_dim = 0;
    int64_t sliding_window = 0;
};

struct MingAudioCodecConfig {
    MingAudioEncoderConfig encoder;
    MingAudioDecoderConfig decoder;
    int64_t patch_size = 0;
    int sample_rate = 0;
};

struct MingConfig {
    MingBackboneConfig backbone;
    int64_t vocabulary_size = 0;
    MingFlowConfig flow;
    MingAudioCodecConfig codec;
};

MingConfig parse_ming_config(const io::json::Value & config);

struct MingAssets {
    assets::ResourceBundle resources;
    MingConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> speaker_weights;
};

std::shared_ptr<const MingAssets> load_ming_tts_assets(const std::filesystem::path & path);

}  // namespace engine::models::ming_tts
