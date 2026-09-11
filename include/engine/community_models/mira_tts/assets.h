#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace engine::community_models::mira_tts {

struct MiraTTSConfig {
    int64_t hidden_size = 896;
    int64_t intermediate_size = 4864;
    int64_t layers = 24;
    int64_t attention_heads = 14;
    int64_t kv_heads = 2;
    int64_t head_dim = 64;
    int64_t vocab_size = 166000;
    int64_t max_position_embeddings = 32768;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1.0e6F;
    int32_t bos_token_id = 151643;
    int32_t eos_token_id = 151645;
    int32_t speech_token_start = 155761;
    int32_t speech_token_end = 163952;
    int32_t prompt_speech_start = 165151;
    int32_t sample_rate = 16000;
    int32_t output_sample_rate = 48000;
};

struct MiraTTSAssets {
    assets::ResourceBundle resources;
    MiraTTSConfig config;
    std::shared_ptr<const assets::TensorSource> language_model_weights;
    std::shared_ptr<const assets::TensorSource> speaker_encoder_weights;
    std::shared_ptr<const assets::TensorSource> processor_weights;
    std::shared_ptr<const assets::TensorSource> decoder_weights;
    std::shared_ptr<const assets::TensorSource> upsampler_weights;
};

struct MiraGenerationOptions {
    int64_t max_new_tokens = 1024;
    int64_t top_k = 50;
    float top_p = 0.95F;
    float min_p = 0.05F;
    float temperature = 0.8F;
    float repetition_penalty = 1.2F;
    uint64_t seed = 0;
    bool has_seed = false;
};

std::shared_ptr<const MiraTTSAssets> load_mira_tts_assets(
    const std::filesystem::path & model_path);

}  // namespace engine::community_models::mira_tts
