#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <memory>

namespace engine::models::glm_tts {

struct GlmTTSLlamaConfig {
    int64_t bos_token_id = 1;
    int64_t eos_token_id = 2;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t max_position_embeddings = 0;
    int64_t num_attention_heads = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t vocab_size = 0;
    float rms_norm_eps = 1.0e-5F;
    float rope_theta = 10000.0F;
};

struct GlmTTSSpeechTokenizerConfig {
    int64_t feature_size = 128;
    int64_t sampling_rate = 16000;
    int64_t d_model = 0;
    int64_t encoder_attention_heads = 0;
    int64_t encoder_ffn_dim = 0;
    int64_t encoder_layers = 0;
    int64_t max_source_positions = 0;
    int64_t pooling_kernel_size = 0;
    int64_t pooling_position = 0;
    int64_t quantize_position = 0;
    int64_t quantize_vocab_size = 0;
};

struct GlmTTSFlowConfig {
    int64_t speech_token_dim = 512;
    int64_t vocab_size = 100000;
    int64_t mel_dim = 80;
    int64_t trans_dim = 768;
    int64_t depth = 18;
    int64_t heads = 12;
    int64_t dim_head = 64;
    int64_t conv_layers = 4;
    int64_t mel_framerate = 50;
    float input_frame_rate = 25.0F;
    float inference_cfg_rate = 0.7F;
    int64_t inference_steps = 10;
};

struct GlmTTSConfig {
    GlmTTSLlamaConfig llama;
    GlmTTSSpeechTokenizerConfig speech_tokenizer;
    GlmTTSFlowConfig flow;
    int64_t sample_rate = 24000;
    int64_t audio_token_start = 0;
    int64_t audio_token_end = 0;
    int64_t begin_audio_token = 0;
    int64_t end_audio_token = 0;
    int64_t pad_token = 0;
};

struct GlmTTSAssets {
    GlmTTSConfig config;
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> llama_weights;
    std::shared_ptr<const assets::TensorSource> speech_tokenizer_weights;
    std::shared_ptr<const assets::TensorSource> flow_weights;
    std::shared_ptr<const assets::TensorSource> hift_weights;
    std::shared_ptr<const assets::TensorSource> campplus_weights;
};

std::shared_ptr<const GlmTTSAssets> load_glm_tts_assets(
    const std::filesystem::path & model_path);

}  // namespace engine::models::glm_tts
