#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "engine/framework/text/chunking.h"
#include "engine/framework/runtime/session.h"

namespace engine::models::confucius4_tts {

struct ConfuciusT2SConfig {
    int64_t num_layers = 24;
    int64_t model_dim = 1280;
    int64_t num_heads = 20;
    int64_t max_text_seq_lens = 520;
    int64_t max_semantic_seq_lens = 1520;
    int64_t vocab_size = 32000;
    int64_t semantic_vocab_size = 8194;
    int64_t text_embedding_dim = 4096;
    int64_t speaker_embedding_dim = 1024;
    int64_t start_semantic_token = 8192;
    int64_t stop_semantic_token = 8193;
};

struct ConfuciusS2AConfig {
    int64_t input_size = 512;
    int64_t output_size = 80;
    int64_t spk_embed_dim = 192;
    int64_t semantic_embed_dim = 1024;
    int64_t lm_latent_dim = 1280;
    float estimator_mlp_ratio = 3.0F;
    int64_t num_inference_steps = 25;
    float guidance_scale = 0.7F;
};

struct ConfuciusAudioConfig {
    int sample_rate = 22050;
    int prompt_sample_rate = 16000;
    int64_t n_fft = 1024;
    int64_t hop_length = 256;
    int64_t win_length = 1024;
    int64_t n_mels = 80;
    float fmin = 0.0F;
    std::optional<float> fmax = std::nullopt;
};

struct ConfuciusStyleEncoderConfig {
    int64_t feat_dim = 80;
    int64_t embedding_size = 192;
};

struct ConfuciusPreprocessorConfig {
    std::string feature_extractor_type;
    int64_t feature_size = 80;
    int64_t num_mel_bins = 80;
    std::string padding_side;
    float padding_value = 1.0F;
    std::string processor_class;
    bool return_attention_mask = true;
    int sampling_rate = 16000;
    int64_t stride = 2;
};

struct ConfuciusBigVganConfig {
    int sample_rate = 22050;
    int64_t n_fft = 1024;
    int64_t hop_size = 256;
    int64_t win_size = 1024;
    int64_t num_mels = 80;
    int64_t upsample_initial_channel = 1536;
};

struct ConfuciusConfig {
    std::string version = "1";
    ConfuciusT2SConfig t2s;
    ConfuciusS2AConfig s2a;
    ConfuciusAudioConfig audio;
    ConfuciusStyleEncoderConfig style_encoder;
    ConfuciusPreprocessorConfig preprocessor;
    ConfuciusBigVganConfig vocoder;
};

struct ConfuciusGenerationOptions {
    float temperature = 0.8F;
    float top_p = 0.8F;
    int top_k = 30;
    int num_beams = 3;
    float repetition_penalty = 10.0F;
    int64_t max_tokens = 1520;
    int64_t num_inference_steps = 25;
    float guidance_scale = 0.7F;
    int64_t max_text_tokens_per_segment = 80;
    float cross_fade_duration_sec = 0.3F;
    float edge_fade_duration_sec = 0.1F;
    float edge_pad_duration_sec = 0.1F;
    uint32_t seed = 1234;
    engine::text::TextChunkMode text_chunk_mode = engine::text::TextChunkMode::Default;
};

struct ConfuciusVoiceReference {
    std::optional<runtime::AudioBuffer> audio = std::nullopt;
    std::string cache_id;
};

struct ConfuciusRequest {
    std::string text;
    std::string language = "zh";
    ConfuciusVoiceReference reference;
    ConfuciusGenerationOptions generation;
};

struct ConfuciusTextSegment {
    std::string text;
    std::vector<int32_t> token_ids;
};

}  // namespace engine::models::confucius4_tts
