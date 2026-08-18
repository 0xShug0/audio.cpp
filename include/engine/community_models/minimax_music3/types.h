#pragma once

#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::minimax_music3 {

struct MiniMaxMusic3QwenConfig {
    int64_t vocab_size = 200000;
    int64_t hidden_size = 4096;
    int64_t intermediate_size = 12288;
    int64_t layers = 36;
    int64_t attention_heads = 32;
    int64_t kv_heads = 8;
    int64_t head_dim = 128;
    int64_t max_position_embeddings = 10240;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1000000.0F;
};

struct MiniMaxMusic3DepthConfig {
    int64_t audio_vocab_size = 1024;
    int64_t hidden_size = 4096;
    int64_t intermediate_size = 6144;
    int64_t max_position_embeddings = 16;
    int64_t attention_heads = 16;
    int64_t codebooks = 8;
    int64_t layers = 4;
    float rms_norm_eps = 1.0e-6F;
};

struct MiniMaxMusic3ConditionConfig {
    int64_t condition_hidden_dim = 4096;
    int64_t condition_layers = 8;
    int64_t out_dim = 2048;
    int64_t input_sample_rate = 24000;
    int64_t input_hop_length = 960;
    int64_t output_sample_rate = 44100;
    int64_t output_hop_length = 512;
};

struct MiniMaxMusic3FlowConfig {
    int64_t in_channels = 128;
    int64_t condition_dim = 2048;
    int64_t layers = 36;
    int64_t attention_heads = 32;
    int64_t head_dim = 64;
    int64_t ff_inner_dim = 8192;
    int64_t rotary_dim = 32;
    int64_t fourier_embedding_dim = 256;
};

struct MiniMaxMusic3VocoderConfig {
    int64_t latent_channels = 128;
    int64_t decoder_input_dim = 1024;
    int64_t decoder_hidden_dim = 1536;
    int sample_rate = 44100;
    int64_t hop_length = 512;
    std::vector<int64_t> upsample_ratios = {8, 8, 4, 2};
};

struct MiniMaxMusic3Config {
    MiniMaxMusic3QwenConfig qwen;
    MiniMaxMusic3DepthConfig depth;
    MiniMaxMusic3ConditionConfig condition;
    MiniMaxMusic3FlowConfig flow;
    MiniMaxMusic3VocoderConfig vocoder;
    int64_t max_prompt_tokens = 5000;
    int64_t max_audio_frames = 9000;
    int64_t frame_rate = 25;
    int64_t chunk_frames = 200;
    int64_t chunk_hop_frames = 100;
    int64_t overlap_latent_length = 172;
};

struct MiniMaxMusic3Request {
    std::string prompt;
    std::string lyrics;
    double duration_sec = 20.0;
    int64_t num_inference_steps = 30;
    float guidance_scale = 1.7F;
    float ar_guidance_scale = 1.5F;
    int64_t top_k = 50;
    uint64_t seed = 0;
};

}  // namespace engine::models::minimax_music3
