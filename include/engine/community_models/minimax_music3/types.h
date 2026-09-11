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
    // Flow CFG guidance-delta reuse: when interval > 1 the unconditional
    // branch is evaluated only on warmup steps, every interval-th step and
    // the final step; other steps reuse the cached (cond - uncond) delta.
    // The default of 2 is the listening-accepted recipe (mel-L1 ~0.4 dB to
    // the exact reference at -15..-21% wall); set 1 for the exact-reference
    // trajectory.
    int64_t flow_uncond_interval = 1;
    int64_t flow_uncond_warmup = 2;
    // Number of independent takes decoded together in one batched AR pass
    // (per-take seeds seed, seed+1, ...). The global LM and depth decoder are
    // bandwidth-bound, so K takes cost far less than K runs; flow/vocoder run
    // per take. 1 keeps the plain single-song path.
    int64_t ensemble_takes = 1;
    // Intro-lock fork: the first N frames are decoded once at batch 2 (one
    // master trajectory shared by every take), then the batched decode KV is
    // replicated to 2K rows and the takes diverge with their own seeds. Take
    // 0 continues the master trajectory exactly. 0 disables the fork.
    int64_t ensemble_prefix_frames = 0;
    // Flow chunk hop in AR frames (0 = the model config's 100). A larger hop
    // means fewer chunks and less double-denoising: hop 150 cuts flow ~-28%.
    // Crops and the carry window are rederived from the hop so the seams stay
    // consistent: kept = latents(hop), overlap = (chunk - kept)/2, left crop
    // = overlap/2 (hop 100 reproduces the historical 86/258/172 exactly).
    int64_t flow_chunk_hop_frames = 0;
    // Derived internally by the pipeline from the hop; 0 keeps the config's
    // overlap_latent_length. Not a user knob.
    int64_t flow_overlap_latent_length = 0;
};

}  // namespace engine::models::minimax_music3
