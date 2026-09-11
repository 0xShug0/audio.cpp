#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::f5_tts {

// F5TTS_v1_Base / Habibi architecture (validated against the checkpoint:
// 22 blocks, dim 1024, 16 heads x 64, ff_mult 2, text_dim 512, conv_layers 4,
// 100 mel channels, embedding rows 2731).
struct F5Architecture {
    int dim = 1024;
    int depth = 22;
    int heads = 16;
    int head_dim = 64;
    int ff_mult = 2;
    int text_dim = 512;
    int conv_layers = 4;
    int mel_dim = 100;
    int vocab_rows = 2731;
    int sample_rate = 24000;
    int hop_length = 256;
    int n_fft = 1024;
};

struct F5SampleOptions {
    float cfg_strength = 2.0F;
    float sway_sampling_coef = -1.0F;
    float speed = 1.0F;
    int steps = 32;
    uint32_t seed = 0;
};

// Compute device for the DiT forward: CUDA device index or CPU threads.
struct F5ComputeDevice {
    bool use_cuda = false;
    int device = 0;      // CUDA device index
    int threads = 0;     // CPU threads; 0 = hardware concurrency
    // FP16 linear weights: GEMMs get ~3x faster on tensor cores but each
    // mul_mat converts the F32 activations to F16 first; at F5's GEMM sizes
    // (K=1024/2048, N~1022) the conversion overhead outweighs the gain on an
    // RTX 3090 (measured 4.0s -> 5.0s per clip). Off by default.
    bool fp16_weights = false;
};

// Debug taps for parity testing: when non-null, intermediate stage outputs are
// appended (column layout, [features, T] flattened feature-major).
struct F5DebugTaps {
    std::vector<float> * text_embed = nullptr;      // after lookup + pos (01)
    std::vector<float> * text_convnext = nullptr;   // after 4 ConvNeXt (02)
    std::vector<float> * text_padded = nullptr;     // after pad/curtail (03)
    std::vector<float> * input_embed = nullptr;     // after proj + cpe (04)
    std::vector<float> * time_embed = nullptr;      // (05)
    std::vector<float> * block0 = nullptr;          // (07_block0)
    std::vector<float> * block21 = nullptr;         // (07_block21)
};

// Full DiT velocity forward for one Euler step inputs.
// x/cond: [seq_len * mel_dim] row-major (seq-major), text: ids (0 = filler),
// returns [seq_len * mel_dim] column layout (out[m * N + n]).
std::vector<float> f5_dit_forward(
    const std::string & weights_path,
    const std::vector<float> & x,
    const std::vector<float> & cond,
    const std::vector<int32_t> & text,
    float time_value,
    int seq_len,
    const F5Architecture & arch,
    bool drop_audio_cond,
    bool drop_text,
    const F5DebugTaps * taps = nullptr,
    const F5ComputeDevice * device = nullptr);

// Batched CFG: one ne3=2 graph compute returning {conditioned, unconditioned}
// velocities. Matches python cfg_infer: the uncond half runs with
// drop_audio_cond (zeroed cond) and drop_text (filler text id 0); the host
// upload in runtime.cpp prepares both halves accordingly.
std::pair<std::vector<float>, std::vector<float>> f5_dit_forward_cfg(
    const std::string & weights_path,
    const std::vector<float> & x,
    const std::vector<float> & cond,
    const std::vector<int32_t> & text,
    float time_value,
    int seq_len,
    const F5Architecture & arch,
    const F5ComputeDevice * device = nullptr);

}  // namespace engine::models::f5_tts
