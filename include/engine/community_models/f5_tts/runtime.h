#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/core/module.h"
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

// Compute device for the DiT forward and the Vocos vocoder: a ggml backend
// plus its device index, or CPU threads.
struct F5ComputeDevice {
    // The backend both graphs run on. Anything other than Cpu takes the
    // device path (no_alloc context, gallocr arena, ggml_backend_tensor_set /
    // _get, compute_backend_graph); Cpu takes the inline-context host path
    // through ggml_graph_compute_with_ctx.
    //
    // This used to be `use_cuda` alone, so Metal and Vulkan fell to the host
    // path. Nothing in the DiT or the vocoder is CUDA-specific: both graphs
    // are composed from framework modules, every op they use is in the Metal
    // backend's supports_op table, and validate_backend_graph_supported is
    // called on both before allocation, so an unsupported op fails loudly
    // rather than silently. The narrow gate was untested caution — the port
    // was written against CUDA and no other GPU was tried — not a Metal bug.
    engine::core::BackendType backend = engine::core::BackendType::Cpu;
    // Legacy alias, still honoured for callers written before `backend`
    // existed (the parity harnesses under tests/f5_tts). Only consulted when
    // `backend` is left at Cpu.
    bool use_cuda = false;
    int device = 0;      // GPU device index
    int threads = 0;     // CPU threads; 0 = hardware concurrency
    // FP16 linear weights: GEMMs get ~3x faster on tensor cores but each
    // mul_mat converts the F32 activations to F16 first; at F5's GEMM sizes
    // (K=1024/2048, N~1022) the conversion overhead outweighs the gain on an
    // RTX 3090 (measured 4.0s -> 5.0s per clip). Off by default.
    bool fp16_weights = false;
};

// The backend a F5ComputeDevice actually asks for, folding the legacy
// `use_cuda` flag into the `backend` field. Pure function, no device needed:
// covered by tests/unittests/test_backend_path_selection.cpp.
constexpr engine::core::BackendType f5_requested_backend(const F5ComputeDevice & device) noexcept {
    if (device.backend != engine::core::BackendType::Cpu) {
        return device.backend;
    }
    return device.use_cuda ? engine::core::BackendType::Cuda : engine::core::BackendType::Cpu;
}

// True when the DiT and the vocoder should take the device (GPU) path.
constexpr bool f5_uses_device_backend(const F5ComputeDevice & device) noexcept {
    return f5_requested_backend(device) != engine::core::BackendType::Cpu;
}

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
