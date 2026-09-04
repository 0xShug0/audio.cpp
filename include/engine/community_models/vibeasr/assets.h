#pragma once

// VibeASR VAE encoder assets.
//
// Ported from https://github.com/microsoft/VibeASR.cpp (src/vae.cpp).

#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::vibeasr {

// One ConvNeXt-style block inside a stage.
struct VaeBlockConfig {
    int64_t channels = 0;
    // Padded depthwise kernel width. The converter left-pads the real kernel
    // (7 taps) up to a SIMD-friendly width with leading zeros, so convolving
    // with the padded width and a matching causal left pad is bit-exact with
    // convolving the unpadded kernel.
    int64_t kernel_size = 0;
    int64_t ffn_hidden = 0;
};

struct VaeStageConfig {
    // Strided causal conv that enters the stage.
    int64_t downsample_kernel_size = 0;
    int64_t downsample_stride = 0;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    std::vector<VaeBlockConfig> blocks;
};

// One of the two encoder branches (acoustic / semantic). Both share the layout
// and differ only in latent width and stage depths.
struct VaeBranchConfig {
    std::string prefix;               // "acoustic" or "semantic"
    std::vector<VaeStageConfig> stages;
    int64_t head_kernel_size = 0;     // padded causal kernel of the latent head
    int64_t latent_dim = 0;           // head output width
    int64_t connector_hidden = 0;     // connector output width, i.e. LM hidden size
    int64_t total_stride = 0;         // product of the stage strides

    // Downsampling factor from waveform samples to encoder frames.
    [[nodiscard]] int64_t frames_for_samples(int64_t num_samples) const;
};

struct VibeASRVaeConfig {
    VaeBranchConfig acoustic;
    VaeBranchConfig semantic;
    // VibeASR's graph hardcodes 1e-5 for every RMS norm, including the ones the
    // checkpoint metadata labels 1e-6. The published weights were validated
    // against the hardcoded value, so the port keeps it.
    float rms_norm_eps = 1e-5f;
};

struct VibeASRVaeAssets {
    std::shared_ptr<const assets::TensorSource> source;
    VibeASRVaeConfig config;
};

// Derives the encoder geometry from the tensor table instead of GGUF metadata:
// stage depths come from which block tensors are present, channel counts and
// kernel widths from the weight shapes. That keeps the loader working for any
// VibeASR VAE checkpoint with this topology, and avoids trusting metadata the
// reference implementation itself ignores.
VibeASRVaeConfig derive_vae_config(const assets::TensorSource & source);

std::shared_ptr<const VibeASRVaeAssets> load_vibeasr_vae_assets(const std::filesystem::path & model_path);

}  // namespace engine::community_models::vibeasr
