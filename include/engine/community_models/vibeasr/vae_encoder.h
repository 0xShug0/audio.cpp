#pragma once

// VibeASR audio VAE encoder: a ConvNeXt-style causal encoder that turns a mono
// waveform into LM-width features, running end to end in GGML_TYPE_I8_S.
//
// Ported from https://github.com/microsoft/VibeASR.cpp (src/vae.cpp).

#include "engine/community_models/vibeasr/assets.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::vibeasr {

struct VaeBlockWeights {
    core::TensorValue mixer_norm;         // [channels]
    core::TensorValue mixer_conv_weight;  // [channels, 1, kernel_size], I8_S
    core::TensorValue mixer_conv_bias;    // [channels]
    core::TensorValue mixer_gamma;        // [channels]
    core::TensorValue ffn_norm;           // [channels]
    core::TensorValue ffn_fc1_weight;     // [ffn_hidden, channels], I8_S
    core::TensorValue ffn_fc1_bias;       // [ffn_hidden]
    core::TensorValue ffn_fc2_weight;     // [channels, ffn_hidden], I8_S
    core::TensorValue ffn_fc2_bias;       // [channels]
    core::TensorValue ffn_gamma;          // [channels]
};

struct VaeStageWeights {
    core::TensorValue downsample_weight;  // [out_channels, in_channels, kernel_size], I8_S
    core::TensorValue downsample_bias;    // [out_channels]
    std::vector<VaeBlockWeights> blocks;
};

struct VaeBranchWeights {
    std::vector<VaeStageWeights> stages;
    core::TensorValue head_weight;           // [latent_dim, channels, kernel_size], I8_S
    core::TensorValue head_bias;             // [latent_dim]
    core::TensorValue connector_fc1_weight;  // [connector_hidden, latent_dim], I8_S
    core::TensorValue connector_fc1_bias;    // [connector_hidden]
    core::TensorValue connector_norm;        // [connector_hidden]
    core::TensorValue connector_fc2_weight;  // [connector_hidden, connector_hidden], I8_S
    core::TensorValue connector_fc2_bias;    // [connector_hidden]
};

struct VibeASRVaeEncoderWeights {
    VaeBranchWeights acoustic;
    VaeBranchWeights semantic;
};

// Encoder output, row-major [frames][dim].
struct VaeEncoderFeatures {
    int64_t frames = 0;
    int64_t dim = 0;
    std::vector<float> values;
};

class VibeASRVaeEncoderRuntime {
public:
    VibeASRVaeEncoderRuntime(
        std::shared_ptr<const VibeASRVaeAssets> assets,
        engine::core::ExecutionContext & execution_context,
        size_t graph_arena_bytes = 64ull * 1024ull * 1024ull);

    // Both branches consume the same waveform, sampled at 24 kHz and scaled to
    // [-1, 1], and produce connector_hidden-wide features.
    VaeEncoderFeatures encode_acoustic(const std::vector<float> & samples);
    VaeEncoderFeatures encode_semantic(const std::vector<float> & samples);

    const VibeASRVaeAssets & assets() const noexcept { return *assets_; }

private:
    VaeEncoderFeatures encode(
        const VaeBranchConfig & config,
        const VaeBranchWeights & weights,
        const std::vector<float> & samples);

    std::shared_ptr<const VibeASRVaeAssets> assets_;
    engine::core::ExecutionContext * execution_context_ = nullptr;
    engine::core::BackendWeightStore weight_store_;
    VibeASRVaeEncoderWeights weights_;
    size_t graph_arena_bytes_;
};

}  // namespace engine::community_models::vibeasr
