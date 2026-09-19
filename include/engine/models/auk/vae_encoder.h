#pragma once

// AuK's audio encoder: waveform -> latent distribution.
//
// Tencent-Hunyuan/AuK, bigvgan_flow_vae.py:Encoder. A plain nn.Sequential, which is
// where it parts company with dots_tts's otherwise closely related audio VAE: that one
// carries a post-encoder LSTM stage and a lookahead-padded tail, and its loader rejects
// mi_num_layers <= 0, so it cannot stand in for this.
//
// ⚠ The encoder is NON-causal even though the decoder is causal. Conv1d_S pads
// symmetrically with dilation*(kernel-1)/2 regardless of the config's `causal` flag,
// which only reaches the decoder side.

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::auk {

struct AukVaeEncoderConfig {
    int64_t latent_dim = 64;
    int64_t in_channels = 1;
    int64_t proj_kernel_size = 3;
    int64_t stack_kernel_size = 3;
    int64_t stack_dilation_base = 2;
    int64_t stacks = 6;
    std::vector<int64_t> downsample_rates{2, 2, 2, 3, 4, 5};
    std::vector<int64_t> downsample_channels{12, 24, 48, 96, 192, 384, 768};

    // The encoder emits mean and log_std concatenated, so 2 * latent_dim channels.
    int64_t output_channels() const { return latent_dim * 2; }
    int64_t hop_size() const;
    void validate() const;
};

struct AukLatentDistribution {
    std::vector<float> values;   // channel-major, [channels, frames]
    int64_t channels = 0;
    int64_t frames = 0;
};

class AukVaeEncoder {
public:
    AukVaeEncoder(
        AukVaeEncoderConfig config,
        const assets::TensorSource & source,
        core::ExecutionContext & execution,
        std::string prefix = "audio_encoder");
    ~AukVaeEncoder();

    AukVaeEncoder(const AukVaeEncoder &) = delete;
    AukVaeEncoder & operator=(const AukVaeEncoder &) = delete;

    // ⚠ Returns the DISTRIBUTION, not a latent. encoding_and_normalization() draws
    // `mean + randn * exp(log_std)` on top of this; the draw is the caller's business
    // precisely because it is the stochastic part.
    AukLatentDistribution encode(const std::vector<float> & waveform);

    const AukVaeEncoderConfig & config() const noexcept;
    int64_t loaded_tensor_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::auk
