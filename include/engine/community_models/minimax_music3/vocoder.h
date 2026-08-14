#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/community_models/minimax_music3/assets.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::models::minimax_music3 {

// Flow-VAE waveform decoder (DAC-style). Decodes flow-matched latents of logical shape
// [latent_channels, length] into an interleaved stereo waveform: the two audio channels
// run as two folded latent_channels / 2 streams through the same decoder.
class MiniMaxMusic3VocoderRuntime final {
public:
    MiniMaxMusic3VocoderRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const assets::TensorSource> source,
        const MiniMaxMusic3Config & config,
        size_t weight_context_bytes);
    ~MiniMaxMusic3VocoderRuntime();

    // latents: row-major [latent_channels, length]. Returns interleaved stereo samples.
    std::vector<float> decode(const std::vector<float> & latents, int64_t length);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::minimax_music3
