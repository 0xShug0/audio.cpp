#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/community_models/minimax_music3/assets.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::minimax_music3 {

// Projects per-frame AR hidden states onto the Flow-VAE latent timeline: softmax-weighted
// mix of the per-frame hidden slots, learned scale, 3-tap Conv1d to the condition width,
// and nearest-neighbor resampling from the frame rate to the latent rate.
class MiniMaxMusic3ConditionEncoderRuntime final {
public:
    MiniMaxMusic3ConditionEncoderRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const assets::TensorSource> source,
        const MiniMaxMusic3Config & config,
        size_t weight_context_bytes);
    ~MiniMaxMusic3ConditionEncoderRuntime();

    // frame_hiddens: row-major [frames, cond_layers * cond_hidden].
    // Returns row-major [latent_length(frames), cond_out_dim].
    std::vector<float> encode(const std::vector<float> & frame_hiddens, int64_t frames);

    int64_t latent_length(int64_t frames) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::minimax_music3
