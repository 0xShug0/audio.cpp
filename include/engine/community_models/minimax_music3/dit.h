#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/community_models/minimax_music3/assets.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::minimax_music3 {

// The flow-matching transformer. One forward evaluates the conditional and unconditional
// CFG branches as a batch of two (the unconditional branch conditions on zeros) and
// returns the guided velocity.
class MiniMaxMusic3DitRuntime final {
public:
    MiniMaxMusic3DitRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const assets::TensorSource> source,
        const MiniMaxMusic3Config & config,
        size_t weight_context_bytes);
    ~MiniMaxMusic3DitRuntime();

    // Prepare graphs and upload the conditioning for one window.
    // condition: channel-major [condition_dim, length].
    void begin_chunk(const std::vector<float> & condition, int64_t length);

    // latent: channel-major [in_channels, length]; t in [0, 1]. Returns the guided
    // velocity, same layout as latent.
    std::vector<float> guided_velocity(const std::vector<float> & latent, float t, float guidance_scale);

    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::minimax_music3
