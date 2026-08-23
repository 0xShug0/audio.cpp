#pragma once

#include "engine/community_models/minimax_music3/assets.h"
#include "engine/community_models/minimax_music3/depth_decoder.h"
#include "engine/community_models/minimax_music3/global_lm.h"
#include "engine/community_models/minimax_music3/prompt.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/sampling/torch_random.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::minimax_music3 {

class MiniMaxMusic3ArRuntime {
public:
    MiniMaxMusic3ArRuntime(
        std::shared_ptr<const MiniMaxMusic3Assets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type);
    ~MiniMaxMusic3ArRuntime();

    std::vector<float> generate_frame_hiddens(
        const MiniMaxMusic3Request & request,
        int64_t target_frames,
        uint64_t & rng_offset_blocks);

    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::minimax_music3
