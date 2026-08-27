#pragma once

#include "engine/community_models/minimax_music3/assets.h"
#include "engine/community_models/minimax_music3/depth_decoder.h"
#include "engine/community_models/minimax_music3/global_lm.h"
#include "engine/community_models/minimax_music3/prompt.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/sampling/torch_random.h"

#include <cstdint>
#include <functional>
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
        assets::TensorStorageType storage_type,
        bool evict_cuda_graph_cache_on_release = false);
    ~MiniMaxMusic3ArRuntime();

    std::vector<float> generate_frame_hiddens(
        const MiniMaxMusic3Request & request,
        int64_t target_frames,
        uint64_t & rng_offset_blocks);

    // Identical generation, but appends into a caller-owned buffer (which must
    // survive without reallocation: callers reserve target capacity up front)
    // and reports completed hidden rows so a consumer thread can start work on
    // finished frames while later frames are still decoding.
    void generate_frame_hiddens_into(
        const MiniMaxMusic3Request & request,
        int64_t target_frames,
        uint64_t & rng_offset_blocks,
        std::vector<float> & frame_hiddens,
        const std::function<void(int64_t rows_done)> * progress);

    // Ensemble decode: K independent takes of the same prompt advance in one
    // batched pass (global LM and depth run at batch 2K), each take sampling
    // with its own seed. Weight reads are amortized across takes, which is
    // where the bandwidth-bound AR stage spends its time. Returns per-take
    // frame hiddens; rng_offset_blocks[i] carries take i's counter onward.
    std::vector<std::vector<float>> generate_frame_hiddens_ensemble(
        const MiniMaxMusic3Request & request,
        int64_t target_frames,
        const std::vector<uint64_t> & take_seeds,
        std::vector<uint64_t> & rng_offset_blocks);

    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::minimax_music3
