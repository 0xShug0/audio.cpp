#pragma once

#include "engine/community_models/minimax_music3/flow_transformer.h"
#include "engine/framework/sampling/torch_random.h"

#include <memory>
#include <vector>

namespace engine::models::minimax_music3 {

class MiniMaxMusic3FlowSamplerRuntime {
public:
    MiniMaxMusic3FlowSamplerRuntime(
        std::shared_ptr<const MiniMaxMusic3Assets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type);
    ~MiniMaxMusic3FlowSamplerRuntime();

    std::vector<float> denoise_chunk(
        const std::vector<float> & condition_values,
        int64_t frames,
        const std::vector<float> & previous_latent,
        const std::vector<float> & previous_condition,
        const MiniMaxMusic3Request & request,
        uint64_t offset_blocks,
        const sampling::TorchCudaSamplingPolicy & sampling_policy,
        std::vector<float> & carry_condition,
        std::vector<float> & carry_latent);
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::minimax_music3
