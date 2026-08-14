#pragma once

#include "engine/community_models/minimax_music3/ar_runtime.h"
#include "engine/community_models/minimax_music3/condition_encoder.h"
#include "engine/community_models/minimax_music3/flow_sampler.h"
#include "engine/community_models/minimax_music3/vocoder.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::minimax_music3 {

class MiniMaxMusic3PipelineRuntime {
public:
    MiniMaxMusic3PipelineRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const MiniMaxMusic3Assets> assets,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type,
        bool memory_saver);
    ~MiniMaxMusic3PipelineRuntime();

    runtime::AudioBuffer generate(const MiniMaxMusic3Request & request);
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::minimax_music3
