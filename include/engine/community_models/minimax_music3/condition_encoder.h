#pragma once

#include "engine/community_models/minimax_music3/assets.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conv_modules.h"

#include <memory>
#include <vector>

namespace engine::models::minimax_music3 {

struct MiniMaxMusic3ConditionWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    std::vector<float> layer_weights;
    modules::Conv1dWeights projection;
};

class MiniMaxMusic3ConditionEncoderRuntime {
public:
    MiniMaxMusic3ConditionEncoderRuntime(
        std::shared_ptr<const MiniMaxMusic3Assets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type);
    ~MiniMaxMusic3ConditionEncoderRuntime();

    std::vector<float> encode(const std::vector<float> & frame_hiddens, int64_t frames, int64_t & condition_frames);
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::minimax_music3
