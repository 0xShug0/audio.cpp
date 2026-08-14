#pragma once

#include "engine/community_models/minimax_music3/assets.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <memory>
#include <vector>

namespace engine::models::minimax_music3 {

struct MiniMaxMusic3FlowBlockWeights {
    modules::NormWeights norm1;
    modules::LinearWeights q;
    modules::LinearWeights k;
    modules::LinearWeights v;
    modules::LinearWeights out;
    modules::NormWeights norm2;
    modules::LinearWeights ff_in;
    modules::LinearWeights ff_out;
};

struct MiniMaxMusic3FlowWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::Conv1dWeights preprocess_conv;
    modules::LinearWeights proj_in;
    core::TensorValue time_proj_weight;
    modules::LinearWeights time_linear_1;
    modules::LinearWeights time_linear_2;
    std::vector<MiniMaxMusic3FlowBlockWeights> blocks;
    modules::LinearWeights proj_out;
    modules::Conv1dWeights postprocess_conv;
};

class MiniMaxMusic3FlowTransformerRuntime {
public:
    MiniMaxMusic3FlowTransformerRuntime(
        std::shared_ptr<const MiniMaxMusic3Assets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type);
    ~MiniMaxMusic3FlowTransformerRuntime();

    std::vector<float> predict_velocity_branches(
        const std::vector<float> & latents,
        const std::vector<float> & condition,
        int64_t latent_frames,
        float timestep);
    void prepare_chunk_condition(
        const std::vector<float> & condition,
        int64_t latent_frames);
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::minimax_music3
