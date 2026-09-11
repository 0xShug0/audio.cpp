#pragma once

#include "engine/models/midashenglm_gen/assets.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/sampling/torch_random.h"

#include <memory>
#include <vector>

namespace engine::models::midashenglm_gen {

struct MiDashengLmGenFlowInput {
    std::vector<float> x;
    std::vector<float> latent_history;
    std::vector<float> condition;
    int64_t batch = 0;
};

struct MiDashengLmGenFlowBlockWeights {
    engine::modules::NormWeights norm1;
    engine::modules::AttentionWeights attention;
    engine::modules::NormWeights norm2;
    engine::modules::LinearWeights mlp_in;
    engine::modules::LinearWeights mlp_out;
};

struct MiDashengLmGenFlowWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::modules::LinearWeights time_in;
    engine::modules::LinearWeights time_out;
    engine::modules::LinearWeights x_embedder;
    engine::modules::LinearWeights c_embedder;
    engine::core::TensorValue fake_latent;
    std::vector<float> fake_latent_values;
    std::vector<MiDashengLmGenFlowBlockWeights> blocks;
    engine::modules::NormWeights final_norm;
    engine::modules::LinearWeights final_linear;
};

class MiDashengLmGenFlowRuntime {
public:
    MiDashengLmGenFlowRuntime(
        std::shared_ptr<const MiDashengLmGenAssets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type);
    ~MiDashengLmGenFlowRuntime();

    MiDashengLmGenFlowRuntime(const MiDashengLmGenFlowRuntime &) = delete;
    MiDashengLmGenFlowRuntime & operator=(const MiDashengLmGenFlowRuntime &) = delete;

    std::vector<float> sample(
        const MiDashengLmGenFlowInput & input,
        float cfg_scale,
        uint64_t seed,
        uint64_t & randn_offset_blocks);
    void release_graphs();

private:
    class DenoiserGraph;

    std::shared_ptr<const MiDashengLmGenAssets> assets_;
    engine::core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    engine::sampling::TorchCudaSamplingPolicy rng_policy_;
    std::shared_ptr<const MiDashengLmGenFlowWeights> weights_;
    std::unique_ptr<DenoiserGraph> graph_;
};

}  // namespace engine::models::midashenglm_gen
