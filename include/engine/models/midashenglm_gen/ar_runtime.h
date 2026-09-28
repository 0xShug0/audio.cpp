#pragma once

#include "engine/models/midashenglm_gen/assets.h"
#include "engine/models/midashenglm_gen/flow_runtime.h"
#include "engine/models/midashenglm_gen/prompt_encoder.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"

#include <memory>
#include <vector>

namespace engine::models::midashenglm_gen {

struct MiDashengLmGenGenerationOptions {
    int64_t seq_len = 500;
    float eval_cfg = 2.0F;
    float stop_threshold = 0.5F;
    int64_t min_stop_step = 5;
    uint32_t seed = 0;
};

struct MiDashengLmGenAROutput {
    std::vector<float> latents;
    std::vector<float> stop_probs;
    int64_t batch = 0;
    int64_t frames = 0;
    int64_t dims = 0;
};

struct MiDashengLmGenQwen3ARWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::modules::CausalDecoderRuntimeWeights qwen;
    engine::modules::LinearWeights audio_projector_in;
    engine::modules::LinearWeights audio_projector_out;
    engine::modules::LinearWeights stop_head;
};

class MiDashengLmGenQwen3ARRuntime {
public:
    MiDashengLmGenQwen3ARRuntime(
        std::shared_ptr<const MiDashengLmGenAssets> assets,
        engine::core::ExecutionContext & execution,
        MiDashengLmGenDiTFlowRuntime & flow,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t helper_graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type);
    ~MiDashengLmGenQwen3ARRuntime();

    MiDashengLmGenQwen3ARRuntime(const MiDashengLmGenQwen3ARRuntime &) = delete;
    MiDashengLmGenQwen3ARRuntime & operator=(const MiDashengLmGenQwen3ARRuntime &) = delete;

    MiDashengLmGenAROutput generate(
        const MiDashengLmGenPromptEncoderOutput & prompt,
        const MiDashengLmGenGenerationOptions & options);

    void release_graphs();

private:
    class ProjectorGraph;
    class StopHeadGraph;

    std::shared_ptr<const MiDashengLmGenAssets> assets_;
    engine::core::ExecutionContext * execution_ = nullptr;
    MiDashengLmGenDiTFlowRuntime * flow_ = nullptr;
    size_t helper_graph_arena_bytes_ = 0;
    std::shared_ptr<const MiDashengLmGenQwen3ARWeights> weights_;
    std::unique_ptr<engine::modules::CausalDecoderRuntime> qwen3_runtime_;
    std::unique_ptr<ProjectorGraph> projector_;
    std::unique_ptr<StopHeadGraph> stop_head_;
};

}  // namespace engine::models::midashenglm_gen
