#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/models/dramabox/assets.h"
#include "engine/models/dramabox/gemma3_encoder.h"

#include <memory>
#include <vector>

namespace engine::core {
class BackendWeightStore;
class ExecutionContext;
}

namespace engine::models::dramabox {

struct DramaBoxConnectorAttentionWeights {
    modules::LinearWeights q;
    modules::LinearWeights k;
    modules::LinearWeights v;
    modules::LinearWeights out;
    modules::LinearWeights gate;
    core::TensorValue q_norm;
    core::TensorValue k_norm;
};

struct DramaBoxConnectorBlockWeights {
    DramaBoxConnectorAttentionWeights attention;
    modules::LinearWeights ff_in;
    modules::LinearWeights ff_out;
};

struct DramaBoxPromptConnectorWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    std::vector<float> learnable_registers_host;
    std::vector<DramaBoxConnectorBlockWeights> blocks;
    core::TensorValue rope_cos;
    core::TensorValue rope_sin;
};

struct DramaBoxConditioningEncoding {
    int64_t batch = 0;
    int64_t tokens = 0;
    int64_t hidden_size = 0;
    std::vector<float> features;
};

DramaBoxPromptConnectorWeights load_dramabox_prompt_connector_weights(
    const DramaBoxAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type,
    int64_t sequence_length);

core::TensorValue build_dramabox_prompt_connector(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const DramaBoxPromptConnectorWeights & weights,
    const DramaBoxConfig & config,
    DramaBoxPerfMode perf_mode = DramaBoxPerfMode::Exact);

std::vector<float> make_branch_conditioning_features(
    const DramaBoxConditioningEncoding & conditioning,
    int64_t branch_count,
    bool cfg_enabled,
    bool stg_enabled);
DramaBoxConditioningEncoding select_conditioning_batch(
    const DramaBoxConditioningEncoding & conditioning,
    int64_t batch);
DramaBoxConditioningEncoding join_positive_negative_conditioning(
    const DramaBoxConditioningEncoding & positive,
    const DramaBoxConditioningEncoding & negative);

class DramaBoxPromptConnectorRuntime {
public:
    DramaBoxPromptConnectorRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const DramaBoxAssets> assets,
        assets::TensorStorageType weight_storage_type,
        int64_t max_batch,
        DramaBoxPerfMode perf_mode);
    ~DramaBoxPromptConnectorRuntime();

    DramaBoxPromptConnectorRuntime(const DramaBoxPromptConnectorRuntime &) = delete;
    DramaBoxPromptConnectorRuntime & operator=(const DramaBoxPromptConnectorRuntime &) = delete;

    void prepare(int64_t batch) const;
    DramaBoxConditioningEncoding encode(const DramaBoxPromptEncoding & prompt) const;
    void release_runtime_state() const;

private:
    class Graph;

    core::ExecutionContext * execution_ = nullptr;
    std::shared_ptr<const DramaBoxAssets> assets_;
    assets::TensorStorageType weight_storage_type_ = assets::TensorStorageType::Native;
    int64_t max_batch_ = 1;
    DramaBoxPerfMode perf_mode_ = DramaBoxPerfMode::Exact;
    mutable std::unique_ptr<DramaBoxPromptConnectorWeights> weights_;
    mutable std::unique_ptr<Graph> graph_;
};

}  // namespace engine::models::dramabox
