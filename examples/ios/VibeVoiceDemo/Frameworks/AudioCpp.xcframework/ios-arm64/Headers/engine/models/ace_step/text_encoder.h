#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/ace_step/assets.h"
#include "engine/models/ace_step/types.h"

#include <memory>
#include <vector>

namespace engine::models::ace_step {

class AceStepQwenTextEncoderRuntime {
public:
    AceStepQwenTextEncoderRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const AceStepAssets> assets,
        assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native);
    ~AceStepQwenTextEncoderRuntime();

    AceStepTextConditioning encode(const AceStepTokenizedText & tokens) const;
    AceStepTextConditioning embed_tokens(const AceStepTokenizedText & tokens) const;
    void prepare_runtime() const;
    void release_runtime_graphs() const;

private:
    class Graph;

    core::ExecutionContext * execution_ = nullptr;
    std::shared_ptr<const AceStepAssets> assets_;
    assets::TensorStorageType weight_storage_type_ = assets::TensorStorageType::Native;
    mutable std::unique_ptr<Graph> graph_;
};

}  // namespace engine::models::ace_step
