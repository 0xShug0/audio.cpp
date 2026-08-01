#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/ace_step/assets.h"
#include "engine/models/ace_step/dit_weights_runtime.h"
#include "engine/models/ace_step/types.h"

#include <memory>

namespace engine::models::ace_step {

class AceStepDiffusionRuntime {
public:
    AceStepDiffusionRuntime(
        std::shared_ptr<const AceStepAssets> assets,
        core::ExecutionContext & execution,
        std::shared_ptr<const AceStepDitWeightsRuntime> dit_weights_runtime,
        size_t graph_arena_bytes = 256ull * 1024ull * 1024ull);
    ~AceStepDiffusionRuntime();

    AceStepLatents generate_latents(
        const AceStepDiffusionConditioning & conditioning,
        const AceStepGenerationOptions & options) const;
    void release_graph_workspace() const;

private:
    class Impl;

    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::ace_step
