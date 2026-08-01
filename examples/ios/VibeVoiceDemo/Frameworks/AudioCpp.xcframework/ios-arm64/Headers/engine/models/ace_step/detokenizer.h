#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/ace_step/assets.h"
#include "engine/models/ace_step/dit_weights_runtime.h"
#include "engine/models/ace_step/types.h"

#include <memory>
#include <vector>

namespace engine::models::ace_step {

class AceStepAudioDetokenizerRuntime {
public:
    AceStepAudioDetokenizerRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const AceStepAssets> assets,
        std::shared_ptr<const AceStepDitWeightsRuntime> dit_weights_runtime);
    ~AceStepAudioDetokenizerRuntime();

    AceStepLatents decode_audio_codes(const std::vector<int32_t> & audio_code_ids) const;
    void release_runtime_graphs() const;

private:
    class Impl;
    std::shared_ptr<const AceStepAssets> assets_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::ace_step
