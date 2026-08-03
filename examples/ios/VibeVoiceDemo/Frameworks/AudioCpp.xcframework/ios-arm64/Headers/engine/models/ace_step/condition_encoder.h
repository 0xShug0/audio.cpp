#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/ace_step/assets.h"
#include "engine/models/ace_step/dit_weights_runtime.h"
#include "engine/models/ace_step/types.h"

#include <memory>
#include <vector>

namespace engine::models::ace_step {

class AceStepConditionEncoderRuntime {
public:
    AceStepConditionEncoderRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const AceStepAssets> assets,
        std::shared_ptr<const AceStepDitWeightsRuntime> dit_weights_runtime);
    ~AceStepConditionEncoderRuntime();

    AceStepConditionEncoderRuntime(AceStepConditionEncoderRuntime &&) noexcept;
    AceStepConditionEncoderRuntime & operator=(AceStepConditionEncoderRuntime &&) noexcept;

    AceStepConditionEncoderRuntime(const AceStepConditionEncoderRuntime &) = delete;
    AceStepConditionEncoderRuntime & operator=(const AceStepConditionEncoderRuntime &) = delete;

    AceStepEncoderConditioning encode(
        const AceStepTextConditioning & text_hidden_states,
        const AceStepTextConditioning & lyric_token_embeddings,
        const std::vector<float> & refer_audio_acoustic_hidden_states_packed,
        int64_t refer_audio_count,
        int64_t refer_audio_frames,
        const std::vector<int32_t> & refer_audio_order_mask) const;
    void release_runtime_graphs() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::ace_step
