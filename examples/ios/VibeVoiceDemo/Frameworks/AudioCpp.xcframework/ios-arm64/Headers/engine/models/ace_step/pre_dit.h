#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/ace_step/assets.h"
#include "engine/models/ace_step/condition_encoder.h"
#include "engine/models/ace_step/cover_tokenizer.h"
#include "engine/models/ace_step/detokenizer.h"
#include "engine/models/ace_step/dit_weights_runtime.h"
#include "engine/models/ace_step/planner.h"
#include "engine/models/ace_step/task_route.h"
#include "engine/models/ace_step/text_encoder.h"
#include "engine/models/ace_step/types.h"
#include "engine/models/ace_step/vae_encoder.h"

#include <memory>

namespace engine::models::ace_step {

class AceStepPreDitRuntime {
public:
    AceStepPreDitRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const AceStepAssets> assets,
        std::shared_ptr<const AceStepDitWeightsRuntime> dit_weights_runtime,
        assets::TensorStorageType dit_weight_storage_type = assets::TensorStorageType::Native,
        assets::TensorStorageType vae_encoder_weight_storage_type = assets::TensorStorageType::Native,
        assets::TensorStorageType text_encoder_weight_storage_type = assets::TensorStorageType::Native);

    AceStepPreDitInputs prepare(
        const AceStepRequest & request,
        const AceStepTaskRoute & route,
        const AceStepPlan & plan) const;
    void prepare_runtime() const;
    void release_runtime_graphs() const;

private:
    void ensure_text_encoder() const;
    void ensure_condition_encoder() const;
    void ensure_cover_tokenizer() const;
    void ensure_detokenizer() const;
    void ensure_vae_encoder() const;
    void release_vae_encoder() const;

    std::shared_ptr<const AceStepAssets> assets_;
    std::shared_ptr<const AceStepDitWeightsRuntime> dit_weights_runtime_;
    assets::TensorStorageType dit_weight_storage_type_ = assets::TensorStorageType::Native;
    assets::TensorStorageType vae_encoder_weight_storage_type_ = assets::TensorStorageType::Native;
    assets::TensorStorageType text_encoder_weight_storage_type_ = assets::TensorStorageType::Native;
    AceStepTextTokenizer tokenizer_;
    core::ExecutionContext * execution_ = nullptr;
    mutable std::unique_ptr<AceStepQwenTextEncoderRuntime> text_encoder_;
    mutable std::unique_ptr<AceStepConditionEncoderRuntime> condition_encoder_;
    mutable std::unique_ptr<AceStepCoverTokenizerRuntime> cover_tokenizer_;
    mutable std::unique_ptr<AceStepAudioDetokenizerRuntime> detokenizer_;
    mutable std::unique_ptr<AceStepVAEEncoderRuntime> vae_encoder_;
    std::vector<float> silence_latent_;
    int64_t silence_latent_frames_ = 0;
    int64_t silence_latent_channels_ = 0;
};

}  // namespace engine::models::ace_step
