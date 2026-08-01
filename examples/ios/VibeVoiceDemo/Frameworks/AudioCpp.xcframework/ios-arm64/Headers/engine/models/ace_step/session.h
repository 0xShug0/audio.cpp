#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/models/ace_step/assets.h"

#include <memory>

#include "engine/models/ace_step/diffusion.h"
#include "engine/models/ace_step/dit_weights_runtime.h"
#include "engine/models/ace_step/planner.h"
#include "engine/models/ace_step/pre_dit.h"
#include "engine/models/ace_step/vae_decoder.h"

namespace engine::models::ace_step {

class AceStepSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    AceStepSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const AceStepAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    void ensure_planner();
    void ensure_dit_weights_runtime();
    void ensure_pre_dit();
    void ensure_diffusion();
    void ensure_vae_decoder();

    runtime::TaskSpec task_;
    std::shared_ptr<const AceStepAssets> assets_;
    assets::TensorStorageType planner_weight_storage_type_ = assets::TensorStorageType::Native;
    assets::TensorStorageType text_encoder_weight_storage_type_ = assets::TensorStorageType::Native;
    assets::TensorStorageType dit_weight_storage_type_ = assets::TensorStorageType::Native;
    assets::TensorStorageType vae_weight_storage_type_ = assets::TensorStorageType::Native;
    bool mem_saver_ = false;
    std::shared_ptr<const AceStepDitWeightsRuntime> dit_weights_runtime_;
    std::unique_ptr<AceStepPlannerRuntime> planner_;
    std::unique_ptr<AceStepPreDitRuntime> pre_dit_;
    std::unique_ptr<AceStepDiffusionRuntime> diffusion_;
    std::shared_ptr<AceStepVAEDecoderRuntime> vae_decoder_;
};

}  // namespace engine::models::ace_step
