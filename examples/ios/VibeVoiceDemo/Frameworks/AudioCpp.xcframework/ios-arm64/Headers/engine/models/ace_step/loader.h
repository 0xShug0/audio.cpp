#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/ace_step/assets.h"

#include <filesystem>
#include <memory>

namespace engine::models::ace_step {

class AceStepLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    AceStepLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const AceStepAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const AceStepAssets> assets_;
};

std::unique_ptr<AceStepLoadedModel> load_ace_step_model(
    const std::filesystem::path & model_path,
    const AceStepModelSelection & selection = {});

std::shared_ptr<runtime::IVoiceModelLoader> make_ace_step_loader();

}  // namespace engine::models::ace_step
