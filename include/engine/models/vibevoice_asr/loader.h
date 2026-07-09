#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/vibevoice_asr/assets.h"

#include <memory>

namespace engine::models::vibevoice_asr {

class VibeVoiceASRLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    VibeVoiceASRLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const VibeVoiceAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const VibeVoiceAssets> assets_;
};

std::unique_ptr<VibeVoiceASRLoadedModel> load_vibevoice_asr_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_vibevoice_asr_loader();

}  // namespace engine::models::vibevoice_asr
