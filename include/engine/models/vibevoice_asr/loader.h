#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/vibevoice_asr/assets.h"

#include <memory>
#include <string>

namespace engine::models::vibevoice_asr {

class VibeVoiceASRLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    VibeVoiceASRLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const VibeVoiceASRAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const VibeVoiceASRAssets> assets_;
};

std::unique_ptr<VibeVoiceASRLoadedModel> load_vibevoice_asr_model(
    const std::filesystem::path & model_path,
    const std::string & family = "vibevoice_asr");
std::shared_ptr<runtime::IVoiceModelLoader> make_vibevoice_asr_loader();
std::shared_ptr<runtime::IVoiceModelLoader> make_vibevoice_asr_streaming_loader();

}  // namespace engine::models::vibevoice_asr
