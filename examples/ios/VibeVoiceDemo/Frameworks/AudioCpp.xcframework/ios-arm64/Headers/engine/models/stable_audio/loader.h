#pragma once

#include "engine/framework/runtime/registry.h"
#include "engine/models/stable_audio/assets.h"

#include <filesystem>
#include <memory>

namespace engine::models::stable_audio {

class StableAudioLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    StableAudioLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const StableAudioAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;

    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const StableAudioAssets> assets_;
};

std::unique_ptr<StableAudioLoadedModel> load_stable_audio_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_stable_audio_loader();

}  // namespace engine::models::stable_audio
