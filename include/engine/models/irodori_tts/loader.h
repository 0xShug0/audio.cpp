#pragma once

#include "engine/framework/runtime/registry.h"
#include "engine/models/irodori_tts/assets.h"

#include <filesystem>
#include <memory>

namespace engine::models::irodori_tts {

class IrodoriTTSLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    IrodoriTTSLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const IrodoriAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;

    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const IrodoriAssets> assets_;
};

std::unique_ptr<IrodoriTTSLoadedModel> load_irodori_tts_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_irodori_tts_loader();

}  // namespace engine::models::irodori_tts
