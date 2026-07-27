#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/confucius4_tts/assets.h"

#include <filesystem>
#include <memory>

namespace engine::models::confucius4_tts {

class ConfuciusLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    ConfuciusLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const ConfuciusAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const ConfuciusAssets> assets_;
};

std::unique_ptr<ConfuciusLoadedModel> load_confucius_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_confucius4_tts_loader();

}  // namespace engine::models::confucius4_tts
