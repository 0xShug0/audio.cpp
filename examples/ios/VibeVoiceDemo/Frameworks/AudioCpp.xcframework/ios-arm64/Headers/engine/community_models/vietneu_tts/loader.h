#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/community_models/vietneu_tts/assets.h"

#include <filesystem>
#include <memory>

namespace engine::models::vietneu_tts {

class VietneuTTSLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    VietneuTTSLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const VietneuTTSAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const VietneuTTSAssets> assets_;
};

std::unique_ptr<VietneuTTSLoadedModel> load_vietneu_tts_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_vietneu_tts_loader();

}  // namespace engine::models::vietneu_tts
