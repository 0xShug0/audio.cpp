#pragma once

#include "engine/community_models/glm_tts/assets.h"
#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::glm_tts {

class GlmTTSLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    GlmTTSLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const GlmTTSAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const GlmTTSAssets> assets_;
};

std::unique_ptr<GlmTTSLoadedModel> load_glm_tts_model(
    const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_glm_tts_loader();

}  // namespace engine::models::glm_tts
