#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/index_tts2_5/assets.h"

#include <filesystem>
#include <memory>

namespace engine::models::index_tts2_5 {

class IndexTTS25LoadedModel final : public runtime::ILoadedVoiceModel {
public:
    IndexTTS25LoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const IndexTTS25Assets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const IndexTTS25Assets> assets_;
};

std::unique_ptr<IndexTTS25LoadedModel> load_index_tts2_5_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_index_tts2_5_loader();

}  // namespace engine::models::index_tts2_5
