#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/index_tts2/assets.h"

#include <filesystem>
#include <memory>

namespace engine::models::index_tts2 {

class IndexTTS2LoadedModel final : public runtime::ILoadedVoiceModel {
public:
    IndexTTS2LoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const IndexTTS2Assets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const IndexTTS2Assets> assets_;
};

std::unique_ptr<IndexTTS2LoadedModel> load_index_tts2_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_index_tts2_loader();

}  // namespace engine::models::index_tts2
