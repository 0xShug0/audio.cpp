#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/kokoro_tts/assets.h"

#include <filesystem>
#include <memory>

namespace engine::models::kokoro_tts {

class KokoroTTSLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    KokoroTTSLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const KokoroAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const KokoroAssets> assets_;
};

std::unique_ptr<KokoroTTSLoadedModel> load_kokoro_tts_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_kokoro_tts_loader();

}  // namespace engine::models::kokoro_tts
