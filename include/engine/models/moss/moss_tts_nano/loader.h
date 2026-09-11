#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/moss/moss_tts_nano/assets.h"

#include <filesystem>
#include <memory>

namespace engine::models::moss_tts_nano {

class MossTTSNanoLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    MossTTSNanoLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const MossTTSNanoAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const MossTTSNanoAssets> assets_;
};

std::unique_ptr<MossTTSNanoLoadedModel> load_moss_tts_nano_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_moss_tts_nano_loader();

}  // namespace engine::models::moss_tts_nano
