#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/nemotron_asr/assets.h"

#include <filesystem>
#include <memory>

namespace engine::models::nemotron_asr {

class NemotronASRLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    NemotronASRLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const NemotronAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const NemotronAssets> assets_;
};

std::unique_ptr<NemotronASRLoadedModel> load_nemotron_asr_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_nemotron_asr_loader();

}  // namespace engine::models::nemotron_asr
