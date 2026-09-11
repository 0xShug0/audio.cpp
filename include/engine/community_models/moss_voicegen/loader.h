#pragma once

#include "engine/community_models/moss_voicegen/assets.h"
#include "engine/framework/runtime/model.h"

#include <filesystem>
#include <memory>

namespace engine::models::moss_voicegen {

class MossVoiceGenLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    MossVoiceGenLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const MossVoiceGenAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const MossVoiceGenAssets> assets_;
};

std::unique_ptr<MossVoiceGenLoadedModel> load_moss_voicegen_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_moss_voicegen_loader();

}  // namespace engine::models::moss_voicegen
