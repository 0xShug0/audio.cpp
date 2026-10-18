#pragma once

#include "engine/community_models/moss_ttsd/assets.h"
#include "engine/framework/runtime/model.h"

#include <filesystem>
#include <memory>

namespace engine::models::moss_ttsd {

class LoadedModel final : public runtime::ILoadedVoiceModel {
public:
    LoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const Assets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const Assets> assets_;
};

std::unique_ptr<LoadedModel> load_moss_ttsd_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_moss_ttsd_loader();

}  // namespace engine::models::moss_ttsd
