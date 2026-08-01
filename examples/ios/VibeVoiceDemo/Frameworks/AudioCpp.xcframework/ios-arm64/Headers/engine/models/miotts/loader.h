#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/miotts/assets.h"

#include <memory>

namespace engine::models::miotts {

class MioTTSLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    MioTTSLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const MioTTSAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const MioTTSAssets> assets_;
};

std::unique_ptr<MioTTSLoadedModel> load_miotts_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_miotts_loader();

}  // namespace engine::models::miotts
