#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/demucs/assets.h"

#include <memory>

namespace engine::models::demucs {

class HTDemucsLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    HTDemucsLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const HTDemucsAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const HTDemucsAssets> assets_;
};

std::unique_ptr<runtime::ILoadedVoiceModel> load_htdemucs_model(
    const runtime::ModelLoadRequest & request);
std::shared_ptr<runtime::IVoiceModelLoader> make_htdemucs_loader();

}  // namespace engine::models::demucs
