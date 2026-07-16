#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/roformer/assets.h"

#include <memory>

namespace engine::models::roformer {

class RoformerLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    RoformerLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const RoformerAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const RoformerAssets> assets_;
};

std::unique_ptr<runtime::ILoadedVoiceModel> load_roformer_model(
    const runtime::ModelLoadRequest & request);
std::shared_ptr<runtime::IVoiceModelLoader> make_mel_band_roformer_loader();

}  // namespace engine::models::roformer
