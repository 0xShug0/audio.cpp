#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/community_models/parakeet_tdt/assets.h"

#include <memory>
#include <string>

namespace engine::community_models::parakeet_tdt {

class ParakeetTDTLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    ParakeetTDTLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const ParakeetTDTAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const ParakeetTDTAssets> assets_;
};

std::unique_ptr<ParakeetTDTLoadedModel> load_parakeet_model(const std::filesystem::path & model_path);

std::shared_ptr<runtime::IVoiceModelLoader> make_parakeet_tdt_loader();

}  // namespace engine::community_models::parakeet_tdt
