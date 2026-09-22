#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/community_models/vieneu_v3_turbo/assets.h"

#include <filesystem>
#include <memory>

namespace engine::models::vieneu_v3_turbo {

class VieNeuTTSLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    VieNeuTTSLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const VieNeuTTSAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const VieNeuTTSAssets> assets_;
};

std::unique_ptr<VieNeuTTSLoadedModel> load_vieneu_v3_turbo_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_vieneu_v3_turbo_loader();

}  // namespace engine::models::vieneu_v3_turbo
