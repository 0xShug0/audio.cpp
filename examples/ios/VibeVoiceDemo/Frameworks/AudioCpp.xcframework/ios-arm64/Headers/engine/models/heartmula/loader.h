#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/heartmula/assets.h"

#include <filesystem>
#include <memory>

namespace engine::models::heartmula {

class HeartMuLaLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    HeartMuLaLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const HeartMuLaAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const HeartMuLaAssets> assets_;
};

std::unique_ptr<HeartMuLaLoadedModel> load_heartmula_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_heartmula_loader();

}  // namespace engine::models::heartmula
