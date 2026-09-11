#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/moonshine_stt/assets.h"

#include <filesystem>
#include <memory>

namespace engine::models::moonshine_stt {

class MoonshineSTTLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    MoonshineSTTLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const MoonshineAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const MoonshineAssets> assets_;
};

std::unique_ptr<MoonshineSTTLoadedModel> load_moonshine_stt_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_moonshine_stt_loader();

}  // namespace engine::models::moonshine_stt
