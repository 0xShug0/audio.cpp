#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/voxtral_realtime/assets.h"

#include <filesystem>
#include <memory>

namespace engine::models::voxtral_realtime {

class VoxtralRealtimeLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    VoxtralRealtimeLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const VoxtralRealtimeAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const VoxtralRealtimeAssets> assets_;
};

std::unique_ptr<VoxtralRealtimeLoadedModel> load_voxtral_realtime_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_voxtral_realtime_loader();

}  // namespace engine::models::voxtral_realtime
