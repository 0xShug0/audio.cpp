#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/higgs_audio_stt/assets.h"

#include <memory>

namespace engine::models::higgs_audio_stt {

class HiggsAudioSTTLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    HiggsAudioSTTLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const HiggsAudioSTTAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const HiggsAudioSTTAssets> assets_;
};

std::unique_ptr<HiggsAudioSTTLoadedModel> load_higgs_audio_stt_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_higgs_audio_stt_loader();

}  // namespace engine::models::higgs_audio_stt
