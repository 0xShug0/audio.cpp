#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/session_base.h"

#include <memory>
#include <string>

namespace engine::models::builtin_audio_utils {

class BuiltinAudioUtilsLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    BuiltinAudioUtilsLoadedModel(std::string model_id, std::filesystem::path asset);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    std::string model_id_;
    std::filesystem::path asset_;
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_builtin_audio_utils_loader();

}  // namespace engine::models::builtin_audio_utils
