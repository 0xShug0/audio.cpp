#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/audiosr/assets.h"

#include <memory>

namespace engine::models::audiosr {

class AudioSRPipelineRuntime;

std::shared_ptr<engine::runtime::IVoiceModelLoader> make_audiosr_loader();

class AudioSRSession final
    : public engine::runtime::RuntimeSessionBase
    , public engine::runtime::IOfflineVoiceTaskSession {
public:
    AudioSRSession(
        engine::runtime::TaskSpec task,
        engine::runtime::SessionOptions options,
        std::shared_ptr<const AudioSRAssets> assets);
    ~AudioSRSession() override;

    std::string family() const override;
    engine::runtime::VoiceTaskKind task_kind() const override;
    engine::runtime::RunMode run_mode() const override;
    void prepare(const engine::runtime::SessionPreparationRequest & request) override;
    engine::runtime::TaskResult run(const engine::runtime::TaskRequest & request) override;

private:
    engine::runtime::TaskSpec task_;
    engine::runtime::SessionOptions options_;
    std::shared_ptr<const AudioSRAssets> assets_;
    std::unique_ptr<engine::core::ExecutionContext> execution_;
    std::unique_ptr<AudioSRPipelineRuntime> pipeline_;
};

}  // namespace engine::models::audiosr
