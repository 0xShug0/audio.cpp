#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"

#include <memory>

namespace engine::models::echo_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_echo_tts_loader();

class EchoTtsSession final
    : public runtime::RuntimeSessionBase,
      public runtime::IOfflineVoiceTaskSession {
public:
    EchoTtsSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~EchoTtsSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;
    void reset();

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
};

}  // namespace engine::models::echo_tts
