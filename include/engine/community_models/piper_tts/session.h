#pragma once

#include "engine/community_models/piper_tts/assets.h"
#include "engine/community_models/piper_tts/frontend.h"
#include "engine/community_models/piper_tts/vits_runtime.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"

#include <memory>

namespace engine::models::piper_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_piper_tts_loader();

class PiperTtsSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    PiperTtsSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const PiperTtsAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~PiperTtsSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    PiperTtsGenerationOptions generation_options(
        const runtime::TaskRequest & request) const;

    runtime::TaskSpec task_;
    std::shared_ptr<const PiperTtsAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::unique_ptr<PiperTtsFrontend> frontend_;
    std::unique_ptr<PiperVitsRuntime> runtime_;
};

}  // namespace engine::models::piper_tts
