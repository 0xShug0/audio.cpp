#pragma once

#include "engine/community_models/sanotts/assets.h"
#include "engine/community_models/sanotts/frontend.h"
#include "engine/community_models/sanotts/piper_runtime.h"
#include "engine/community_models/sanotts/runtime.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"

#include <memory>

namespace engine::models::sanotts {

std::shared_ptr<runtime::IVoiceModelLoader> make_sanotts_loader();

class SanoTtsSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    SanoTtsSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const SanoTtsAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~SanoTtsSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const SanoTtsAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::unique_ptr<SanoTtsFrontend> frontend_;            // nano graph
    std::unique_ptr<SanoTtsNativeRuntime> runtime_;         // nano graph
    std::unique_ptr<SanoTtsPiperFrontend> piper_frontend_;  // piperlite graph
    std::unique_ptr<SanoTtsPiperRuntime> piper_runtime_;    // piperlite graph
};

}  // namespace engine::models::sanotts
