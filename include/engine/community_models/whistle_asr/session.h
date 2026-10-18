#pragma once

#include "engine/community_models/whistle_asr/assets.h"
#include "engine/community_models/whistle_asr/runtime.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"

#include <memory>
#include <string>

namespace engine::community_models::whistle_asr {

[[nodiscard]] std::shared_ptr<runtime::IVoiceModelLoader> make_whistle_asr_loader();

class WhistleAsrSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    WhistleAsrSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const WhistleAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    WhistleRuntime runtime_;
};

}  // namespace engine::community_models::whistle_asr
