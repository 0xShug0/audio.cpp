#pragma once

#include "engine/community_models/niagara_asr/assets.h"
#include "engine/community_models/niagara_asr/frontend.h"
#include "engine/community_models/niagara_asr/runtime.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"

#include <memory>
#include <string>

namespace engine::community_models::niagara_asr {

std::shared_ptr<runtime::IVoiceModelLoader> make_niagara_asr_loader();

class NiagaraAsrSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    NiagaraAsrSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const NiagaraAsrAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~NiagaraAsrSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    std::string decode(const NiagaraInferenceResult & inference) const;

    runtime::TaskSpec task_;
    std::shared_ptr<const NiagaraAsrAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    NiagaraFrontend frontend_;
    NiagaraRuntime runtime_;
};

}  // namespace engine::community_models::niagara_asr
