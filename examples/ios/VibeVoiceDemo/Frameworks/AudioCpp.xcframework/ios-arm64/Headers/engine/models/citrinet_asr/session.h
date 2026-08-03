#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/citrinet_asr/assets.h"
#include "engine/models/citrinet_asr/runtime.h"

#include <memory>

namespace engine::models::citrinet_asr {

class CitrinetASRSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    CitrinetASRSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const CitrinetWeights> weights);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    assets::TensorStorageType weight_storage_type_;
    CitrinetRuntime runtime_;
};

class CitrinetASRLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    CitrinetASRLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const CitrinetWeights> weights);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const CitrinetWeights> weights_;
};

std::unique_ptr<CitrinetASRLoadedModel> load_citrinet_asr_model(const runtime::ModelLoadRequest & request);
std::shared_ptr<runtime::IVoiceModelLoader> make_citrinet_asr_loader();

}  // namespace engine::models::citrinet_asr
