#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/marblenet_vad/assets.h"
#include "engine/models/marblenet_vad/runtime.h"

#include <memory>

namespace engine::models::marblenet_vad {

class MarbleNetVADSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    MarbleNetVADSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const MarbleNetWeights> weights);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    assets::TensorStorageType weight_storage_type_;
    MarbleNetRuntime runtime_;
};

class MarbleNetVADLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    MarbleNetVADLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const MarbleNetWeights> weights);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const MarbleNetWeights> weights_;
};

std::unique_ptr<MarbleNetVADLoadedModel> load_marblenet_vad_model(const runtime::ModelLoadRequest & request);
std::shared_ptr<runtime::IVoiceModelLoader> make_marblenet_vad_loader();

}  // namespace engine::models::marblenet_vad
