#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/models/stable_audio/assets.h"
#include "engine/models/stable_audio/conditioner.h"
#include "engine/models/stable_audio/foundation/conditioner.h"
#include "engine/models/stable_audio/foundation/oobleck_autoencoder.h"
#include "engine/models/stable_audio/foundation/rf_dit.h"

#include <memory>
#include <string>

namespace engine::models::stable_audio::foundation {

class FoundationSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    FoundationSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const StableAudioAssets> assets);
    ~FoundationSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const StableAudioAssets> assets_;
    engine::assets::TensorStorageType weight_storage_type_ = engine::assets::TensorStorageType::Native;
    std::unique_ptr<StableAudioConditionerInputs> conditioner_inputs_;
    std::unique_ptr<FoundationConditionerRuntime> conditioner_runtime_;
    std::unique_ptr<FoundationRfDitRuntime> rf_dit_;
    std::unique_ptr<OobleckAutoencoderRuntime> oobleck_;
    int64_t max_batch_ = 1;
};

}  // namespace engine::models::stable_audio::foundation
