#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/models/stable_audio/assets.h"
#include "engine/models/stable_audio/conditioner.h"
#include "engine/models/stable_audio/conditioner_runtime.h"
#include "engine/models/stable_audio/rf_dit.h"
#include "engine/models/stable_audio/same_autoencoder.h"

#include <memory>
#include <string>

namespace engine::models::stable_audio {

class StableAudioSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    StableAudioSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const StableAudioAssets> assets);
    ~StableAudioSession() override;

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
    std::unique_ptr<StableAudioConditionerRuntime> conditioner_runtime_;
    std::unique_ptr<StableAudioRfDitRuntime> rf_dit_;
    std::unique_ptr<StableAudioSameRuntime> same_;
    int64_t max_batch_ = 1;
    bool mem_saver_ = false;
};

}  // namespace engine::models::stable_audio
