#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/firered_audio/assets.h"
#include "engine/models/firered_audio/tokenizer.h"

#include <memory>

namespace engine::models::firered_audio {

class FireRedAudioGenerationRuntime;
class FireRedAudioUnderstandingRuntime;

std::shared_ptr<engine::runtime::IVoiceModelLoader> make_firered_audio_loader();

class FireRedAudioSession final
    : public engine::runtime::RuntimeSessionBase
    , public engine::runtime::IOfflineVoiceTaskSession {
public:
    FireRedAudioSession(
        engine::runtime::TaskSpec task,
        engine::runtime::SessionOptions options,
        std::shared_ptr<const FireRedAudioAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~FireRedAudioSession() override;

    std::string family() const override;
    engine::runtime::VoiceTaskKind task_kind() const override;
    engine::runtime::RunMode run_mode() const override;
    void prepare(const engine::runtime::SessionPreparationRequest & request) override;
    engine::runtime::TaskResult run(const engine::runtime::TaskRequest & request) override;

private:
    engine::runtime::TaskSpec task_;
    std::shared_ptr<const FireRedAudioAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::unique_ptr<FireRedAudioTokenizer> tokenizer_;
    std::unique_ptr<FireRedAudioGenerationRuntime> runtime_;
    std::unique_ptr<FireRedAudioUnderstandingRuntime> understanding_runtime_;
    bool mem_saver_ = false;
};

}  // namespace engine::models::firered_audio
