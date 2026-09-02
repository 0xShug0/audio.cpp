#pragma once

#include "engine/community_models/mira_tts/assets.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::community_models::mira_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_mira_tts_loader();

class MiraPromptBuilder;
class MiraSpeakerEncoder;
class MiraGenerator;
class MiraAcousticProcessor;
class MiraDecoder;

class MiraTTSOfflineSession final : public runtime::RuntimeSessionBase,
                                    public runtime::IOfflineVoiceTaskSession {
public:
    MiraTTSOfflineSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const MiraTTSAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~MiraTTSOfflineSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    const runtime::AudioBuffer & reference_audio(
        const runtime::TaskRequest & request) const;

    runtime::TaskSpec task_;
    std::shared_ptr<const MiraTTSAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::optional<runtime::AudioBuffer> prepared_reference_;
    std::unique_ptr<MiraPromptBuilder> prompt_;
    std::unique_ptr<MiraSpeakerEncoder> speaker_encoder_;
    std::unique_ptr<MiraGenerator> generator_;
    std::unique_ptr<MiraAcousticProcessor> processor_;
    std::unique_ptr<MiraDecoder> decoder_;
};

}  // namespace engine::community_models::mira_tts
