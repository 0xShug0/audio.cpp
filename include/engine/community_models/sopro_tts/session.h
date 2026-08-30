#pragma once

#include "engine/community_models/sopro_tts/assets.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::sopro_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_sopro_tts_loader();

class SoproAcousticRuntime;
class SoproReferenceBuilder;
class SoproSemanticEncoderRuntime;
class SoproSemanticLMRuntime;
class SoproSpeakerEncoderRuntime;
class SoproTextTokenizer;
class SoproVocoderRuntime;

class SoproTTSSession final : public runtime::RuntimeSessionBase,
                              public runtime::IOfflineVoiceTaskSession {
public:
    SoproTTSSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const SoproTTSAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~SoproTTSSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    SoproRequestOptions parse_options(const runtime::TaskRequest & request) const;

    runtime::TaskSpec task_;
    std::shared_ptr<const SoproTTSAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::string default_language_;

    std::unique_ptr<SoproTextTokenizer> tokenizer_;
    std::unique_ptr<SoproSpeakerEncoderRuntime> speaker_encoder_;
    std::unique_ptr<SoproSemanticEncoderRuntime> semantic_encoder_;
    std::unique_ptr<SoproVocoderRuntime> vocoder_;
    std::unique_ptr<SoproSemanticLMRuntime> semantic_lm_;
    std::unique_ptr<SoproAcousticRuntime> acoustic_;
    std::unique_ptr<SoproReferenceBuilder> reference_builder_;
};

}  // namespace engine::community_models::sopro_tts
