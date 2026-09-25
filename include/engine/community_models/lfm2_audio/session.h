#pragma once

// Offline ASR session for LFM2.5-Audio: audio -> FastConformer -> adapter ->
// LFM2 backbone, prompted and decoded the way liquid-audio's
// generate_sequential does it for "Perform ASR.".

#include "engine/community_models/lfm2_audio/assets.h"
#include "engine/community_models/lfm2_audio/audio_encoder.h"
#include "engine/community_models/lfm2_audio/backbone.h"
#include "engine/community_models/lfm2_audio/tokenizer.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::lfm2_audio {

std::shared_ptr<runtime::IVoiceModelLoader> make_lfm2_audio_loader();

class Lfm2AudioSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    Lfm2AudioSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const Lfm2AudioAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~Lfm2AudioSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    struct RequestOptions {
        int64_t max_tokens = 512;
    };

    RequestOptions parse_request_options(const runtime::TaskRequest & request) const;
    std::vector<float> to_mono_16k(const runtime::AudioBuffer & audio) const;
    std::string transcribe(const std::vector<float> & samples, const RequestOptions & options);

    runtime::TaskSpec task_;
    std::shared_ptr<const Lfm2AudioAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::shared_ptr<const Lfm2AudioComponents> components_;
    Lfm2TextTokenizer tokenizer_;
    Lfm2FastConformerEncoderRuntime encoder_;
    Lfm2BackboneRuntime backbone_;
    std::string language_;
    std::vector<int32_t> prompt_prefix_;
    std::vector<int32_t> prompt_suffix_;
    std::vector<int32_t> stop_token_ids_;
};

}  // namespace engine::community_models::lfm2_audio
