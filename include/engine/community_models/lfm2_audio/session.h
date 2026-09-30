#pragma once

// Offline ASR session for LFM2.5-Audio: audio -> FastConformer -> adapter ->
// LFM2 backbone, prompted with the ASR system prompt from liquid-audio's
// README and decoded greedily like LFM2AudioModel.generate_sequential
// (model/lfm2_audio.py).

#include "engine/community_models/lfm2_audio/asr_inputs.h"
#include "engine/community_models/lfm2_audio/assets.h"
#include "engine/community_models/lfm2_audio/audio_encoder.h"
#include "engine/community_models/lfm2_audio/backbone.h"
#include "engine/community_models/lfm2_audio/tokenizer.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"

#include <cstdint>
#include <filesystem>
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

    // Whether the last run() cut a chunk's transcript off at max_tokens.
    [[nodiscard]] bool reached_max_tokens() const;

private:
    struct RequestOptions {
        int64_t max_tokens = 512;
    };

    RequestOptions parse_request_options(const runtime::TaskRequest & request) const;
    std::vector<runtime::TimeSpan> plan_chunks(const runtime::TaskRequest & request, const std::vector<float> & samples);
    runtime::IOfflineVoiceTaskSession & vad_session();
    std::string transcribe(const std::vector<float> & samples, const runtime::TimeSpan & span, const RequestOptions & options);

    runtime::TaskSpec task_;
    std::shared_ptr<const Lfm2AudioAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::shared_ptr<const Lfm2AudioComponents> components_;
    Lfm2TextTokenizer tokenizer_;
    Lfm2AudioFeatureExtractor features_;
    Lfm2FastConformerEncoderRuntime encoder_;
    Lfm2BackboneRuntime backbone_;
    std::string language_;
    Lfm2AsrPrompt prompt_;
    std::filesystem::path vad_model_path_;
    std::unique_ptr<runtime::ILoadedVoiceModel> vad_model_;
    std::unique_ptr<runtime::IOfflineVoiceTaskSession> vad_session_;
    bool reached_max_tokens_ = false;
};

}  // namespace engine::community_models::lfm2_audio
