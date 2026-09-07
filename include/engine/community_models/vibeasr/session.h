#pragma once

// Offline ASR session for the VibeASR package: I8_S VAE encoder -> ternary I2_S
// Qwen2 decoder, with VibeASR.cpp's ChatML prompt around the speech features.
//
// Ported from https://github.com/microsoft/VibeASR.cpp (src/asr_server.cpp,
// utils/prompt_builder.h).

#include "engine/community_models/vibeasr/assets.h"
#include "engine/community_models/vibeasr/lm_decoder.h"
#include "engine/community_models/vibeasr/vae_encoder.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <memory>
#include <string>

namespace engine::community_models::vibeasr {

std::shared_ptr<runtime::IVoiceModelLoader> make_vibeasr_loader();

class VibeASRSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    VibeASRSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const VibeASRAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~VibeASRSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    struct RequestOptions {
        std::string output_format = "text";
        std::string context;
        int64_t max_new_tokens = 1024;
    };

    RequestOptions parse_request_options(const runtime::TaskRequest & request) const;
    runtime::AudioBuffer normalize(const runtime::AudioBuffer & audio) const;
    VibeASRSpeechEmbeddings encode_speech(const std::vector<float> & samples);
    VibeASRLmPrompt build_prompt(int64_t speech_tokens, float duration_seconds, const RequestOptions & options) const;
    std::string decode_tokens(const std::vector<int32_t> & token_ids) const;

    runtime::TaskSpec task_;
    std::shared_ptr<const VibeASRAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer_;
    VibeASRVaeEncoderRuntime encoder_;
    VibeASRLmRuntime lm_;
};

}  // namespace engine::community_models::vibeasr
