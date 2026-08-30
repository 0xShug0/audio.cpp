#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/models/vibevoice/assets.h"
#include "engine/models/vibevoice/connector.h"
#include "engine/models/vibevoice/decoder.h"
#include "engine/models/vibevoice/diffusion_head.h"
#include "engine/models/vibevoice/generator.h"
#include "engine/models/vibevoice/tokenizer_audio.h"
#include "engine/models/vibevoice/tokenizer_text.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::vibevoice {

class VibeVoiceSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    VibeVoiceSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const VibeVoiceAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    struct ReferenceVoiceStateCacheEntry {
        std::vector<std::string> sample_paths;
        std::vector<runtime::AudioBuffer> audio;
        std::vector<VibeVoiceReferenceVoiceState> states;
    };

    VibeVoiceRequest make_request(const runtime::TaskRequest & request) const;
    std::vector<VibeVoiceSpeakerPrompt> resolve_voice_sample_prompts(
        const std::vector<std::string> & sample_paths) const;

    runtime::TaskSpec task_;
    std::shared_ptr<const VibeVoiceAssets> assets_;
    // Reference voice prompt cap in seconds; 0 means the whole reference.
    // Defaults to 30 on CUDA/HIP and 10 elsewhere, overridable with
    // `vibevoice.voice_prompt_max_seconds`.
    int64_t voice_prompt_max_seconds_ = 10;
    VibeVoiceTextTokenizer text_tokenizer_;
    VibeVoiceTokenizerWeightsRuntime audio_tokenizer_;
    VibeVoiceConnectorWeightsRuntime connector_;
    VibeVoiceDecoderWeightsRuntime decoder_;
    VibeVoiceDiffusionHeadWeightsRuntime diffusion_head_;
    VibeVoiceDecoderCachedState positive_decoder_cache_;
    VibeVoiceDecoderCachedState negative_decoder_cache_;
    mutable std::vector<ReferenceVoiceStateCacheEntry> reference_voice_state_cache_;
};

}  // namespace engine::models::vibevoice
