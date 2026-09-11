#pragma once

#include "engine/community_models/vietneu_tts/speaker_encoder.h"
#include "engine/community_models/vietneu_tts/talker.h"
#include "engine/community_models/vietneu_tts/tokenizer_speech_encoder.h"
#include "engine/community_models/vietneu_tts/tokenizer_text.h"
#include "engine/community_models/vietneu_tts/types.h"

#include <optional>

namespace engine::models::vietneu_tts {

struct Qwen3VoiceClonePrompt {
    VietneuSpeakerEmbedding speaker_embedding;
    std::optional<Qwen3SpeechCodes> reference_codes = std::nullopt;
    std::vector<int32_t> reference_text_ids;
    bool icl_mode = true;
};

class VietneuTTSVoiceClonePromptBuilder {
public:
    VietneuTTSVoiceClonePromptBuilder(
        const Qwen3TextTokenizer & tokenizer,
        const Qwen3SpeechTokenizerEncoderRuntime * speech_encoder,
        const VietneuSpeakerEncoderRuntime * speaker_encoder,
        int64_t text_token_limit);

    Qwen3VoiceClonePrompt build_voice_prompt(const Qwen3VoiceCloneInput & input) const;
    VietneuTalkerPrefill build_prefill(const VietneuTTSRequest & request, const Qwen3VoiceClonePrompt & prompt) const;

private:
    const Qwen3TextTokenizer & tokenizer_;
    const Qwen3SpeechTokenizerEncoderRuntime * speech_encoder_ = nullptr;
    const VietneuSpeakerEncoderRuntime * speaker_encoder_ = nullptr;
    int64_t text_token_limit_ = 0;
};

}  // namespace engine::models::vietneu_tts
