#pragma once

#include "engine/community_models/vieneu_v3_turbo/speaker_encoder.h"
#include "engine/community_models/vieneu_v3_turbo/talker.h"
#include "engine/framework/codecs/moss_audio_tokenizer_codec_runtime.h"
#include "engine/community_models/vieneu_v3_turbo/tokenizer_text.h"
#include "engine/community_models/vieneu_v3_turbo/types.h"

#include <optional>

namespace engine::models::vieneu_v3_turbo {

struct Qwen3VoiceClonePrompt {
    VieNeuSpeakerEmbedding speaker_embedding;
    std::optional<Qwen3SpeechCodes> reference_codes = std::nullopt;
    std::vector<int32_t> reference_text_ids;
    bool icl_mode = true;
};

class VieNeuTTSVoiceClonePromptBuilder {
public:
    VieNeuTTSVoiceClonePromptBuilder(
        const Qwen3TextTokenizer & tokenizer,
        engine::codecs::MossAudioTokenizerCodecRuntime * codec,
        const VieNeuSpeakerEncoderRuntime * speaker_encoder,
        int64_t text_token_limit);

    // Reference codes exactly as the Python engine builds them (`_encode_ref_wav`):
    // mono downmix, resample to 48 kHz, cap at kMaxReferenceSeconds, zero-pad to whole
    // codec frames, duplicate to stereo, encode, keep the first n_padded / 3840 frames.
    static Qwen3SpeechCodes encode_reference_codes(
        engine::codecs::MossAudioTokenizerCodecRuntime & codec,
        const runtime::AudioBuffer & audio,
        int64_t code_groups);

    Qwen3VoiceClonePrompt build_voice_prompt(const Qwen3VoiceCloneInput & input) const;
    VieNeuTalkerPrefill build_prefill(const VieNeuTTSRequest & request, const Qwen3VoiceClonePrompt & prompt) const;

private:
    const Qwen3TextTokenizer & tokenizer_;
    engine::codecs::MossAudioTokenizerCodecRuntime * codec_ = nullptr;
    const VieNeuSpeakerEncoderRuntime * speaker_encoder_ = nullptr;
    int64_t text_token_limit_ = 0;
};

}  // namespace engine::models::vieneu_v3_turbo
