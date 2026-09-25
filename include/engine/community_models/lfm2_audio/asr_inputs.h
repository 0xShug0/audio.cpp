#pragma once

// What the ASR session feeds LFM2.5-Audio: 16 kHz mono audio, and the chat
// prompt liquid-audio's ChatState (processor.py) builds around the audio.

#include "engine/community_models/lfm2_audio/backbone.h"
#include "engine/community_models/lfm2_audio/tokenizer.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::community_models::lfm2_audio {

// Averages the channels and resamples to 16 kHz the way ChatState.add_audio
// does (torchaudio.functional.resample). Throws on empty or non-finite audio.
std::vector<float> lfm2_audio_mono_16k(const runtime::AudioBuffer & audio);

// <|startoftext|><|im_start|>system\n{system prompt}<|im_end|>\n
// <|im_start|>user\n{audio}<|im_end|>\n<|im_start|>assistant\n
struct Lfm2AsrPrompt {
    std::vector<int32_t> prefix;          // before the audio
    std::vector<int32_t> suffix;          // after it
    std::vector<int32_t> stop_token_ids;  // <|im_end|> and <|audio_start|>

    // The prompt with `audio_tokens` audio positions between prefix and suffix.
    [[nodiscard]] Lfm2Prompt with_audio(int64_t audio_tokens) const;
};

// The ASR prompt of an "en" or "ja" checkpoint.
Lfm2AsrPrompt make_lfm2_asr_prompt(const Lfm2TextTokenizer & tokenizer, const std::string & language);

}  // namespace engine::community_models::lfm2_audio
