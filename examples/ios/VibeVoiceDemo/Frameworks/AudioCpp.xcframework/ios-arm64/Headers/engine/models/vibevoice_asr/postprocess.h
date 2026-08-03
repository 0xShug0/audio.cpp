#pragma once

#include "engine/models/vibevoice_asr/tokenizer_text.h"
#include "engine/models/vibevoice_asr/types.h"

namespace engine::models::vibevoice_asr {

class VibeVoiceASRPostprocessor {
public:
    explicit VibeVoiceASRPostprocessor(const VibeVoiceASRTextTokenizer & tokenizer);

    VibeVoiceASRDecoded decode(const VibeVoiceASRGeneratedTokens & tokens) const;

private:
    const VibeVoiceASRTextTokenizer & tokenizer_;
};

}  // namespace engine::models::vibevoice_asr
