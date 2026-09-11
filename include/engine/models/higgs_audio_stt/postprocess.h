#pragma once

#include "engine/models/higgs_audio_stt/tokenizer_text.h"
#include "engine/models/higgs_audio_stt/types.h"

namespace engine::models::higgs_audio_stt {

class HiggsAudioSTTPostprocessor {
public:
    explicit HiggsAudioSTTPostprocessor(const HiggsAudioSTTTextTokenizer & tokenizer);

    HiggsAudioSTTResult decode(const HiggsAudioSTTGeneratedTokens & tokens, const HiggsAudioSTTRequest & request) const;

private:
    const HiggsAudioSTTTextTokenizer & tokenizer_;
};

}  // namespace engine::models::higgs_audio_stt
