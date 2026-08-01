#pragma once

#include "engine/models/higgs_audio_stt/tokenizer_text.h"
#include "engine/models/higgs_audio_stt/types.h"

namespace engine::models::higgs_audio_stt {

class HiggsAudioSTTPromptBuilder {
public:
    explicit HiggsAudioSTTPromptBuilder(const HiggsAudioSTTTextTokenizer & tokenizer);

    HiggsAudioSTTPrompt build(const HiggsAudioSTTRequest & request, int64_t audio_feature_tokens) const;

private:
    const HiggsAudioSTTTextTokenizer & tokenizer_;
};

}  // namespace engine::models::higgs_audio_stt
