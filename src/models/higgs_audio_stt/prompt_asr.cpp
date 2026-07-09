#include "engine/models/higgs_audio_stt/prompt_asr.h"

namespace engine::models::higgs_audio_stt {

HiggsAudioSTTPromptBuilder::HiggsAudioSTTPromptBuilder(const HiggsAudioSTTTextTokenizer & tokenizer)
    : tokenizer_(tokenizer) {}

HiggsAudioSTTPrompt HiggsAudioSTTPromptBuilder::build(const HiggsAudioSTTRequest & request, int64_t audio_feature_tokens) const {
    return tokenizer_.build_prompt(
        request.context,
        request.language,
        request.generation.enable_thinking,
        audio_feature_tokens);
}

}  // namespace engine::models::higgs_audio_stt
