#include "engine/community_models/r2t2_asr/prompt_asr.h"

namespace engine::community_models::r2t2_asr {

R2T2ASRPromptBuilder::R2T2ASRPromptBuilder(const R2T2ASRTextTokenizer & tokenizer)
    : tokenizer_(tokenizer) {}

R2T2ASRPrompt R2T2ASRPromptBuilder::build(const R2T2ASRRequest & request, int64_t audio_feature_tokens) const {
    return tokenizer_.build_prompt(request.context, request.language, audio_feature_tokens);
}

}  // namespace engine::community_models::r2t2_asr
