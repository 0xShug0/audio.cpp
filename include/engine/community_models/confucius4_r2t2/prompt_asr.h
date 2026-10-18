#pragma once

#include "engine/community_models/confucius4_r2t2/tokenizer_text.h"
#include "engine/community_models/confucius4_r2t2/types.h"

namespace engine::community_models::confucius4_r2t2 {

class R2T2ASRPromptBuilder {
public:
    explicit R2T2ASRPromptBuilder(const R2T2ASRTextTokenizer & tokenizer);

    R2T2ASRPrompt build(const R2T2ASRRequest & request, int64_t audio_feature_tokens) const;

private:
    const R2T2ASRTextTokenizer & tokenizer_;
};

}  // namespace engine::community_models::confucius4_r2t2
