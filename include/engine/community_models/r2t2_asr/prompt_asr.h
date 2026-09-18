#pragma once

#include "engine/community_models/r2t2_asr/tokenizer_text.h"
#include "engine/community_models/r2t2_asr/types.h"

namespace engine::community_models::r2t2_asr {

class R2T2ASRPromptBuilder {
public:
    explicit R2T2ASRPromptBuilder(const R2T2ASRTextTokenizer & tokenizer);

    R2T2ASRPrompt build(const R2T2ASRRequest & request, int64_t audio_feature_tokens) const;

private:
    const R2T2ASRTextTokenizer & tokenizer_;
};

}  // namespace engine::community_models::r2t2_asr
