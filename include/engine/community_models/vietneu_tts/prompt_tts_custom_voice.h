#pragma once

#include "engine/community_models/vietneu_tts/talker.h"
#include "engine/community_models/vietneu_tts/tokenizer_text.h"
#include "engine/community_models/vietneu_tts/types.h"

namespace engine::models::vietneu_tts {

class VietneuTTSCustomVoicePromptBuilder {
public:
    VietneuTTSCustomVoicePromptBuilder(
        const Qwen3TextTokenizer & tokenizer,
        int64_t text_token_limit,
        int64_t instruction_token_limit);

    VietneuTalkerPrefill build_prefill(const VietneuTTSRequest & request) const;

private:
    const Qwen3TextTokenizer & tokenizer_;
    int64_t text_token_limit_ = 0;
    int64_t instruction_token_limit_ = 0;
};

}  // namespace engine::models::vietneu_tts
