#pragma once

#include "engine/community_models/glm_tts/assets.h"
#include "engine/community_models/glm_tts/tokenizer_text.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::glm_tts {

struct GlmTTSPrompt {
    std::vector<int32_t> input_ids;
    int64_t text_token_count = 0;
    int64_t minimum_audio_tokens = 0;
    int64_t maximum_audio_tokens = 0;
};

GlmTTSPrompt build_glm_tts_prompt(
    const GlmTTSConfig & config,
    const GlmTTSTextTokenizer & tokenizer,
    const std::string & reference_text,
    const std::string & text,
    const std::vector<int32_t> & reference_speech_tokens);

}  // namespace engine::models::glm_tts
