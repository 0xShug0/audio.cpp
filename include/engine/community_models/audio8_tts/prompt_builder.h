#pragma once

#include "engine/community_models/audio8_tts/tokenizer_text.h"
#include "engine/community_models/audio8_tts/types.h"

namespace engine::models::audio8_tts {

class Audio8TtsPromptBuilder {
public:
    Audio8TtsPromptBuilder(std::shared_ptr<const Audio8TtsAssets> assets, Audio8TtsTextTokenizer tokenizer);

    Audio8TtsPrompt build(
        const Audio8TtsRequest & request,
        const std::vector<Audio8TtsCodes> & reference_codes,
        const std::optional<Audio8TtsConversationTurn> & previous_turn) const;

private:
    std::shared_ptr<const Audio8TtsAssets> assets_;
    Audio8TtsTextTokenizer tokenizer_;
};

}  // namespace engine::models::audio8_tts
