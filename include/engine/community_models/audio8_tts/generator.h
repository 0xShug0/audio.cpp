#pragma once

#include "engine/community_models/audio8_tts/ar.h"
#include "engine/community_models/audio8_tts/codec.h"
#include "engine/community_models/audio8_tts/prompt_builder.h"
#include "engine/community_models/audio8_tts/tokenizer_text.h"

#include <memory>
#include <optional>

namespace engine::models::audio8_tts {

struct Audio8TtsGenerationResult {
    runtime::AudioBuffer audio;
    Audio8TtsCodes codes;
};

class Audio8TtsGenerator {
public:
    Audio8TtsGenerator(
        std::shared_ptr<const Audio8TtsAssets> assets,
        std::unique_ptr<Audio8TtsARRuntime> ar,
        std::unique_ptr<Audio8TtsCodecRuntime> codec);
    ~Audio8TtsGenerator();

    Audio8TtsCodes encode_reference(const runtime::AudioBuffer & audio);
    Audio8TtsGenerationResult generate(
        const Audio8TtsRequest & request,
        const std::vector<Audio8TtsCodes> & reference_codes,
        const std::optional<Audio8TtsConversationTurn> & previous_turn,
        bool mem_saver);

private:
    std::shared_ptr<const Audio8TtsAssets> assets_;
    Audio8TtsTextTokenizer tokenizer_;
    Audio8TtsPromptBuilder prompt_builder_;
    std::unique_ptr<Audio8TtsARRuntime> ar_;
    std::unique_ptr<Audio8TtsCodecRuntime> codec_;
};

}  // namespace engine::models::audio8_tts
