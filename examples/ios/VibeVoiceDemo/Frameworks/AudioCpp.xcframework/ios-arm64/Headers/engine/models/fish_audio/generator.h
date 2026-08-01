#pragma once

#include "engine/models/fish_audio/ar.h"
#include "engine/models/fish_audio/codec.h"
#include "engine/models/fish_audio/prompt_builder.h"
#include "engine/models/fish_audio/tokenizer_text.h"

#include <memory>
#include <optional>

namespace engine::models::fish_audio {

struct FishAudioGenerationResult {
    runtime::AudioBuffer audio;
    FishAudioCodes codes;
};

class FishAudioGenerator {
public:
    FishAudioGenerator(
        std::shared_ptr<const FishAudioAssets> assets,
        std::unique_ptr<FishAudioARRuntime> ar,
        std::unique_ptr<FishAudioCodecRuntime> codec);
    ~FishAudioGenerator();

    FishAudioCodes encode_reference(const runtime::AudioBuffer & audio);
    FishAudioGenerationResult generate(
        const FishAudioRequest & request,
        const std::optional<FishAudioCodes> & reference_codes,
        const std::optional<FishAudioConversationTurn> & previous_turn,
        bool mem_saver);

private:
    std::shared_ptr<const FishAudioAssets> assets_;
    FishAudioTextTokenizer tokenizer_;
    FishAudioPromptBuilder prompt_builder_;
    std::unique_ptr<FishAudioARRuntime> ar_;
    std::unique_ptr<FishAudioCodecRuntime> codec_;
};

}  // namespace engine::models::fish_audio
