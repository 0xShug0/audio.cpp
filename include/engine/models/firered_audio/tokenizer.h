#pragma once

#include "engine/models/firered_audio/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace engine::models::firered_audio {

struct FireRedAudioPromptEncoding {
    std::vector<int32_t> token_ids;
    std::vector<uint8_t> audio_mask;
    std::vector<uint8_t> audio_no_latent_mask;
    int64_t generation_audio_patches = 0;
    int64_t understanding_audio_tokens = 0;
};

class FireRedAudioTokenizer {
public:
    explicit FireRedAudioTokenizer(std::shared_ptr<const FireRedAudioAssets> assets);
    ~FireRedAudioTokenizer();

    FireRedAudioTokenizer(const FireRedAudioTokenizer &) = delete;
    FireRedAudioTokenizer & operator=(const FireRedAudioTokenizer &) = delete;

    FireRedAudioPromptEncoding encode_tts_clone_prompt(
        std::string_view reference_text,
        std::string_view text,
        std::string_view language,
        int64_t reference_audio_patches) const;
    FireRedAudioPromptEncoding encode_voice_design_prompt(
        std::string_view instruction,
        std::string_view text) const;
    FireRedAudioPromptEncoding encode_edit_prompt(
        std::string_view instruction,
        std::string_view edit_type,
        int64_t reference_audio_patches) const;
    FireRedAudioPromptEncoding encode_understanding_prompt(
        std::string_view prompt,
        int64_t audio_tokens,
        bool enable_thinking) const;
    std::string decode(const std::vector<int32_t> & token_ids, bool skip_special_tokens = false) const;

private:
    class Impl;
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::firered_audio
