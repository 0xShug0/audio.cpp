#pragma once

#include "engine/models/fireredtts3/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace engine::models::fireredtts3 {

struct FireRedTTS3InstructTokens {
    std::vector<int32_t> token_ids;
    std::vector<uint8_t> latent_in_mask;
    std::vector<uint8_t> latent_out_mask;
};

class FireRedTTS3TextTokenizer {
public:
    explicit FireRedTTS3TextTokenizer(std::shared_ptr<const FireRedTTS3Assets> assets);
    ~FireRedTTS3TextTokenizer();

    FireRedTTS3TextTokenizer(const FireRedTTS3TextTokenizer &) = delete;
    FireRedTTS3TextTokenizer & operator=(const FireRedTTS3TextTokenizer &) = delete;

    std::string build_base_prompt(
        std::string_view language,
        std::string_view reference_text,
        std::string_view text) const;
    std::vector<int32_t> encode_base_prompt(
        std::string_view language,
        std::string_view reference_text,
        std::string_view text) const;
    FireRedTTS3InstructTokens encode_instruct_clone_prompt(
        int64_t prompt_latent_patches,
        std::string_view reference_text,
        std::string_view text) const;
    FireRedTTS3InstructTokens encode_voice_design_prompt(
        std::string_view instruction,
        std::string_view text) const;
    FireRedTTS3InstructTokens encode_semantic_edit_prompt(
        int64_t input_latent_patches,
        std::string_view instruction) const;
    FireRedTTS3InstructTokens encode_acoustic_edit_prompt(
        int64_t input_latent_patches,
        std::string_view instruction) const;
    std::string decode(const std::vector<int32_t> & token_ids, bool skip_special_tokens = false) const;

private:
    class Impl;
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::fireredtts3
