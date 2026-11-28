#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::nemotron_transcribe {

struct PromptTokens {
    std::vector<int32_t> before_audio;
    std::vector<int32_t> after_audio;
};

class TextTokenizer {
public:
    TextTokenizer(const std::filesystem::path & tokenizer_json_path,
        const std::filesystem::path & tokenizer_config_path);

    // No BOS is added. parse_special controls recognition of chat/control
    // tokens; transcript tags retain their checkpoint tokenization.
    std::vector<int32_t> encode(const std::string & text, bool parse_special = true) const;
    std::string decode(const std::vector<int32_t> & ids, bool skip_special_tokens = false) const;

    // An audio locator is represented by the gap between these vectors. Its
    // ID 131072 never enters the 131072-row language-model embedding table.
    // Instruction and system text may name transcript tags, but cannot inject
    // chat controls or an audio locator. Text is otherwise preserved verbatim.
    PromptTokens build_prompt(const std::string & instruction, const std::string & system_prompt = {}) const;
    std::string special_token_in(const std::string & text) const;
    const std::vector<std::string> & alignment_special_tokens() const noexcept;

    static constexpr int32_t end_of_turn_id = 11;
    static constexpr int32_t audio_locator_id = 131072;
    static constexpr int32_t embedding_rows = 131072;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::nemotron_transcribe
