#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::glm_tts {

class GlmTTSTextTokenizer {
public:
    GlmTTSTextTokenizer(
        const std::filesystem::path & vocab_path,
        const std::filesystem::path & merges_path,
        const std::filesystem::path & tokenizer_config_path);
    ~GlmTTSTextTokenizer();

    GlmTTSTextTokenizer(const GlmTTSTextTokenizer &) = default;
    GlmTTSTextTokenizer & operator=(const GlmTTSTextTokenizer &) = default;

    std::vector<int32_t> encode(const std::string & text) const;
    int32_t require_token_id(const std::string & token) const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::glm_tts
