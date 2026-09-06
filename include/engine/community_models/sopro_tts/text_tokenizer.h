#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::sopro_tts {

// Mirrors sopro/text.py. The reference pipeline is deliberately minimal: no
// grapheme-to-phoneme stage, just light punctuation clean-up, an optional
// language tag and a SentencePiece unigram model with 8192 pieces.

// sopro.text.split_text: sentence -> clause -> word packing, codepoint budget.
std::vector<std::string> split_text(const std::string & text, int64_t max_chars);

// sopro.text.normalize_text.
std::string normalize_text(const std::string & text);

// sopro.text.language_tag; throws for anything outside {en, pt, fr, de}.
// An empty language yields an empty tag.
std::string language_tag(const std::string & language);

class SoproTextTokenizer {
public:
    explicit SoproTextTokenizer(const std::filesystem::path & model_path, int64_t max_length = 512);
    ~SoproTextTokenizer();

    SoproTextTokenizer(const SoproTextTokenizer &) = delete;
    SoproTextTokenizer & operator=(const SoproTextTokenizer &) = delete;

    // [bos] + pieces(normalize_text(tag + text)) + [eos], truncated to
    // max_length; never empty (falls back to the unk id).
    std::vector<int32_t> encode(const std::string & text, const std::string & language) const;

    int32_t bos_id() const noexcept;
    int32_t eos_id() const noexcept;
    int32_t unk_id() const noexcept;
    int64_t vocab_size() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::sopro_tts
