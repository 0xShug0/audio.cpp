#pragma once

#include "engine/models/index_tts2_5/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llama_tokenizer_vendor {
struct BpeVocabulary;
}  // namespace llama_tokenizer_vendor

namespace engine::models::index_tts2_5 {

struct IndexTTS25TextEncoding {
    std::string lang;
    std::string normalized_text;
    std::vector<std::string> segments;
    std::vector<std::vector<int32_t>> segment_token_ids;
};

// Whisper-style tiktoken BPE text tokenizer for IndexTTS-2.5 (vocab size 60509).
// Replaces the SentencePiece tokenizer used by index_tts2.
class IndexTTS25TextTokenizer {
public:
    explicit IndexTTS25TextTokenizer(std::shared_ptr<const IndexTTS25Assets> assets);

    std::string normalize_english(const std::string & text) const;
    std::string normalize_chinese(const std::string & text) const;

    // Raw tiktoken encode with allowed_special="all"; does not apply any text
    // normalization. Special tokens present in the text are recognized directly.
    std::vector<int32_t> encode(const std::string & text) const;

    // Returns the id of an exact token text (e.g. "<|zh|>"), or -1 when unknown.
    int32_t special_token_id(const std::string & token_text) const;

    // Maps a language code to the GPT lang_embedding row, following the
    // LANGUAGES order of indextts/utils/tokenizer.py (en=0, zh=1, ...).
    // Unknown codes map to "common".
    static int32_t lang_to_id(const std::string & lang);

    // Full inference pipeline: normalize -> case rules -> pronunciation
    // annotations -> special-token name uppercasing -> segment by token budget.
    // Each segment is encoded as encode("<|{lang}|> " + segment) plus a trailing
    // pad token id 1. When lang is empty, it is inferred (Han -> zh, else en).
    IndexTTS25TextEncoding encode_for_inference(
        const std::string & text,
        int max_text_tokens_per_segment,
        const std::string & lang = "") const;

private:
    std::shared_ptr<const IndexTTS25Assets> assets_;
    std::shared_ptr<llama_tokenizer_vendor::BpeVocabulary> vocab_;
};

}  // namespace engine::models::index_tts2_5
