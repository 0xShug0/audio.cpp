#pragma once

#include "engine/framework/tokenizers/sentencepiece.h"
#include "engine/models/index_tts2/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llama_tokenizer_vendor {
struct BpeVocabulary;
}  // namespace llama_tokenizer_vendor

namespace engine::models::index_tts2 {

struct IndexTTS2TextEncoding {
    // v2.5: resolved language code; empty for v2.
    std::string lang;
    std::string normalized_text;
    // v2 SentencePiece pieces; empty for v2.5.
    std::vector<std::string> pieces;
    std::vector<int32_t> token_ids;
    // v2: piece strings per segment; v2.5: processed text per segment.
    std::vector<std::vector<std::string>> segments;
    std::vector<std::vector<int32_t>> segment_token_ids;
};

// Variant-aware IndexTTS2 text tokenizer. v2 keeps the SentencePiece bpe.model
// behavior; v2.5 uses the multilingual tiktoken/BPE vocabulary with language
// special tokens and language-id handling. The variant is selected from the
// model config version, not from probing tokenizer files.
class IndexTTS2TextTokenizer {
public:
    explicit IndexTTS2TextTokenizer(std::shared_ptr<const IndexTTS2Assets> assets);

    IndexTTS2Variant variant() const noexcept {
        return variant_;
    }

    std::string normalize_english(const std::string & text) const;
    std::string normalize_chinese(const std::string & text) const;

    // v2: SentencePiece encode of the normalized text.
    // v2.5: raw tiktoken encode with allowed_special="all"; does not apply any
    // text normalization. Special tokens present in the text are recognized
    // directly.
    std::vector<int32_t> encode(const std::string & text) const;

    // v2 only: normalize then tokenize helpers kept for parity/debug.
    std::string normalize_text(const std::string & text) const;
    std::vector<std::string> tokenize_to_pieces(const std::string & text) const;

    // v2.5 only: returns the id of an exact token text (e.g. "<|zh|>"), or -1
    // when unknown.
    int32_t special_token_id(const std::string & token_text) const;

    // v2.5 only: maps a language code to the GPT lang_embedding row, following
    // the LANGUAGES order of indextts/utils/tokenizer.py (en=0, zh=1, ...).
    // Unknown codes map to "common".
    static int32_t lang_to_id(const std::string & lang);

    // v2: normalize -> SentencePiece encode -> segment pieces by token budget.
    // v2.5: normalize -> case rules -> pronunciation annotations ->
    // special-token name uppercasing -> segment by token budget. Each segment
    // is encoded as encode("<|{lang}|> " + segment) plus a trailing pad token
    // id 1. When lang is empty, it is inferred (Han -> zh, else en).
    IndexTTS2TextEncoding encode_for_inference(
        const std::string & text,
        int max_text_tokens_per_segment,
        const std::string & lang = "") const;

private:
    // v2 SentencePiece path.
    IndexTTS2TextEncoding encode_for_inference_v2(
        const std::string & text,
        int max_text_tokens_per_segment) const;
    int32_t piece_to_id(const std::string & piece) const;
    std::string id_to_piece(int32_t id) const;
    std::vector<std::vector<std::string>> split_segments(
        const std::vector<std::string> & pieces,
        int max_text_tokens_per_segment) const;

    // v2.5 tiktoken path.
    IndexTTS2TextEncoding encode_for_inference_v2_5(
        const std::string & text,
        int max_text_tokens_per_segment,
        const std::string & lang) const;

    std::shared_ptr<const IndexTTS2Assets> assets_;
    IndexTTS2Variant variant_ = IndexTTS2Variant::kV2;
    // v2 SentencePiece model.
    std::vector<engine::tokenizers::SentencePiecePiece> pieces_;
    std::unordered_map<std::string, int32_t> piece_to_id_;
    // v2.5 tiktoken vocabulary.
    std::shared_ptr<llama_tokenizer_vendor::BpeVocabulary> vocab_;
};

}  // namespace engine::models::index_tts2
