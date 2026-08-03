#pragma once

#include "engine/framework/tokenizers/sentencepiece.h"
#include "engine/models/index_tts2/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::index_tts2 {

struct IndexTTS2TextEncoding {
    std::string normalized_text;
    std::vector<std::string> pieces;
    std::vector<int32_t> token_ids;
    std::vector<std::vector<std::string>> segments;
    std::vector<std::vector<int32_t>> segment_token_ids;
};

class IndexTTS2TextTokenizer {
public:
    explicit IndexTTS2TextTokenizer(std::shared_ptr<const IndexTTS2Assets> assets);

    std::string normalize_english(const std::string & text) const;
    std::string normalize_chinese(const std::string & text) const;
    std::string normalize_text(const std::string & text) const;
    std::vector<int32_t> encode(const std::string & text) const;
    std::vector<std::string> tokenize_to_pieces(const std::string & text) const;
    IndexTTS2TextEncoding encode_for_inference(
        const std::string & text,
        int max_text_tokens_per_segment) const;

private:
    int32_t piece_to_id(const std::string & piece) const;
    std::string id_to_piece(int32_t id) const;
    std::vector<std::vector<std::string>> split_segments(
        const std::vector<std::string> & pieces,
        int max_text_tokens_per_segment) const;

    std::shared_ptr<const IndexTTS2Assets> assets_;
    std::vector<engine::tokenizers::SentencePiecePiece> pieces_;
    std::unordered_map<std::string, int32_t> piece_to_id_;
};

}  // namespace engine::models::index_tts2
