#include "engine/community_models/kroko_asr/tokenizer.h"

#include <cctype>
#include <stdexcept>
#include <utility>

namespace engine::models::kroko_asr {

KrokoTokenizer::KrokoTokenizer(
    std::vector<std::string> pieces,
    int32_t blank_id,
    int32_t unk_id)
    : pieces_(std::move(pieces)),
      blank_id_(blank_id),
      unk_id_(unk_id) {
    if (pieces_.empty() || blank_id_ < 0 ||
        blank_id_ >= static_cast<int32_t>(pieces_.size())) {
        throw std::runtime_error("Kroko tokenizer configuration is invalid");
    }
}

std::string KrokoTokenizer::decode(const std::vector<int32_t> & ids) const {
    std::string text;
    for (const int32_t id : ids) {
        if (id == blank_id_ || id == 1) {
            continue;
        }
        if (id < 0 || id >= static_cast<int32_t>(pieces_.size())) {
            continue;
        }
        const std::string & piece = pieces_[static_cast<size_t>(id)];
        if (piece == "<unk>" || id == unk_id_) {
            text.append("\xEF\xBF\xBD");
            continue;
        }
        for (size_t index = 0; index < piece.size();) {
            if (index + 3 <= piece.size() &&
                static_cast<unsigned char>(piece[index]) == 0xE2 &&
                static_cast<unsigned char>(piece[index + 1]) == 0x96 &&
                static_cast<unsigned char>(piece[index + 2]) == 0x81) {
                if (!text.empty() && text.back() != ' ') {
                    text.push_back(' ');
                }
                index += 3;
            } else {
                text.push_back(piece[index++]);
            }
        }
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0) {
        text.erase(text.begin());
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0) {
        text.pop_back();
    }
    return text;
}

const std::string & KrokoTokenizer::piece(int32_t id) const {
    if (id < 0 || id >= static_cast<int32_t>(pieces_.size())) {
        throw std::runtime_error("Kroko token id is outside the vocabulary");
    }
    return pieces_[static_cast<size_t>(id)];
}

int32_t KrokoTokenizer::blank_id() const noexcept {
    return blank_id_;
}

int32_t KrokoTokenizer::vocab_size() const noexcept {
    return static_cast<int32_t>(pieces_.size());
}

}  // namespace engine::models::kroko_asr
