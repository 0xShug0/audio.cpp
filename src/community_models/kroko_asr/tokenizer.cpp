#include "engine/community_models/kroko_asr/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::models::kroko_asr {
namespace {

constexpr const char * kSentencePieceSpace = "\xE2\x96\x81";

bool starts_with_sentencepiece_space(const std::string & value) {
    return value.rfind(kSentencePieceSpace, 0) == 0;
}

std::string normalized_hotword(const std::string & text) {
    std::string normalized;
    normalized.reserve(text.size() + 3);
    bool pending_space = true;
    for (size_t index = 0; index < text.size();) {
        const unsigned char character =
            static_cast<unsigned char>(text[index]);
        if (std::isspace(character) != 0) {
            pending_space = true;
            ++index;
            continue;
        }
        if (pending_space &&
            (normalized.empty() ||
             !starts_with_sentencepiece_space(
                 text.substr(index)))) {
            normalized.append(kSentencePieceSpace);
        }
        pending_space = false;
        size_t bytes = 1;
        if ((character & 0xE0U) == 0xC0U) {
            bytes = 2;
        } else if ((character & 0xF0U) == 0xE0U) {
            bytes = 3;
        } else if ((character & 0xF8U) == 0xF0U) {
            bytes = 4;
        }
        bytes = std::min(bytes, text.size() - index);
        normalized.append(text, index, bytes);
        index += bytes;
    }
    return normalized;
}

}  // namespace

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

std::vector<int32_t> KrokoTokenizer::encode_hotword(
    const std::string & text) const {
    const std::string normalized = normalized_hotword(text);
    if (normalized.empty()) {
        throw std::runtime_error(
            "Kroko hotword cannot be empty");
    }
    constexpr int kUnreachable =
        std::numeric_limits<int>::max() / 4;
    std::vector<int> cost(normalized.size() + 1, kUnreachable);
    std::vector<int32_t> previous_token(
        normalized.size() + 1, -1);
    std::vector<size_t> previous_offset(
        normalized.size() + 1, 0);
    cost.front() = 0;
    for (size_t offset = 0;
         offset < normalized.size();
         ++offset) {
        if (cost[offset] == kUnreachable) {
            continue;
        }
        for (int32_t id = 0;
             id < static_cast<int32_t>(pieces_.size());
             ++id) {
            if (id == blank_id_ || id == unk_id_ || id == 1) {
                continue;
            }
            const std::string & candidate =
                pieces_[static_cast<size_t>(id)];
            if (candidate.empty() ||
                normalized.compare(
                    offset,
                    candidate.size(),
                    candidate) != 0) {
                continue;
            }
            const size_t next = offset + candidate.size();
            const int candidate_cost = cost[offset] + 1;
            const size_t previous_length =
                previous_token[next] < 0
                ? 0
                : pieces_[static_cast<size_t>(
                      previous_token[next])]
                      .size();
            if (candidate_cost < cost[next] ||
                (candidate_cost == cost[next] &&
                 candidate.size() > previous_length)) {
                cost[next] = candidate_cost;
                previous_token[next] = id;
                previous_offset[next] = offset;
            }
        }
    }
    if (cost.back() == kUnreachable) {
        throw std::runtime_error(
            "Kroko hotword cannot be represented by this package vocabulary: " +
            text);
    }
    std::vector<int32_t> reversed;
    for (size_t offset = normalized.size();
         offset > 0;) {
        const int32_t token = previous_token[offset];
        if (token < 0) {
            throw std::runtime_error(
                "Kroko hotword tokenizer reconstruction failed");
        }
        reversed.push_back(token);
        offset = previous_offset[offset];
    }
    return std::vector<int32_t>(
        reversed.rbegin(), reversed.rend());
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
