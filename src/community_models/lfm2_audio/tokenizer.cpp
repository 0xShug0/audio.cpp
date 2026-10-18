#include "engine/community_models/lfm2_audio/tokenizer.h"

#include "bpe-core.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::community_models::lfm2_audio {
namespace {

namespace vendor = llama_tokenizer_vendor;

// tokenizer.ggml.token_type values (gguf-py TokenType,
// https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py).
constexpr int32_t kTokenTypeUnknown = 2;
constexpr int32_t kTokenTypeControl = 3;
constexpr int32_t kTokenTypeUserDefined = 4;
constexpr int32_t kTokenTypeUnused = 5;

uint32_t token_attr(int32_t type) {
    switch (type) {
        case kTokenTypeUnknown: return vendor::TOKEN_ATTR_UNKNOWN;
        case kTokenTypeControl: return vendor::TOKEN_ATTR_CONTROL;
        case kTokenTypeUserDefined: return vendor::TOKEN_ATTR_USER_DEFINED;
        default: return 0;
    }
}

vendor::PreTokenizerType pre_tokenizer_type(const std::string & name) {
    // llama.cpp writes "lfm2" for LFM2 checkpoints and maps it to its Llama 3
    // pre-tokenizer (llama_vocab::impl::load in src/llama-vocab.cpp), whose
    // regex the vendored BPE core shares. That regex is the Split pattern of the
    // checkpoint's tokenizer.json with the (?i:'s|'t|...) group spelled out per
    // case.
    if (name == "lfm2") {
        return vendor::PreTokenizerType::Llama3;
    }

    throw std::runtime_error("LFM2-Audio does not support tokenizer.ggml.pre = " + name);
}

std::shared_ptr<const vendor::BpeVocabulary> build_vocabulary(const Lfm2TextVocabulary & source) {
    auto vocab = std::make_shared<vendor::BpeVocabulary>();
    vocab->pre_type = pre_tokenizer_type(source.pre_tokenizer);
    for (size_t id = 0; id < source.tokens.size(); ++id) {
        const auto token_id = static_cast<int32_t>(id);
        vocab->token_to_id.emplace(source.tokens[id], token_id);
        vocab->id_to_token.emplace(token_id, vendor::TokenData{source.tokens[id], token_attr(source.token_types[id])});
    }

    for (size_t rank = 0; rank < source.merges.size(); ++rank) {
        const auto & merge = source.merges[rank];
        const auto split = merge.find(' ', 1);
        if (split == std::string::npos) {
            throw std::runtime_error("LFM2-Audio tokenizer merge has no separator: " + merge);
        }

        std::string key = merge.substr(0, split);
        key.push_back('\0');
        key += merge.substr(split + 1);
        vocab->bpe_ranks.emplace(std::move(key), static_cast<int32_t>(rank));
    }

    vendor::rebuild_special_tokens_cache(*vocab);
    return vocab;
}

// The well-formed UTF-8 byte sequences (Unicode Table 3-7) by the range of
// their first byte: the range of the second byte, and the length. Any further
// bytes are 80..BF.
struct Utf8Sequence {
    unsigned char first_low;
    unsigned char first_high;
    unsigned char second_low;
    unsigned char second_high;
    size_t length;
};

constexpr Utf8Sequence kWellFormedUtf8[] = {
    {0x00, 0x7F, 0x00, 0x00, 1}, {0xC2, 0xDF, 0x80, 0xBF, 2}, {0xE0, 0xE0, 0xA0, 0xBF, 3},
    {0xE1, 0xEC, 0x80, 0xBF, 3}, {0xED, 0xED, 0x80, 0x9F, 3}, {0xEE, 0xEF, 0x80, 0xBF, 3},
    {0xF0, 0xF0, 0x90, 0xBF, 4}, {0xF1, 0xF3, 0x80, 0xBF, 4}, {0xF4, 0xF4, 0x80, 0x8F, 4},
};

}  // namespace

Lfm2TextTokenizer::Lfm2TextTokenizer(const Lfm2TextVocabulary & vocabulary) : vocab_(build_vocabulary(vocabulary)) {
    dropped_in_decode_.reserve(vocabulary.token_types.size());
    for (const int32_t type : vocabulary.token_types) {
        dropped_in_decode_.push_back(type == kTokenTypeControl || type == kTokenTypeUnused);
    }
}

std::vector<int32_t> Lfm2TextTokenizer::encode(const std::string & text) const {
    return vendor::tokenize_bpe(*vocab_, text, true);
}

std::string Lfm2TextTokenizer::decode(const std::vector<int32_t> & token_ids) const {
    // decode_bpe's skip flag also drops user-defined tokens, and in this
    // vocabulary those are words ("python" and "Mathias" are added tokens with
    // special = false in the checkpoint's tokenizer.json), so the filtering is
    // done here. Control tokens are markup; unused ones are the [PADn] filler
    // that llama.cpp's converter (TextModel.get_vocab_base) writes past the
    // trained vocabulary. llama.cpp prints neither.
    std::vector<int32_t> text_ids;
    text_ids.reserve(token_ids.size());
    for (const int32_t token_id : token_ids) {
        const bool known = token_id >= 0 && static_cast<size_t>(token_id) < dropped_in_decode_.size();
        if (!known || !dropped_in_decode_[static_cast<size_t>(token_id)]) {
            text_ids.push_back(token_id);
        }
    }

    return vendor::decode_bpe(*vocab_, text_ids, false);
}

int32_t Lfm2TextTokenizer::require_token_id(const std::string & token) const {
    const auto it = vocab_->token_to_id.find(token);
    if (it == vocab_->token_to_id.end()) {
        throw std::runtime_error("LFM2-Audio tokenizer has no token " + token);
    }

    return it->second;
}

bool Lfm2TextTokenizer::is_control_token(int32_t token_id) const {
    const auto it = vocab_->id_to_token.find(token_id);
    return it != vocab_->id_to_token.end() && (it->second.attr & vendor::TOKEN_ATTR_CONTROL) != 0;
}

std::string lfm2_take_text(std::string & bytes) {
    std::string text;
    size_t pos = 0;
    while (pos < bytes.size()) {
        const auto first = static_cast<unsigned char>(bytes[pos]);
        const auto * sequence = std::find_if(std::begin(kWellFormedUtf8), std::end(kWellFormedUtf8),
            [&](const Utf8Sequence & row) { return first >= row.first_low && first <= row.first_high; });
        const size_t needed = sequence == std::end(kWellFormedUtf8) ? 0 : sequence->length;

        // How much of a well-formed sequence starts here: at least this byte,
        // which is the maximal subpart when no sequence starts with it.
        size_t length = 1;
        while (length < needed && pos + length < bytes.size()) {
            const auto byte = static_cast<unsigned char>(bytes[pos + length]);
            const bool fits = length == 1 ? byte >= sequence->second_low && byte <= sequence->second_high : (byte & 0xC0) == 0x80;
            if (!fits) {
                break;
            }

            ++length;
        }

        if (length == needed) {
            text.append(bytes, pos, length);
        } else if (needed != 0 && pos + length == bytes.size()) {
            break;  // the next bytes may finish it
        } else {
            text += "\xEF\xBF\xBD";
        }

        pos += length;
    }

    bytes.erase(0, pos);
    return text;
}

std::string Lfm2StreamedText::add(const std::string & bytes) {
    bytes_ += bytes;
    text_ += lfm2_take_text(bytes_);
    return partials_.publish(text_);
}

}  // namespace engine::community_models::lfm2_audio
