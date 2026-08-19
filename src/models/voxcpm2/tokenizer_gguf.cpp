#include "engine/models/voxcpm2/tokenizer_gguf.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/models/voxcpm2/gguf_metadata.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::voxcpm2 {
namespace {

// UTF-8 handling functions (copied from tokenizer_text.cpp)
uint32_t next_utf8_codepoint(std::string_view text, size_t & offset) {
    if (offset >= text.size()) {
        throw std::runtime_error("VoxCPM2 tokenizer UTF-8 offset is out of range");
    }
    const unsigned char first = static_cast<unsigned char>(text[offset]);
    uint32_t codepoint = 0;
    size_t len = 1;
    if ((first & 0x80U) == 0) {
        codepoint = first;
    } else if ((first & 0xE0U) == 0xC0U) {
        len = 2;
        codepoint = first & 0x1FU;
    } else if ((first & 0xF0U) == 0xE0U) {
        len = 3;
        codepoint = first & 0x0FU;
    } else if ((first & 0xF8U) == 0xF0U) {
        len = 4;
        codepoint = first & 0x07U;
    } else {
        throw std::runtime_error("VoxCPM2 tokenizer encountered invalid UTF-8");
    }
    if (offset + len > text.size()) {
        throw std::runtime_error("VoxCPM2 tokenizer encountered truncated UTF-8");
    }
    for (size_t i = 1; i < len; ++i) {
        const unsigned char ch = static_cast<unsigned char>(text[offset + i]);
        if ((ch & 0xC0U) != 0x80U) {
            throw std::runtime_error("VoxCPM2 tokenizer encountered invalid UTF-8 continuation");
        }
        codepoint = (codepoint << 6U) | (ch & 0x3FU);
    }
    offset += len;
    return codepoint;
}

std::vector<std::string> utf8_codepoints(std::string_view text) {
    std::vector<std::string> out;
    for (size_t offset = 0; offset < text.size();) {
        const size_t start = offset;
        (void) next_utf8_codepoint(text, offset);
        out.emplace_back(text.substr(start, offset - start));
    }
    return out;
}

std::string normalize_text(std::string_view text) {
    const std::string space = "\xE2\x96\x81";
    std::string out = space;
    for (char ch : text) {
        if (ch == ' ') {
            out += space;
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

std::string byte_fallback_token(unsigned char byte) {
    constexpr char kHex[] = "0123456789ABCDEF";
    std::string out = "<0x";
    out.push_back(kHex[(byte >> 4U) & 0x0FU]);
    out.push_back(kHex[byte & 0x0FU]);
    out.push_back('>');
    return out;
}

std::vector<std::string> bpe_initial_pieces(
    std::string_view normalized_text,
    const std::unordered_map<std::string, int32_t> & vocab) {
    std::vector<std::string> pieces;
    for (size_t offset = 0; offset < normalized_text.size();) {
        const size_t start = offset;
        (void) next_utf8_codepoint(normalized_text, offset);
        std::string piece(normalized_text.substr(start, offset - start));
        if (vocab.find(piece) != vocab.end()) {
            pieces.push_back(std::move(piece));
            continue;
        }
        for (size_t i = start; i < offset; ++i) {
            pieces.push_back(byte_fallback_token(static_cast<unsigned char>(normalized_text[i])));
        }
    }
    return pieces;
}

bool is_cjk_codepoint(uint32_t codepoint) {
    return (codepoint >= 0x4E00 && codepoint <= 0x9FFF) ||
        (codepoint >= 0x3400 && codepoint <= 0x4DBF) ||
        (codepoint >= 0xF900 && codepoint <= 0xFAFF) ||
        (codepoint >= 0x20000 && codepoint <= 0x2A6DF);
}

bool is_pure_multichar_cjk(std::string_view text) {
    size_t count = 0;
    for (size_t offset = 0; offset < text.size();) {
        if (!is_cjk_codepoint(next_utf8_codepoint(text, offset))) {
            return false;
        }
        ++count;
    }
    return count >= 2;
}

std::string strip_sentencepiece_prefix(std::string token) {
    const std::string prefix = "\xE2\x96\x81";
    size_t pos = 0;
    while ((pos = token.find(prefix, pos)) != std::string::npos) {
        token.erase(pos, prefix.size());
    }
    return token;
}

bool starts_with_at(std::string_view text, size_t pos, std::string_view prefix) {
    return pos + prefix.size() <= text.size() && text.substr(pos, prefix.size()) == prefix;
}

std::string pair_key(const std::string & left, const std::string & right) {
    std::string key = left;
    key.push_back('\0');
    key += right;
    return key;
}

}  // namespace

struct VoxCPM1GgufTokenizer::Impl {
    std::unordered_map<std::string, int32_t> vocab;
    std::unordered_map<int32_t, std::string> id_to_token;
    std::unordered_map<std::string, int32_t> special_tokens;
    std::unordered_map<std::string, int32_t> merge_ranks;
    std::unordered_map<int32_t, std::vector<int32_t>> cjk_split_map;
    int32_t audio_start_token_id = 101;
    int32_t audio_end_token_id = 102;
    int32_t reference_audio_start_token_id = 103;
    int32_t reference_audio_end_token_id = 104;
    int32_t bos_token_id_ = 1;
    int32_t eos_token_id_ = 2;
    int32_t unk_token_id_ = 3;

    std::vector<std::string> bpe(std::string_view normalized_text) const {
        std::vector<std::string> word = bpe_initial_pieces(normalized_text, vocab);
        if (word.size() <= 1) {
            return word;
        }
        while (true) {
            int32_t best_rank = std::numeric_limits<int32_t>::max();
            size_t best_index = word.size();
            for (size_t i = 0; i + 1 < word.size(); ++i) {
                const auto it = merge_ranks.find(pair_key(word[i], word[i + 1]));
                if (it != merge_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_index = i;
                }
            }
            if (best_index == word.size()) {
                break;
            }
            word[best_index] += word[best_index + 1];
            word.erase(word.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
            if (word.size() <= 1) {
                break;
            }
        }
        return word;
    }

    void append_expanded_id(std::vector<int32_t> & ids, int32_t id) const {
        const auto split = cjk_split_map.find(id);
        if (split == cjk_split_map.end()) {
            ids.push_back(id);
            return;
        }
        ids.insert(ids.end(), split->second.begin(), split->second.end());
    }
};

VoxCPM1GgufTokenizer::VoxCPM1GgufTokenizer(std::shared_ptr<const engine::assets::TensorSource> gguf_source) {
    if (!gguf_source) {
        throw std::runtime_error("VoxCPM1 GGUF tokenizer requires a valid GGUF tensor source");
    }
    const GgufMetadataReader metadata(*gguf_source);
    if (!metadata.valid()) {
        throw std::runtime_error("VoxCPM1 GGUF tokenizer requires a GGUF tensor source");
    }
    impl_ = std::make_shared<Impl>();
    auto & impl = *impl_;

    // Read tokenizer metadata from GGUF directly in constructor
    const std::string tokenizer_model = metadata.require_string("tokenizer.ggml.model");
    const std::string tokenizer_pre = metadata.require_string("tokenizer.ggml.pre");
    const std::vector<std::string> tokens = metadata.require_string_array("tokenizer.ggml.tokens");
    const std::vector<int32_t> token_types = metadata.require_i32_array("tokenizer.ggml.token_type");
    const std::vector<std::string> merges = metadata.require_string_array("tokenizer.ggml.merges");
    const uint32_t bos_id = metadata.require_u32("tokenizer.ggml.bos_token_id");
    const uint32_t eos_id = metadata.require_u32("tokenizer.ggml.eos_token_id");
    const uint32_t unk_id = metadata.require_u32("tokenizer.ggml.unknown_token_id");

    if (tokenizer_model != "gpt2" || tokens.empty() || merges.empty() || token_types.size() != tokens.size()) {
        throw std::runtime_error("Invalid VoxCPM1 GGUF tokenizer metadata");
    }

    constexpr int32_t kTokenTypeNormal = 1;
    constexpr int32_t kTokenTypeByte = 6;

    for (size_t i = 0; i < tokens.size(); ++i) {
        const int32_t id = static_cast<int32_t>(i);
        impl.vocab.emplace(tokens[i], id);
        impl.id_to_token.emplace(id, tokens[i]);
        if (token_types[i] != kTokenTypeNormal && token_types[i] != kTokenTypeByte) {
            impl.special_tokens.emplace(tokens[i], id);
        }
    }

    impl.bos_token_id_ = static_cast<int32_t>(bos_id);
    impl.eos_token_id_ = static_cast<int32_t>(eos_id);
    impl.unk_token_id_ = static_cast<int32_t>(unk_id);

    // Build merge ranks
    int32_t rank = 0;
    for (const std::string & merge_text : merges) {
        const size_t split = merge_text.find(' ');
        if (split == std::string::npos) {
            ++rank;
            continue;
        }
        const std::string left = merge_text.substr(0, split);
        const std::string right = merge_text.substr(split + 1);
        const auto left_it = impl.vocab.find(left);
        const auto right_it = impl.vocab.find(right);
        const auto merged_it = impl.vocab.find(left + right);
        if (left_it != impl.vocab.end() && right_it != impl.vocab.end() && merged_it != impl.vocab.end()) {
            impl.merge_ranks.emplace(pair_key(left, right), rank);
        }
        ++rank;
    }

    if (impl.merge_ranks.empty()) {
        throw std::runtime_error("VoxCPM1 GGUF tokenizer has no valid merge rules");
    }

    // Build CJK split map
    for (const auto & [id, token] : impl.id_to_token) {
        const std::string clean = strip_sentencepiece_prefix(token);
        if (!is_pure_multichar_cjk(clean)) {
            continue;
        }
        std::vector<int32_t> char_ids;
        for (const auto & ch : utf8_codepoints(clean)) {
            const auto it = impl.vocab.find(ch);
            if (it == impl.vocab.end()) {
                char_ids.clear();
                break;
            }
            char_ids.push_back(it->second);
        }
        if (!char_ids.empty()) {
            impl.cjk_split_map.emplace(id, std::move(char_ids));
        }
    }
}

std::vector<int32_t> VoxCPM1GgufTokenizer::encode(const std::string & text) const {
    std::vector<int32_t> ids;
    for (size_t i = 0; i < text.size();) {
        const auto special_it = std::find_if(
            impl_->special_tokens.begin(),
            impl_->special_tokens.end(),
            [&](const auto & item) { return starts_with_at(text, i, item.first); });
        if (special_it != impl_->special_tokens.end()) {
            impl_->append_expanded_id(ids, special_it->second);
            i += special_it->first.size();
            continue;
        }

        size_t next_special = text.size();
        for (const auto & [special, _] : impl_->special_tokens) {
            const size_t pos = text.find(special, i);
            if (pos != std::string::npos) {
                next_special = std::min(next_special, pos);
            }
        }
        const std::string normalized = normalize_text(std::string_view(
            text.data() + static_cast<std::ptrdiff_t>(i),
            next_special - i));
        for (const auto & bpe_token : impl_->bpe(normalized)) {
            const auto vocab_it = impl_->vocab.find(bpe_token);
            if (vocab_it == impl_->vocab.end()) {
                throw std::runtime_error("VoxCPM1 tokenizer produced token not present in vocab: " + bpe_token);
            }
            impl_->append_expanded_id(ids, vocab_it->second);
        }
        i = next_special;
    }
    return ids;
}

VoxCPM2TextPrompt VoxCPM1GgufTokenizer::build_prompt(const std::string & text) const {
    if (text.empty()) {
        throw std::runtime_error("VoxCPM1 requires non-empty text input");
    }
    VoxCPM2TextPrompt prompt;
    prompt.text = text;
    prompt.input_ids = encode(text);
    if (prompt.input_ids.empty()) {
        throw std::runtime_error("VoxCPM1 tokenizer produced no tokens");
    }
    return prompt;
}

int32_t VoxCPM1GgufTokenizer::audio_start_token_id() const noexcept {
    return impl_->audio_start_token_id;
}

int32_t VoxCPM1GgufTokenizer::audio_end_token_id() const noexcept {
    return impl_->audio_end_token_id;
}

int32_t VoxCPM1GgufTokenizer::reference_audio_start_token_id() const noexcept {
    return impl_->reference_audio_start_token_id;
}

int32_t VoxCPM1GgufTokenizer::reference_audio_end_token_id() const noexcept {
    return impl_->reference_audio_end_token_id;
}

int32_t VoxCPM1GgufTokenizer::bos_token_id() const noexcept {
    return impl_->bos_token_id_;
}

int32_t VoxCPM1GgufTokenizer::eos_token_id() const noexcept {
    return impl_->eos_token_id_;
}

int32_t VoxCPM1GgufTokenizer::unk_token_id() const noexcept {
    return impl_->unk_token_id_;
}

bool VoxCPM1GgufTokenizer::has_tokenizer_metadata(const engine::assets::TensorSource & source) {
    const GgufMetadataReader metadata(source);
    return metadata.optional_string("tokenizer.ggml.model").has_value() &&
           metadata.optional_string_array("tokenizer.ggml.tokens").has_value() &&
           metadata.optional_string_array("tokenizer.ggml.merges").has_value();
}

}  // namespace engine::models::voxcpm2