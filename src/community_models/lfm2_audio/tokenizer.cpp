#include "engine/community_models/lfm2_audio/tokenizer.h"

#include "bpe-core.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace engine::community_models::lfm2_audio {
namespace {

namespace vendor = llama_tokenizer_vendor;

// llama.cpp token types, as stored in tokenizer.ggml.token_type.
constexpr int32_t kTokenTypeUnknown = 2;
constexpr int32_t kTokenTypeControl = 3;
constexpr int32_t kTokenTypeUserDefined = 4;

uint32_t token_attr(int32_t type) {
    switch (type) {
        case kTokenTypeUnknown: return vendor::TOKEN_ATTR_UNKNOWN;
        case kTokenTypeControl: return vendor::TOKEN_ATTR_CONTROL;
        case kTokenTypeUserDefined: return vendor::TOKEN_ATTR_USER_DEFINED;
        default: return 0;
    }
}

vendor::PreTokenizerType pre_tokenizer_type(const std::string & name) {
    // LFM2's pre-tokenizer regex is the Llama 3 one.
    if (name == "lfm2" || name == "llama-bpe" || name == "llama3") {
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

}  // namespace

Lfm2TextTokenizer::Lfm2TextTokenizer(const Lfm2TextVocabulary & vocabulary) : vocab_(build_vocabulary(vocabulary)) {}

std::vector<int32_t> Lfm2TextTokenizer::encode(const std::string & text) const {
    return vendor::tokenize_bpe(*vocab_, text, true);
}

std::string Lfm2TextTokenizer::decode(const std::vector<int32_t> & token_ids) const {
    return vendor::decode_bpe(*vocab_, token_ids, true);
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

}  // namespace engine::community_models::lfm2_audio
