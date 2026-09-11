#include "engine/models/vevo2/tokenizer_ar.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <cctype>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace engine::models::vevo2 {
namespace {

std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> load_tokenizer(const Vevo2Assets & assets) {
    engine::tokenizers::LlamaBpeTokenizerSpec spec;
    spec.vocab_path = assets.resources.require_file("ar_vocab");
    spec.merges_path = assets.resources.require_file("ar_merges");
    spec.tokenizer_config_path = assets.resources.require_file("ar_tokenizer_config");
    spec.tokenizer_json_path = assets.resources.require_file("ar_tokenizer_json");
    spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;
    return engine::tokenizers::load_llama_bpe_tokenizer(spec);
}

int32_t require_token_id(
    const engine::tokenizers::LlamaBpeTokenizer & tokenizer,
    const std::string & token,
    const char * role) {
    const auto id = tokenizer.find_token_id(token);
    if (!id.has_value()) {
        throw std::runtime_error(std::string("Vevo2 AR tokenizer missing ") + role + " token: " + token);
    }
    return *id;
}

}  // namespace

Vevo2ARTokenizer::Vevo2ARTokenizer(std::shared_ptr<const Vevo2Assets> assets)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Vevo2 AR tokenizer requires assets");
    }
    tokenizer_ = load_tokenizer(*assets_);
    eos_token_id_ = require_token_id(*tokenizer_, "<|im_end|>", "eos");
    pad_token_id_ = require_token_id(*tokenizer_, "<|endoftext|>", "pad");
    if (eos_token_id_ != assets_->config.ar.eos_token_id) {
        throw std::runtime_error("Vevo2 AR tokenizer eos token id does not match model config");
    }
    if (pad_token_id_ != assets_->config.ar.pad_token_id) {
        throw std::runtime_error("Vevo2 AR tokenizer pad token id does not match generation config");
    }
}

Vevo2TokenizedPrompt Vevo2ARTokenizer::tokenize_prompt(const std::string & prompt) const {
    Vevo2TokenizedPrompt out;
    out.text = prompt;
    out.input_ids = tokenizer_->encode(prompt);
    return out;
}

std::string Vevo2ARTokenizer::decode(const std::vector<int32_t> & token_ids, bool skip_special_tokens) const {
    return tokenizer_->decode(token_ids, skip_special_tokens);
}

std::optional<int32_t> Vevo2ARTokenizer::find_token_id(const std::string & token) const {
    return tokenizer_->find_token_id(token);
}

int32_t Vevo2ARTokenizer::eos_token_id() const noexcept {
    return eos_token_id_;
}

int32_t Vevo2ARTokenizer::pad_token_id() const noexcept {
    return pad_token_id_;
}

Vevo2TokenSequence parse_content_style_tokens(const std::string & generated_text) {
    constexpr std::string_view kPrefix = "<|content_style_";
    constexpr std::string_view kSuffix = "|>";
    Vevo2TokenSequence out;
    size_t cursor = 0;
    while (true) {
        const size_t start = generated_text.find(kPrefix, cursor);
        if (start == std::string::npos) {
            break;
        }
        const size_t number_start = start + kPrefix.size();
        const size_t end = generated_text.find(kSuffix, number_start);
        if (end == std::string::npos) {
            break;
        }
        int32_t value = 0;
        bool saw_digit = false;
        for (size_t index = number_start; index < end; ++index) {
            const unsigned char ch = static_cast<unsigned char>(generated_text[index]);
            if (!std::isdigit(ch)) {
                saw_digit = false;
                break;
            }
            saw_digit = true;
            value = static_cast<int32_t>(value * 10 + static_cast<int32_t>(ch - '0'));
        }
        if (saw_digit) {
            out.ids.push_back(value);
        }
        cursor = end + kSuffix.size();
    }
    return out;
}

}  // namespace engine::models::vevo2
