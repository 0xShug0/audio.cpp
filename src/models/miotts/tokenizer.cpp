#include "engine/models/miotts/tokenizer.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <stdexcept>
#include <utility>

namespace engine::models::miotts {
namespace {

int32_t require_token_id(
    const engine::tokenizers::LlamaBpeTokenizer & tokenizer,
    const std::string & token,
    const char * role) {
    const auto id = tokenizer.find_token_id(token);
    if (!id.has_value()) {
        throw std::runtime_error(std::string("MioTTS tokenizer missing ") + role + " token: " + token);
    }
    return *id;
}

std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> load_tokenizer(const MioTTSAssets & assets) {
    engine::tokenizers::LlamaBpeTokenizerSpec spec;
    spec.vocab_path = assets.resources.require_file("vocab");
    spec.merges_path = assets.resources.require_file("merges");
    spec.tokenizer_config_path = assets.resources.require_file("tokenizer_config");
    if (const auto * tokenizer_json = assets.resources.find_file("tokenizer_json"); tokenizer_json != nullptr) {
        spec.tokenizer_json_path = *tokenizer_json;
    }
    spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen35;
    return engine::tokenizers::load_llama_bpe_tokenizer(spec);
}

}  // namespace

MioTTSTokenizer::MioTTSTokenizer(std::shared_ptr<const MioTTSAssets> assets)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("MioTTS tokenizer requires assets");
    }
    tokenizer_ = load_tokenizer(*assets_);
    im_start_token_id_ = require_token_id(*tokenizer_, "<|im_start|>", "im_start");
    im_end_token_id_ = require_token_id(*tokenizer_, "<|im_end|>", "im_end");
    if (im_end_token_id_ != assets_->config.eos_token_id) {
        throw std::runtime_error("MioTTS tokenizer im_end token does not match eos_token_id");
    }
}

MioTTSPrompt MioTTSTokenizer::build_prompt(const std::string & text) const {
    if (text.empty()) {
        throw std::runtime_error("MioTTS requires non-empty text input");
    }
    MioTTSPrompt prompt;
    prompt.text = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
    prompt.input_ids = tokenizer_->encode(prompt.text);
    if (prompt.input_ids.empty()) {
        throw std::runtime_error("MioTTS prompt tokenizer produced no tokens");
    }
    if (prompt.input_ids.front() != im_start_token_id_) {
        throw std::runtime_error("MioTTS prompt must begin with im_start");
    }
    return prompt;
}

std::string MioTTSTokenizer::decode(const std::vector<int32_t> & token_ids, bool skip_special_tokens) const {
    return tokenizer_->decode(token_ids, skip_special_tokens);
}

}  // namespace engine::models::miotts
