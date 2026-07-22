#include "engine/models/ace_step/tokenizer_text.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <stdexcept>

namespace engine::models::ace_step {

struct AceStepTextTokenizer::Impl {
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer;
    int32_t pad_token_id = 0;
};

namespace {

std::shared_ptr<const AceStepTextTokenizer::Impl> load_impl(
    const std::filesystem::path & vocab_path,
    const std::filesystem::path & merges_path,
    const std::filesystem::path & tokenizer_config_path,
    const std::filesystem::path & tokenizer_json_path) {
    engine::tokenizers::LlamaBpeTokenizerSpec spec;
    spec.vocab_path = vocab_path;
    spec.merges_path = merges_path;
    spec.tokenizer_config_path = tokenizer_config_path;
    spec.tokenizer_json_path = tokenizer_json_path;
    spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;

    auto impl = std::make_shared<AceStepTextTokenizer::Impl>();
    impl->tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);

    const auto pad_token_id = impl->tokenizer->configured_pad_token_id();
    if (!pad_token_id.has_value()) {
        throw std::runtime_error("ACE-Step tokenizer is missing configured pad_token_id");
    }
    impl->pad_token_id = *pad_token_id;
    return impl;
}

}  // namespace

AceStepTextTokenizer::AceStepTextTokenizer(
    std::shared_ptr<const AceStepAssets> assets,
    ResourceSet resources)
    : assets_(std::move(assets)),
      resources_(resources) {
    if (assets_ == nullptr) {
        throw std::runtime_error("ACE-Step tokenizer requires assets");
    }
    const bool use_planner = resources_ == ResourceSet::Planner;
    impl_ = load_impl(
        assets_->resources.require_file(use_planner ? "lm_vocab" : "text_encoder_vocab"),
        assets_->resources.require_file(use_planner ? "lm_merges" : "text_encoder_merges"),
        assets_->resources.require_file(use_planner ? "lm_tokenizer_config" : "text_encoder_tokenizer_config"),
        assets_->resources.require_file(use_planner ? "lm_tokenizer_json" : "text_encoder_tokenizer_json"));
}

int32_t AceStepTextTokenizer::pad_token_id() const noexcept {
    return impl_->pad_token_id;
}

std::vector<int32_t> AceStepTextTokenizer::encode(const std::string & text) const {
    return impl_->tokenizer->encode(text);
}

std::string AceStepTextTokenizer::decode(
    const std::vector<int32_t> & token_ids,
    bool skip_control_special_tokens) const {
    std::vector<int32_t> filtered;
    filtered.reserve(token_ids.size());
    for (const int32_t token_id : token_ids) {
        if (token_id != impl_->pad_token_id) {
            filtered.push_back(token_id);
        }
    }
    return impl_->tokenizer->decode(filtered, skip_control_special_tokens);
}

std::optional<int32_t> AceStepTextTokenizer::find_token_id(const std::string & token) const {
    return impl_->tokenizer->find_token_id(token);
}

AceStepTokenizedText AceStepTextTokenizer::tokenize_text(const std::string & text, int64_t max_length) const {
    if (max_length <= 0) {
        throw std::runtime_error("ACE-Step tokenizer max_length must be positive");
    }
    AceStepTokenizedText tokenized;
    tokenized.text = text;
    tokenized.input_ids = impl_->tokenizer->encode(text);
    if (resources_ == ResourceSet::TextEncoder) {
        tokenized.input_ids.push_back(impl_->pad_token_id);
    }
    if (static_cast<int64_t>(tokenized.input_ids.size()) > max_length) {
        tokenized.input_ids.resize(static_cast<size_t>(max_length));
    }
    tokenized.attention_mask.assign(tokenized.input_ids.size(), 1);
    const size_t padded = static_cast<size_t>(max_length);
    tokenized.input_ids.resize(padded, impl_->pad_token_id);
    tokenized.attention_mask.resize(padded, 0);
    return tokenized;
}

std::string AceStepTextTokenizer::apply_chat_template(
    const std::string & system_content,
    const std::string & user_content,
    bool add_generation_prompt) const {
    std::string formatted = "<|im_start|>system\n" + system_content + "<|im_end|>\n"
        "<|im_start|>user\n" + user_content + "<|im_end|>\n";
    if (add_generation_prompt) {
        formatted += "<|im_start|>assistant\n";
    }
    return formatted;
}

}  // namespace engine::models::ace_step
