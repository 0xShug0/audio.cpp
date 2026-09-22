#include "engine/community_models/vieneu_v3_turbo/tokenizer_text.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <stdexcept>

namespace engine::models::vieneu_v3_turbo {

struct Qwen3TextTokenizer::Impl {
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer;
};

namespace {

std::shared_ptr<const Qwen3TextTokenizer::Impl> load_impl(const VieNeuTTSAssets & assets) {
    auto impl = std::make_shared<Qwen3TextTokenizer::Impl>();
    engine::tokenizers::LlamaBpeTokenizerSpec spec;
    spec.tokenizer_json_path = assets.resources.require_file("tokenizer_json");
    spec.tokenizer_config_path = assets.resources.require_file("tokenizer_config");
    spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;
    impl->tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);
    return impl;
}

}  // namespace

Qwen3TextTokenizer::Qwen3TextTokenizer(std::shared_ptr<const VieNeuTTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("VieNeu-TTS text tokenizer requires assets");
    }
    impl_ = load_impl(*assets);
}

std::string Qwen3TextTokenizer::build_assistant_prompt(const std::string & text) const {
    return text;
}

std::string Qwen3TextTokenizer::build_reference_prompt(const std::string & text) const {
    return text;
}

std::string Qwen3TextTokenizer::build_instruct_prompt(const std::string & text) const {
    return text;
}

std::vector<int32_t> Qwen3TextTokenizer::encode(const std::string & text) const {
    return impl_->tokenizer->encode(text);
}

}  // namespace engine::models::vieneu_v3_turbo
