#include "engine/community_models/glm_tts/tokenizer_text.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <stdexcept>

namespace engine::models::glm_tts {

struct GlmTTSTextTokenizer::Impl {
    Impl(
        const std::filesystem::path & vocab_path,
        const std::filesystem::path & merges_path,
        const std::filesystem::path & tokenizer_config_path)
        : tokenizer(engine::tokenizers::LlamaBpeTokenizerSpec{
              vocab_path,
              merges_path,
              tokenizer_config_path,
              std::nullopt,
              engine::tokenizers::LlamaBpePreTokenizer::Chatglm4}) {}

    engine::tokenizers::LlamaBpeTokenizer tokenizer;
};

GlmTTSTextTokenizer::GlmTTSTextTokenizer(
    const std::filesystem::path & vocab_path,
    const std::filesystem::path & merges_path,
    const std::filesystem::path & tokenizer_config_path)
    : impl_(std::make_shared<Impl>(
          vocab_path, merges_path, tokenizer_config_path)) {}

GlmTTSTextTokenizer::~GlmTTSTextTokenizer() = default;

std::vector<int32_t> GlmTTSTextTokenizer::encode(
    const std::string & text) const {
    return impl_->tokenizer.encode(text, true);
}

int32_t GlmTTSTextTokenizer::require_token_id(
    const std::string & token) const {
    const auto id = impl_->tokenizer.find_token_id(token);
    if (!id.has_value()) {
        throw std::runtime_error(
            "GLM-TTS tokenizer is missing required token: " + token);
    }
    return *id;
}

}  // namespace engine::models::glm_tts
