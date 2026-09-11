#include "engine/models/higgs_audio_tts/tokenizer_text.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::higgs_audio_tts {
namespace {

int32_t require_token_id(const engine::tokenizers::LlamaBpeTokenizer & tokenizer, const std::string & token) {
    const auto id = tokenizer.find_token_id(token);
    if (!id.has_value()) {
        throw std::runtime_error("Higgs TTS tokenizer missing required token: " + token);
    }
    return *id;
}

}  // namespace

struct HiggsTextTokenizer::Impl {
    explicit Impl(std::shared_ptr<const HiggsAssets> input_assets)
        : assets(std::move(input_assets)),
          tokenizer(engine::tokenizers::LlamaBpeTokenizerSpec{
              {},
              {},
              assets->resources.require_file("tokenizer_config"),
              assets->resources.require_file("tokenizer_json"),
              engine::tokenizers::LlamaBpePreTokenizer::Qwen2,
          }),
          tts_id(require_token_id(tokenizer, "<|tts|>")),
          ref_audio_id(require_token_id(tokenizer, "<|ref_audio|>")),
          ref_text_id(require_token_id(tokenizer, "<|ref_text|>")),
          text_id(require_token_id(tokenizer, "<|text|>")),
          audio_id(require_token_id(tokenizer, "<|audio|>")),
          audio_placeholder_id(static_cast<int32_t>(assets->config.audio_token_id)) {}

    std::shared_ptr<const HiggsAssets> assets;
    engine::tokenizers::LlamaBpeTokenizer tokenizer;
    int32_t tts_id = 0;
    int32_t ref_audio_id = 0;
    int32_t ref_text_id = 0;
    int32_t text_id = 0;
    int32_t audio_id = 0;
    int32_t audio_placeholder_id = -100;
};

HiggsTextTokenizer::HiggsTextTokenizer(std::shared_ptr<const HiggsAssets> assets)
    : impl_([&]() {
          if (assets == nullptr) {
              throw std::runtime_error("Higgs TTS text tokenizer requires assets");
          }
          return std::make_shared<Impl>(std::move(assets));
      }()) {}

std::vector<int32_t> HiggsTextTokenizer::encode(const std::string & text) const {
    return impl_->tokenizer.encode(text, true);
}

HiggsPromptEncoding HiggsTextTokenizer::encode_prompt(const HiggsPromptRequest & request) const {
    if (request.delayed_reference_tokens < 0) {
        throw std::runtime_error("Higgs TTS delayed_reference_tokens must be non-negative");
    }

    HiggsPromptEncoding encoding;
    encoding.text_ids = encode(request.text);
    if (!request.reference_text.empty() && request.delayed_reference_tokens > 0) {
        encoding.reference_text_ids = encode(request.reference_text);
    }

    encoding.token_ids.push_back(impl_->tts_id);
    if (!encoding.reference_text_ids.empty()) {
        encoding.token_ids.push_back(impl_->ref_text_id);
        encoding.token_ids.insert(
            encoding.token_ids.end(),
            encoding.reference_text_ids.begin(),
            encoding.reference_text_ids.end());
    }
    if (request.delayed_reference_tokens > 0) {
        encoding.token_ids.push_back(impl_->ref_audio_id);
        encoding.token_ids.insert(
            encoding.token_ids.end(),
            static_cast<size_t>(request.delayed_reference_tokens),
            impl_->audio_placeholder_id);
    }
    encoding.token_ids.push_back(impl_->text_id);
    encoding.token_ids.insert(encoding.token_ids.end(), encoding.text_ids.begin(), encoding.text_ids.end());
    encoding.token_ids.push_back(impl_->audio_id);
    return encoding;
}

}  // namespace engine::models::higgs_audio_tts
