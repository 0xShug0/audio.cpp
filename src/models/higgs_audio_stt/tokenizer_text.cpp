#include "engine/models/higgs_audio_stt/tokenizer_text.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::higgs_audio_stt {

struct HiggsAudioSTTTextTokenizer::Impl {
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer;
};

namespace {

std::shared_ptr<const HiggsAudioSTTTextTokenizer::Impl> load_impl(const HiggsAudioSTTAssets & assets) {
    engine::tokenizers::LlamaBpeTokenizerSpec spec;
    spec.vocab_path = assets.paths.tokenizer_vocab_path;
    spec.merges_path = assets.paths.tokenizer_merges_path;
    spec.tokenizer_config_path = assets.paths.tokenizer_config_path;
    spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;

    auto impl = std::make_shared<HiggsAudioSTTTextTokenizer::Impl>();
    impl->tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);
    return impl;
}

std::string default_user_prompt(const std::string & context, bool enable_thinking) {
    const std::string prompt = context.empty()
        ? "Transcribe the speech. Output only the spoken words in lowercase with no punctuation."
        : context;
    std::string text = "<|im_start|>user\n" + prompt +
        "<|audio_bos|><|AUDIO|><|audio_eos|><|im_end|>\n<|im_start|>assistant\n";
    if (!enable_thinking) {
        text += "<think>\n\n</think>\n\n";
    }
    return text;
}

}  // namespace

HiggsAudioSTTTextTokenizer::HiggsAudioSTTTextTokenizer(std::shared_ptr<const HiggsAudioSTTAssets> assets)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Higgs Audio STT tokenizer requires assets");
    }
    impl_ = load_impl(*assets_);
}

HiggsAudioSTTPrompt HiggsAudioSTTTextTokenizer::build_prompt(
    const std::string & context,
    const std::string & language,
    const bool enable_thinking,
    const int64_t audio_feature_tokens) const {
    (void) language;
    const std::string prompt = default_user_prompt(context, enable_thinking);
    return build_raw_audio_prompt(prompt, audio_feature_tokens);
}

HiggsAudioSTTPrompt HiggsAudioSTTTextTokenizer::build_raw_audio_prompt(
    const std::string & text,
    const int64_t audio_feature_tokens) const {
    if (audio_feature_tokens <= 0) {
        throw std::runtime_error("Higgs Audio STT prompt requires positive audio feature token count");
    }
    const auto ids = impl_->tokenizer->encode(text);
    const int32_t audio_token = static_cast<int32_t>(assets_->config.text_decoder.audio_in_token_id);
    std::vector<int32_t> expanded;
    expanded.reserve(ids.size() + static_cast<size_t>(audio_feature_tokens));
    HiggsAudioSTTPrompt result;
    for (const int32_t id : ids) {
        if (id == audio_token) {
            for (int64_t i = 0; i < audio_feature_tokens; ++i) {
                result.audio_token_positions.push_back(static_cast<int32_t>(expanded.size()));
                expanded.push_back(id);
            }
        } else {
            expanded.push_back(id);
        }
    }
    result.input_ids = std::move(expanded);
    result.attention_mask.assign(result.input_ids.size(), 1);
    return result;
}

std::string HiggsAudioSTTTextTokenizer::decode(const std::vector<int32_t> & token_ids) const {
    std::vector<int32_t> filtered;
    filtered.reserve(token_ids.size());
    for (const int32_t id : token_ids) {
        if (id == assets_->config.text_decoder.pad_token_id ||
            std::find(
                assets_->config.text_decoder.eos_token_ids.begin(),
                assets_->config.text_decoder.eos_token_ids.end(),
                static_cast<int64_t>(id)) != assets_->config.text_decoder.eos_token_ids.end() ||
            impl_->tokenizer->is_control_token_id(id)) {
            continue;
        }
        filtered.push_back(id);
    }
    return impl_->tokenizer->decode(filtered);
}

}  // namespace engine::models::higgs_audio_stt
