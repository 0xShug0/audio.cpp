#include "engine/community_models/mira_tts/prompt.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <stdexcept>
#include <utility>

namespace engine::community_models::mira_tts {
namespace {

int32_t require_token_id(
    const tokenizers::LlamaBpeTokenizer & tokenizer,
    const std::string & token) {
    const auto id = tokenizer.find_token_id(token);
    if (!id.has_value()) {
        throw std::runtime_error("MiraTTS tokenizer is missing token " + token);
    }
    return *id;
}

}  // namespace

struct MiraPromptBuilder::Impl {
    explicit Impl(std::shared_ptr<const MiraTTSAssets> input_assets)
        : assets(std::move(input_assets)) {
        if (assets == nullptr) {
            throw std::runtime_error("MiraTTS prompt builder requires assets");
        }
        tokenizers::LlamaBpeTokenizerSpec spec;
        spec.tokenizer_json_path = assets->resources.require_file("tokenizer_json");
        spec.tokenizer_config_path = assets->resources.require_file("tokenizer_config");
        spec.pre_type = tokenizers::LlamaBpePreTokenizer::Qwen2;
        tokenizer = tokenizers::load_llama_bpe_tokenizer(spec);
        task_tts = require_token_id(*tokenizer, "<|task_tts|>");
        start_text = require_token_id(*tokenizer, "<|start_text|>");
        end_text = require_token_id(*tokenizer, "<|end_text|>");
        context_start = require_token_id(*tokenizer, "<|context_audio_start|>");
        context_end = require_token_id(*tokenizer, "<|context_audio_end|>");
        speech_start = require_token_id(*tokenizer, "<|prompt_speech_start|>");
        context_token_start = require_token_id(*tokenizer, "<|context_token_0|>");
    }

    std::shared_ptr<const MiraTTSAssets> assets;
    std::shared_ptr<tokenizers::LlamaBpeTokenizer> tokenizer;
    int32_t task_tts = 0;
    int32_t start_text = 0;
    int32_t end_text = 0;
    int32_t context_start = 0;
    int32_t context_end = 0;
    int32_t speech_start = 0;
    int32_t context_token_start = 0;
};

MiraPromptBuilder::MiraPromptBuilder(std::shared_ptr<const MiraTTSAssets> assets)
    : impl_(std::make_unique<Impl>(std::move(assets))) {}

MiraPromptBuilder::~MiraPromptBuilder() = default;

std::vector<int32_t> MiraPromptBuilder::build(
    const std::string & text,
    const std::vector<int32_t> & context_codes) const {
    if (text.empty()) {
        throw std::runtime_error("MiraTTS requires non-empty text");
    }
    if (context_codes.size() != 32) {
        throw std::runtime_error("MiraTTS speaker encoder must produce 32 context codes");
    }
    auto text_ids = impl_->tokenizer->encode(text, false);
    std::vector<int32_t> out;
    out.reserve(text_ids.size() + context_codes.size() + 6);
    out.push_back(impl_->task_tts);
    out.push_back(impl_->start_text);
    out.insert(out.end(), text_ids.begin(), text_ids.end());
    out.push_back(impl_->end_text);
    out.push_back(impl_->context_start);
    for (const int32_t code : context_codes) {
        if (code < 0 || code >= 4096) {
            throw std::runtime_error("MiraTTS context code is outside [0, 4096)");
        }
        out.push_back(impl_->context_token_start + code);
    }
    out.push_back(impl_->context_end);
    out.push_back(impl_->speech_start);
    return out;
}

}  // namespace engine::community_models::mira_tts
