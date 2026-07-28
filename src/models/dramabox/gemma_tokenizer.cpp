#include "engine/models/dramabox/gemma_tokenizer.h"

#include "engine/framework/tokenizers/sentencepiece.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <utility>

namespace engine::models::dramabox {
namespace {

constexpr int32_t kGemmaPadTokenId = 0;
constexpr int32_t kGemmaBosTokenId = 2;

std::string strip_prompt(std::string text) {
    const auto begin = std::find_if_not(text.begin(), text.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
    const auto end = std::find_if_not(text.rbegin(), text.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    if (begin >= end) {
        return {};
    }
    return std::string(begin, end);
}

}  // namespace

struct DramaBoxGemmaTokenizer::Impl {
    std::vector<engine::tokenizers::SentencePiecePiece> pieces;
    int64_t max_length = 1024;
};

DramaBoxGemmaTokenizer::DramaBoxGemmaTokenizer(std::shared_ptr<const DramaBoxAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("DramaBox Gemma tokenizer requires assets");
    }
    auto impl = std::make_shared<Impl>();
    impl->pieces = engine::tokenizers::load_sentencepiece_model(assets->resources.require_file("gemma_tokenizer_model"));
    impl->max_length = assets->config.gemma.prompt_max_length;
    if (impl->max_length <= 0) {
        throw std::runtime_error("DramaBox Gemma tokenizer max length must be positive");
    }
    impl_ = std::move(impl);
}

DramaBoxGemmaTokenBatch DramaBoxGemmaTokenizer::encode(const std::vector<std::string> & prompts) const {
    if (prompts.empty()) {
        throw std::runtime_error("DramaBox Gemma tokenizer requires at least one prompt");
    }
    DramaBoxGemmaTokenBatch batch;
    batch.batch = static_cast<int64_t>(prompts.size());
    batch.tokens = impl_->max_length;
    batch.input_ids.assign(static_cast<size_t>(batch.batch * batch.tokens), kGemmaPadTokenId);
    batch.attention_mask.assign(static_cast<size_t>(batch.batch * batch.tokens), 0);
    for (int64_t b = 0; b < batch.batch; ++b) {
        std::vector<int32_t> ids;
        ids.reserve(static_cast<size_t>(batch.tokens));
        ids.push_back(kGemmaBosTokenId);
        auto pieces = engine::tokenizers::tokenize_sentencepiece(
            impl_->pieces,
            strip_prompt(prompts[static_cast<size_t>(b)]));
        ids.insert(ids.end(), pieces.begin(), pieces.end());
        if (static_cast<int64_t>(ids.size()) > batch.tokens) {
            ids.resize(static_cast<size_t>(batch.tokens));
        }
        const int64_t offset = batch.tokens - static_cast<int64_t>(ids.size());
        for (int64_t i = 0; i < static_cast<int64_t>(ids.size()); ++i) {
            const size_t index = static_cast<size_t>(b * batch.tokens + offset + i);
            batch.input_ids[index] = ids[static_cast<size_t>(i)];
            batch.attention_mask[index] = 1;
        }
    }
    return batch;
}

}  // namespace engine::models::dramabox
