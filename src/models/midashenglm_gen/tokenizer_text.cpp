#include "engine/models/midashenglm_gen/tokenizer_text.h"

#include "engine/framework/io/filesystem.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::midashenglm_gen {
namespace {

constexpr int32_t kQwen3PadTokenId = 151643;

std::filesystem::path require_tokenizer_file(const MiDashengLmGenAssets & assets, const char * filename) {
    const auto path = assets.model_root / filename;
    if (!engine::io::is_existing_file(path)) {
        throw std::runtime_error(
            std::string("MiDashengLM-Gen tokenizer file is missing: ") + filename +
            ". Copy the Qwen3 tokenizer sidecar into the model package.");
    }
    return path;
}

}  // namespace

class MiDashengLmGenTextTokenizer::Impl {
public:
    explicit Impl(const MiDashengLmGenAssets & assets) {
        engine::tokenizers::LlamaBpeTokenizerSpec spec;
        spec.vocab_path = require_tokenizer_file(assets, "vocab.json");
        spec.merges_path = require_tokenizer_file(assets, "merges.txt");
        spec.tokenizer_config_path = require_tokenizer_file(assets, "tokenizer_config.json");
        spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;
        tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);
    }

    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer;
};

MiDashengLmGenTextTokenizer::MiDashengLmGenTextTokenizer(std::shared_ptr<const MiDashengLmGenAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("MiDashengLM-Gen text tokenizer requires assets");
    }
    impl_ = std::make_shared<Impl>(*assets);
}

MiDashengLmGenTextTokenizer::~MiDashengLmGenTextTokenizer() = default;

std::string MiDashengLmGenTextTokenizer::build_prompt(std::string_view text) const {
    return "<|im_start|>user\nAudio Generation\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|im_end|>\n" +
        std::string(text) + "\n<|audio_bos|>";
}

MiDashengLmGenPromptBatch MiDashengLmGenTextTokenizer::encode_batch(const std::vector<std::string> & texts) const {
    if (texts.empty()) {
        throw std::runtime_error("MiDashengLM-Gen text tokenizer requires at least one input text");
    }
    std::vector<std::vector<int32_t>> rows;
    rows.reserve(texts.size());
    MiDashengLmGenPromptBatch out;
    out.prompts.reserve(texts.size());
    int64_t max_tokens = 0;
    for (const auto & text : texts) {
        auto prompt = build_prompt(text);
        rows.push_back(impl_->tokenizer->encode(prompt, true));
        if (rows.back().empty()) {
            throw std::runtime_error("MiDashengLM-Gen text tokenizer produced empty token sequence");
        }
        max_tokens = std::max<int64_t>(max_tokens, static_cast<int64_t>(rows.back().size()));
        out.prompts.push_back(std::move(prompt));
    }

    out.batch = static_cast<int64_t>(rows.size());
    out.tokens = max_tokens;
    out.token_ids.assign(static_cast<size_t>(out.batch * out.tokens), kQwen3PadTokenId);
    out.attention_mask.assign(static_cast<size_t>(out.batch * out.tokens), 0);
    for (int64_t batch = 0; batch < out.batch; ++batch) {
        const auto & row = rows[static_cast<size_t>(batch)];
        const int64_t valid = static_cast<int64_t>(row.size());
        for (int64_t token = 0; token < valid; ++token) {
            const size_t index = static_cast<size_t>(batch * out.tokens + token);
            out.token_ids[index] = row[static_cast<size_t>(token)];
            out.attention_mask[index] = 1;
        }
    }
    return out;
}

}  // namespace engine::models::midashenglm_gen
