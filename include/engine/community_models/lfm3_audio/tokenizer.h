#pragma once

// LFM3-Audio text tokenizer: the framework Llama-BPE tokenizer over the
// checkpoint's tokenizer.json, or over the vocabulary of Liquid's published
// backbone GGUF, with special-token parsing as HF does it.

#include "engine/framework/tokenizers/llama_bpe.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::lfm3_audio {

struct Lfm3TokenIds {  // resolved by content via find_token_id; throws if any is missing
    int32_t bos, im_start, im_end, nl /*"\n" = 207*/, text_end, audio_start, mixed_start, audio, audio_out, think, think_end;
};

class Lfm3Tokenizer {
public:
    explicit Lfm3Tokenizer(const std::filesystem::path & tokenizer_json);  // Llama3 pre-tokenizer, ignore_merges honoured
    explicit Lfm3Tokenizer(const tokenizers::LlamaBpeGgufVocabulary & vocabulary);  // the same from tokenizer.ggml.*
    [[nodiscard]] std::vector<int32_t> encode(const std::string & text) const;  // parse_special = true (HF semantics)
    // HF decode(skip_special_tokens=True): drops CONTROL ids (is_control_token_id) and <|audio_out|>,
    // keeps <think>/</think> and tool tags. Implemented as filter + decode(ids, false), because vendor decode_bpe
    // skip=true also drops USER_DEFINED. skip_special = false is HF decode(skip_special_tokens=False).
    [[nodiscard]] std::string decode(const std::vector<int32_t> & ids, bool skip_special = true) const;
    [[nodiscard]] const Lfm3TokenIds & ids() const noexcept;

private:
    explicit Lfm3Tokenizer(tokenizers::LlamaBpeTokenizerSpec spec);

    std::shared_ptr<const tokenizers::LlamaBpeTokenizer> bpe_;
    Lfm3TokenIds ids_;
};

}  // namespace engine::community_models::lfm3_audio
