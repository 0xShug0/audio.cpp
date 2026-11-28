#pragma once

#include "engine/models/qwen3_asr/tokenizer_text.h"
#include "engine/models/qwen3_asr/types.h"

namespace engine::models::qwen3_asr {

// Select original punctuated text using the session's retained word selection.
// Only called for explicitly enabled trimming of overlapping timestamp chunks.
std::string trim_chunk_text(
    const std::string & text,
    const std::vector<runtime::WordTimestamp> & words,
    const runtime::WordTimestamp * kept,
    size_t kept_count,
    int64_t source_start_sample);

class Qwen3ASRPostprocessor {
public:
    explicit Qwen3ASRPostprocessor(const Qwen3ASRTextTokenizer & tokenizer);

    Qwen3ASRResult decode(const Qwen3ASRGeneratedTokens & tokens, const Qwen3ASRRequest & request) const;

private:
    const Qwen3ASRTextTokenizer & tokenizer_;
};

}  // namespace engine::models::qwen3_asr
