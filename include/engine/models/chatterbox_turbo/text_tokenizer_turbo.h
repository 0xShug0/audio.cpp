#pragma once

#include "engine/framework/tokenizers/llama_bpe.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::chatterbox_turbo {

// Builds a GPT2-style BPE tokenizer from the vocab/merges arrays embedded as GGUF metadata
// (`tokenizer.ggml.tokens`/`tokenizer.ggml.merges`) in Chatterbox Turbo's T3 GGUF — there is no
// separate tokenizer.json file in this package. Materializes a vocab.json + merges.txt pair
// under a scratch directory and feeds those to the existing LlamaBpeTokenizer loader (Gpt2
// pre-tokenizer mode) rather than teaching that shared loader a new in-memory-array input path.
std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> load_chatterbox_turbo_tokenizer(
    const std::filesystem::path & t3_gguf_path,
    const std::filesystem::path & scratch_dir);

// Direct port of tts_turbo.py::punc_norm: capitalizes the first letter, collapses whitespace,
// replaces punctuation characters uncommon in the training data, and ensures a trailing
// sentence-ending punctuation mark.
std::string chatterbox_turbo_punc_norm(const std::string & text);

std::vector<int32_t> encode_chatterbox_turbo_text(
    const engine::tokenizers::LlamaBpeTokenizer & tokenizer,
    const std::string & text);

}  // namespace engine::models::chatterbox_turbo
