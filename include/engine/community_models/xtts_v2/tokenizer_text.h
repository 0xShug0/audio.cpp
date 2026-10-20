#pragma once

// BPE tokenizer + per-language text normalization for xTTS v2.
//
// Phase 1 languages (fully portable): en, de, fr, es, it, pt, pl, tr, ru, nl, cs, ar, hu
// Phase 2 languages (mapping-table based): zh-cn, ko
// Phase 3 languages (external dependency): ja (requires MeCab/cutlet), hi

#include <filesystem>
#include <string>
#include <vector>
#include <cstdint>

namespace engine::models::xtts_v2 {

struct XttSv2BpeVocab;

// Load tokenizer from GGUF metadata.
bool load_xtts_v2_tokenizer(XttSv2BpeVocab * tok, const std::filesystem::path & gguf_path);

// Tokenize text with language prefix. Returns token IDs.
std::vector<int32_t> tokenize_xtts_v2(const XttSv2BpeVocab * tok, const std::string & text, const char * language);

// Text normalization (language-specific). Returns normalized text.
std::string normalize_xtts_v2_text(const std::string & text, const char * language);

// Get the language index for a language code string. Returns -1 if not found.
int xtts_v2_language_index(const char * language);

}  // namespace engine::models::xtts_v2
