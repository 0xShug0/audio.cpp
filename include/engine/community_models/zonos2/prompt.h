#pragma once

#include "engine/community_models/zonos2/assets.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace engine::models::zonos2 {

struct Zonos2PromptOptions {
    bool speaker = false;                  // prepend the speaker slot and its marker rows (cloning)
    bool clean_speaker_background = false; // marker row: clean (true) or noisy (false) reference
    bool accurate_mode = true;             // marker row: accurate (true) or expressive (false, no row)
    int speaking_rate_bucket = -1;         // -1 = no speaking-rate row
    std::vector<int> quality_buckets;      // empty = release default; else one entry per feature, -1 = skip
    bool trailing_silence = true;          // false drops the trailing_silence_s row: another chunk continues this one
};

// An earlier chunk of a long-form request: its text and the delayed rows kept from its generation.
struct Zonos2ContextChunk {
    std::string_view text;
    const int32_t * rows = nullptr;  // [n_rows][n_codebooks]
    int n_rows = 0;
};

struct Zonos2Prompt {
    std::vector<int32_t> ids;  // row-major [n_rows][n_codebooks + 1]
    int n_rows = 0;
    int speaker_row = -1;      // row whose embedding the speaker projection replaces; -1 = none
};

// Row order: [speaker slot, background, accurate] [speaking rate] [quality...] BOS (192 + byte)... EOS
// [17 sheared silence rows]. Text is raw UTF-8 bytes (no normalization), as in Zyphra's prompt builder.
Zonos2Prompt build_zonos2_prompt(const Zonos2Hparams & hp, const std::string & utf8, const Zonos2PromptOptions & options);

// Continuation prompt of ZONOS2's long-form generation (tts/longform.py): the same conditioning rows, then BOS, the
// context texts and `utf8` joined by single spaces, EOS, and the context rows as a teacher-forced audio prefix in
// place of the silence rows. Generation continues the last context row.
Zonos2Prompt build_zonos2_continuation_prompt(const Zonos2Hparams & hp, const std::vector<Zonos2ContextChunk> & context,
                                              const std::string & utf8, const Zonos2PromptOptions & options);

// ZONOS2's long-form word packing (tts/longform.py split_text_words): whitespace-separated words packed greedily into
// chunks of at most `max_codepoints`, joined by single spaces, so chunks usually end mid-sentence. A word longer than
// the budget is a chunk of its own. Words are split at ASCII whitespace.
std::vector<std::string> split_zonos2_long_form_text(std::string_view utf8, int max_codepoints);

// Parses the quality_buckets request option ("feature:bucket,..." or "none") into one bucket per feature.
std::vector<int> parse_zonos2_quality_buckets(const Zonos2Hparams & hp, const std::string & spec);

}  // namespace engine::models::zonos2
