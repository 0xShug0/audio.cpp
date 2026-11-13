#pragma once

#include "engine/community_models/zonos2/assets.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::zonos2 {

struct Zonos2PromptOptions {
    bool speaker = false;                  // prepend the speaker slot and its marker rows (cloning)
    bool clean_speaker_background = false; // marker row: clean (true) or noisy (false) reference
    bool accurate_mode = true;             // marker row: accurate (true) or expressive (false, no row)
    int speaking_rate_bucket = -1;         // -1 = no speaking-rate row
    std::vector<int> quality_buckets;      // empty = release default; else one entry per feature, -1 = skip
};

struct Zonos2Prompt {
    std::vector<int32_t> ids;  // row-major [n_rows][n_codebooks + 1]
    int n_rows = 0;
    int speaker_row = -1;      // row whose embedding the speaker projection replaces; -1 = none
};

// Row order: [speaker slot, background, accurate] [speaking rate] [quality...] BOS (192 + byte)... EOS
// [17 sheared silence rows]. Text is raw UTF-8 bytes (no normalization), as in Zyphra's prompt builder.
Zonos2Prompt build_zonos2_prompt(const Zonos2Hparams & hp, const std::string & utf8, const Zonos2PromptOptions & options);

// Parses the quality_buckets request option ("feature:bucket,..." or "none") into one bucket per feature.
std::vector<int> parse_zonos2_quality_buckets(const Zonos2Hparams & hp, const std::string & spec);

}  // namespace engine::models::zonos2
