#pragma once

#include "engine/community_models/zonos3/assets.h"
#include "engine/community_models/zonos3/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::zonos3 {

// Prompt rows whose summed input embedding is replaced before emb_norm (absolute rows).
struct Zonos3Conditioning {
    int speaker_row = -1;          // replaced by speaker_proj . speaker + bias; -1 = none
    std::vector<float> speaker;    // [speaker_dim] when speaker_row >= 0
    int condition_row = -1;        // first of n_cond rows; row i is replaced by cond_embd[i][buckets[i]]
    std::vector<int32_t> buckets;  // [n_cond]; buckets[i] == cond_bucket_counts[i] means unknown
};

struct Zonos3PromptOptions {
    const int32_t * reference_codes = nullptr;  // undelayed [frames][10] (reference mode)
    int reference_frames = 0;                   // 0..900
    const float * speaker_embedding = nullptr;  // [speaker_dim] (embedding mode); exclusive with reference codes
    std::vector<int32_t> condition_buckets;     // empty = all unknown, else n_cond entries
    int silence_prefill_frames = 3;             // 0..30
};

struct Zonos3Prompt {
    std::vector<int32_t> ids;  // row-major [n_rows][n_codebooks + 1]
    int n_rows = 0;
    int speaker_rows = 0;      // leading rows the CFG twin drops: 1, reference frames + 9, or 0
    int prefill_frames = 0;    // trailing silence rows = first rows of the audio stream
    Zonos3Conditioning cond;
};

// Row order: [speaker row | reference prefix] [loudness][fmax][speaking_rate] BOS text EOS [silence rows].
Zonos3Prompt build_zonos3_prompt(const Zonos3Hparams & hp, const Zonos3Tokenizer & tokenizer, const std::string & text,
                                 const Zonos3PromptOptions & options);
// The speaker-free CFG twin: speaker rows dropped, every condition bucket unknown.
Zonos3Prompt build_zonos3_cfg_twin(const Zonos3Hparams & hp, const Zonos3Prompt & prompt);
// bisect_right over the condition's edges; -1 when the value is not finite.
int zonos3_condition_bucket(const Zonos3Hparams & hp, int index, double value);
// The audio columns of the silence-prefill rows: [prefill_frames][n_codebooks].
std::vector<int32_t> zonos3_prefill_codes(const Zonos3Hparams & hp, const Zonos3Prompt & prompt);
// Every all-pad row is a replaced row, and every replaced row is all-pad.
void zonos3_check_conditioning(const Zonos3Hparams & hp, const Zonos3Prompt & prompt);
// out = uncond + s * (cond - uncond), in F32.
void zonos3_cfg_mix(const float * cond, const float * uncond, float scale, size_t n, float * out);

}  // namespace engine::models::zonos3
