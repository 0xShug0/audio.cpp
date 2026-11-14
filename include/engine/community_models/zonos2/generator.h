#pragma once

#include "engine/community_models/zonos2/backbone.h"
#include "engine/community_models/zonos2/sampler.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace engine::models::zonos2 {

inline constexpr int kZonos2EosDrainFrames = 10;

struct Zonos2Codes {
    std::vector<int32_t> rows;  // [count][n_codebooks] raw delayed rows (EOA and pads included)
    int count = 0;
    int eos_frame = -1;          // undelayed frame where speech ends; -1 = none
    bool reached_limit = false;  // stopped at max_frames without an end of speech
};

// Called with each generated delayed row and the end-of-speech frame once known (-1 before).
using Zonos2RowCallback = std::function<void(const int32_t * row, int eos_frame)>;

// ZONOS2's decode loop: each frame is fed back as [codes, text pad]; the first EOA in any codebook c at step s sets
// eos_frame = max(0, s - c) and starts a 10-frame drain (the detection frame counts). max_frames counts generated
// frames, the drain included.
Zonos2Codes generate_zonos2_codes(const Zonos2Hparams & hp, Zonos2Backbone & backbone, const Zonos2Prompt & prompt,
                                  const float * speaker, int max_frames, const Zonos2SamplingOptions & sampling,
                                  const Zonos2RowCallback & on_row);

}  // namespace engine::models::zonos2
