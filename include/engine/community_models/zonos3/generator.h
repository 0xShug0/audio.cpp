#pragma once

#include "engine/community_models/zonos3/backbone.h"
#include "engine/community_models/zonos3/sampler.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace engine::models::zonos3 {

struct Zonos3Codes {
    int eos_frame = -1;         // audio-stream frame (silence prefill included) where speech ends; -1 = none
    int count = 0;
    bool reached_limit = false;  // stopped at max_frames without an end of speech
};

// Called with each generated delayed row and the end-of-speech frame once known (-1 before).
using Zonos3RowCallback = std::function<void(const int32_t * row, int eos_frame)>;

// ZONOS3's decode loop: cold start (codebooks whose delayed stream has not started are forced to the pad), an
// 11-row end-of-speech drain after the first EOA, speaker CFG on the post-softcap logits when `twin` is set.
// max_frames counts generated rows, the drain included.
Zonos3Codes generate_codes(const Zonos3Hparams & hp, Zonos3Backbone & backbone, const Zonos3Prompt & prompt,
                           const Zonos3Prompt * twin, float guidance_scale, int max_frames,
                           const Zonos3SamplingOptions & sampling, const Zonos3RowCallback & on_row);

}  // namespace engine::models::zonos3
