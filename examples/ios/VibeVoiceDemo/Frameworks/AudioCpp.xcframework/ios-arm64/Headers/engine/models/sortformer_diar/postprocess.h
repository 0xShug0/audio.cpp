#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/models/sortformer_diar/assets.h"
#include "engine/models/sortformer_diar/types.h"

#include <string>
#include <vector>

namespace engine::models::sortformer_diar {

SortformerPostprocessConfig parse_sortformer_postprocess_config(const runtime::SessionOptions & options);

SortformerFixedContextContract parse_sortformer_fixed_context_contract(
    const runtime::SessionOptions & options,
    const SortformerAssets & assets);
SortformerFixedContextContract make_sortformer_fixed_context_contract_for_samples(
    int64_t sample_count,
    const SortformerAssets & assets);

void emit_sortformer_timings(const SortformerRunTimings & timings);

std::vector<runtime::SpeakerTurn> decode_sortformer_speaker_turns(
    const std::vector<float> & probabilities,
    int64_t frames,
    int64_t valid_frames,
    int64_t num_speakers,
    int64_t frame_step_samples,
    const SortformerPostprocessConfig & config);

}  // namespace engine::models::sortformer_diar
