#pragma once

#include "engine/models/dettato/runtime.h"
#include <array>

namespace engine::models::dettato::detail {
struct NoteThresholds {
    float onset = 0.25f, second_onset = 0.3f, offset = 0.3f, frame = 0.4f;
    int low_frames = 2, onset_nms = 3;
};

// Six probability maps in upstream head order, each [frames, 88].
std::vector<Note> decode_notes(const std::array<std::vector<float>, 6> &maps, int64_t frames, double hop,
                               const NoteThresholds &thresholds);
} // namespace engine::models::dettato::detail
