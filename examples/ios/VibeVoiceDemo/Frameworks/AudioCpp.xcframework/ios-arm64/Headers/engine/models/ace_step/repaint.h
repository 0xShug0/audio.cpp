#pragma once

#include "engine/framework/runtime/session.h"

namespace engine::models::ace_step {

struct AceStepRepaintSpliceSource {
    runtime::AudioBuffer audio;
    float start_seconds = 0.0F;
    float end_seconds = 0.0F;
};

void ace_step_apply_repaint_waveform_splice(
    runtime::AudioBuffer & decoded,
    const AceStepRepaintSpliceSource & source,
    float crossfade_seconds);

}  // namespace engine::models::ace_step
