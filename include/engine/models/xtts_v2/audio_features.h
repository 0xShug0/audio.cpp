#pragma once

#include "engine/framework/runtime/model.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine::models::xtts_v2 {

struct XttsV2MelFeatures {
    std::vector<float> values;  // channel-major [channels, frames]
    int64_t channels = 0;
    int64_t frames = 0;
};

struct XttsV2PreparedReference {
    std::vector<float> waveform_22050;
    std::vector<float> waveform_16000;
};

XttsV2PreparedReference prepare_xtts_v2_reference(const runtime::AudioBuffer & audio);

XttsV2MelFeatures compute_xtts_v2_conditioning_mel(
    const std::vector<float> & waveform_22050,
    const std::vector<float> & mel_stats,
    size_t threads = 0);

XttsV2MelFeatures compute_xtts_v2_speaker_mel(
    const std::vector<float> & waveform_16000,
    const std::vector<float> & window,
    const std::vector<float> & mel_filterbank,
    size_t threads = 0);

}  // namespace engine::models::xtts_v2
