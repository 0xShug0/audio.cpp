#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace engine::models::dots_tts {

struct DotsFbankOutput {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t dims = 0;
};

struct DotsPreparedReferenceAudio {
    std::vector<float> waveform_16k;
    std::vector<float> waveform_vocoder_rate;
    DotsFbankOutput speaker_fbank;
};

DotsPreparedReferenceAudio prepare_dots_reference_audio(
    const runtime::AudioBuffer & audio,
    int vocoder_sample_rate,
    int64_t samples_per_patch,
    std::optional<float> max_duration_seconds = std::nullopt);

std::vector<float> prepare_dots_edit_source_audio(
    const runtime::AudioBuffer & audio,
    int vocoder_sample_rate,
    int64_t samples_per_patch);

DotsFbankOutput compute_dots_speaker_fbank_16k(const std::vector<float> & waveform_16k);

}  // namespace engine::models::dots_tts
