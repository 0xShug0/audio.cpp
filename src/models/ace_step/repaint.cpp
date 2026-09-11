#include "engine/models/ace_step/repaint.h"

#include "engine/models/ace_step/types.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::ace_step {

void ace_step_apply_repaint_waveform_splice(
    runtime::AudioBuffer & decoded,
    const AceStepRepaintSpliceSource & source,
    float crossfade_seconds) {
    if (decoded.sample_rate != kAceStepAudioSampleRate || source.audio.sample_rate != kAceStepAudioSampleRate ||
        decoded.channels != kAceStepAudioChannels || source.audio.channels != kAceStepAudioChannels) {
        throw std::runtime_error("ACE-Step repaint waveform splice requires stereo 48 kHz audio");
    }
    const int64_t decoded_frames =
        static_cast<int64_t>(decoded.samples.size() / static_cast<size_t>(decoded.channels));
    const int64_t source_frames =
        static_cast<int64_t>(source.audio.samples.size() / static_cast<size_t>(source.audio.channels));
    const int64_t frames = std::min(decoded_frames, source_frames);
    if (frames <= 0) {
        throw std::runtime_error("ACE-Step repaint waveform splice requires non-empty audio");
    }

    int64_t start_frame =
        static_cast<int64_t>(static_cast<double>(source.start_seconds) * static_cast<double>(kAceStepAudioSampleRate));
    int64_t end_frame =
        static_cast<int64_t>(static_cast<double>(source.end_seconds) * static_cast<double>(kAceStepAudioSampleRate));
    start_frame = std::max<int64_t>(0, std::min<int64_t>(start_frame, frames));
    end_frame = std::max<int64_t>(start_frame, std::min<int64_t>(end_frame, frames));
    if (start_frame == 0 && end_frame >= frames) {
        return;
    }

    const int64_t crossfade_frames =
        static_cast<int64_t>(static_cast<double>(crossfade_seconds) * static_cast<double>(kAceStepAudioSampleRate));
    const int64_t fade_start = std::max<int64_t>(0, start_frame - std::max<int64_t>(0, crossfade_frames));
    const int64_t fade_end = std::min<int64_t>(frames, end_frame + std::max<int64_t>(0, crossfade_frames));
    const int64_t left_ramp_len = start_frame - fade_start;
    const int64_t right_ramp_len = fade_end - end_frame;

    if (fade_start > 0) {
        std::copy(
            source.audio.samples.begin(),
            source.audio.samples.begin() + static_cast<std::ptrdiff_t>(fade_start * kAceStepAudioChannels),
            decoded.samples.begin());
    }
    for (int64_t frame = fade_start; frame < start_frame; ++frame) {
        const float mask = static_cast<float>(frame - fade_start + 1) / static_cast<float>(left_ramp_len + 1);
        const size_t base = static_cast<size_t>(frame * kAceStepAudioChannels);
        for (int channel = 0; channel < kAceStepAudioChannels; ++channel) {
            const size_t index = base + static_cast<size_t>(channel);
            decoded.samples[index] =
                mask * decoded.samples[index] + (1.0F - mask) * source.audio.samples[index];
        }
    }
    for (int64_t frame = end_frame; frame < fade_end; ++frame) {
        const float mask = 1.0F - static_cast<float>(frame - end_frame + 1) / static_cast<float>(right_ramp_len + 1);
        const size_t base = static_cast<size_t>(frame * kAceStepAudioChannels);
        for (int channel = 0; channel < kAceStepAudioChannels; ++channel) {
            const size_t index = base + static_cast<size_t>(channel);
            decoded.samples[index] =
                mask * decoded.samples[index] + (1.0F - mask) * source.audio.samples[index];
        }
    }
    if (fade_end < frames) {
        const auto source_begin =
            source.audio.samples.begin() + static_cast<std::ptrdiff_t>(fade_end * kAceStepAudioChannels);
        const auto source_end =
            source.audio.samples.begin() + static_cast<std::ptrdiff_t>(frames * kAceStepAudioChannels);
        const auto decoded_begin =
            decoded.samples.begin() + static_cast<std::ptrdiff_t>(fade_end * kAceStepAudioChannels);
        std::copy(source_begin, source_end, decoded_begin);
    }
}

}  // namespace engine::models::ace_step
