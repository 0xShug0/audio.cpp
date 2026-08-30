#include "engine/framework/audio/mixing.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/audio/waveform_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace engine::audio {
namespace {

int64_t frame_count(const WavData & audio) {
    if (audio.channels <= 0) {
        throw std::runtime_error("audio mix input has invalid channel count");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("audio mix input samples are not divisible by channel count");
    }
    return static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels));
}

std::vector<float> resample_interleaved_to_rate(
    const std::vector<float> & samples,
    int source_rate,
    int channels,
    int target_rate) {
    if (source_rate == target_rate) {
        return samples;
    }
    const int64_t frames = static_cast<int64_t>(samples.size() / static_cast<size_t>(channels));
    std::vector<std::vector<float>> channel_outputs;
    channel_outputs.reserve(static_cast<size_t>(channels));
    int64_t output_frames = -1;
    for (int ch = 0; ch < channels; ++ch) {
        std::vector<float> mono(static_cast<size_t>(frames), 0.0F);
        for (int64_t frame = 0; frame < frames; ++frame) {
            mono[static_cast<size_t>(frame)] = samples[static_cast<size_t>(frame * channels + ch)];
        }
        SoxrResampleOptions options;
        options.profile = SoxrResampleProfile::QualityOnly;
        options.output_length_policy = SoxrOutputLengthPolicy::ExactExpected;
        options.warning_context = "audio mix";
        options.fallback_description = "windowed-sinc resampling";
        // This is the audible mix output. The old fallback was a two-tap linear
        // interpolator, which round-trips music at 41 dB SNR against 130 dB for
        // the windowed sinc, and which ignored output_length_policy entirely —
        // it sized with llround where soxr sizes with ceil, so a channel could
        // come back one sample short and trip the length check below purely on
        // whether libsoxr happened to be installed.
        auto resampled = resample_mono_soxr_or_sinc(mono, source_rate, target_rate, options);
        if (output_frames < 0) {
            output_frames = static_cast<int64_t>(resampled.size());
        } else if (output_frames != static_cast<int64_t>(resampled.size())) {
            throw std::runtime_error("audio mix resampled channels produced inconsistent lengths");
        }
        channel_outputs.push_back(std::move(resampled));
    }
    std::vector<float> out(static_cast<size_t>(output_frames * channels), 0.0F);
    for (int64_t frame = 0; frame < output_frames; ++frame) {
        for (int ch = 0; ch < channels; ++ch) {
            out[static_cast<size_t>(frame * channels + ch)] = channel_outputs[static_cast<size_t>(ch)][static_cast<size_t>(frame)];
        }
    }
    return out;
}

std::vector<float> adapt_audio(
    const WavData & input,
    int target_rate,
    int target_channels,
    int64_t target_frames) {
    if (input.sample_rate <= 0 || target_rate <= 0 || target_channels <= 0) {
        throw std::runtime_error("audio mix received invalid sample rate or channel count");
    }
    std::vector<float> samples = resample_interleaved_to_rate(
        input.samples,
        input.sample_rate,
        input.channels,
        target_rate);
    int channels = input.channels;
    if (channels != target_channels) {
        if (target_channels == 1) {
            samples = mixdown_interleaved_to_mono_average(samples, channels);
        } else if (channels == 1) {
            samples = duplicate_mono_to_interleaved_channels(samples, target_channels);
        } else {
            auto mono = mixdown_interleaved_to_mono_average(samples, channels);
            samples = duplicate_mono_to_interleaved_channels(mono, target_channels);
        }
        channels = target_channels;
    }
    const size_t target_samples = static_cast<size_t>(target_frames * target_channels);
    if (samples.size() < target_samples) {
        samples.resize(target_samples, 0.0F);
    } else if (samples.size() > target_samples) {
        samples.resize(target_samples);
    }
    return samples;
}

void apply_activity_gate_in_place(
    std::vector<float> & samples,
    const std::vector<float> & reference,
    int channels,
    int sample_rate,
    float threshold_dbfs,
    double window_seconds,
    double margin_seconds,
    double fade_seconds) {
    if (channels <= 0 || sample_rate <= 0 || window_seconds <= 0.0 || margin_seconds < 0.0 || fade_seconds < 0.0) {
        throw std::runtime_error("audio activity gate received invalid parameters");
    }
    if (samples.size() != reference.size() || samples.size() % static_cast<size_t>(channels) != 0) {
        throw std::runtime_error("audio activity gate requires matching input and reference shapes");
    }
    const int64_t frames = static_cast<int64_t>(samples.size() / static_cast<size_t>(channels));
    if (frames <= 0) {
        return;
    }
    const int64_t window_frames =
        std::max<int64_t>(1, static_cast<int64_t>(std::llround(window_seconds * static_cast<double>(sample_rate))));
    const int64_t margin_frames =
        std::max<int64_t>(0, static_cast<int64_t>(std::llround(margin_seconds * static_cast<double>(sample_rate))));
    const int64_t fade_frames =
        std::max<int64_t>(0, static_cast<int64_t>(std::llround(fade_seconds * static_cast<double>(sample_rate))));
    const double threshold = std::pow(10.0, static_cast<double>(threshold_dbfs) / 20.0);
    const double threshold_energy = threshold * threshold;
    std::vector<uint8_t> active(static_cast<size_t>(frames), 0);
    for (int64_t window_start = 0; window_start < frames; window_start += window_frames) {
        const int64_t window_end = std::min(frames, window_start + window_frames);
        double energy = 0.0;
        int64_t count = 0;
        for (int64_t frame = window_start; frame < window_end; ++frame) {
            const size_t base = static_cast<size_t>(frame * channels);
            for (int channel = 0; channel < channels; ++channel) {
                const float sample = reference[base + static_cast<size_t>(channel)];
                energy += static_cast<double>(sample) * static_cast<double>(sample);
                ++count;
            }
        }
        const double mean_energy = count > 0 ? energy / static_cast<double>(count) : 0.0;
        if (mean_energy > threshold_energy) {
            const int64_t begin = std::max<int64_t>(0, window_start - margin_frames);
            const int64_t end = std::min<int64_t>(frames, window_end + margin_frames);
            std::fill(
                active.begin() + static_cast<std::ptrdiff_t>(begin),
                active.begin() + static_cast<std::ptrdiff_t>(end),
                static_cast<uint8_t>(1));
        }
    }

    std::vector<int64_t> prev_active(static_cast<size_t>(frames), -1);
    std::vector<int64_t> next_active(static_cast<size_t>(frames), frames);
    int64_t prev = -1;
    for (int64_t frame = 0; frame < frames; ++frame) {
        if (active[static_cast<size_t>(frame)] != 0) {
            prev = frame;
        }
        prev_active[static_cast<size_t>(frame)] = prev;
    }
    int64_t next = frames;
    for (int64_t frame = frames; frame-- > 0;) {
        if (active[static_cast<size_t>(frame)] != 0) {
            next = frame;
        }
        next_active[static_cast<size_t>(frame)] = next;
    }

    for (int64_t frame = 0; frame < frames; ++frame) {
        float gate = active[static_cast<size_t>(frame)] != 0 ? 1.0F : 0.0F;
        if (gate == 0.0F && fade_frames > 0) {
            const int64_t distance_prev = prev_active[static_cast<size_t>(frame)] >= 0
                ? frame - prev_active[static_cast<size_t>(frame)]
                : fade_frames + 1;
            const int64_t distance_next = next_active[static_cast<size_t>(frame)] < frames
                ? next_active[static_cast<size_t>(frame)] - frame
                : fade_frames + 1;
            const int64_t distance = std::min(distance_prev, distance_next);
            if (distance <= fade_frames) {
                gate = 1.0F - static_cast<float>(distance) / static_cast<float>(fade_frames + 1);
            }
        }
        const size_t base = static_cast<size_t>(frame * channels);
        for (int channel = 0; channel < channels; ++channel) {
            samples[base + static_cast<size_t>(channel)] *= gate;
        }
    }
}

}  // namespace

WavData mix_audio_to_reference_shape(
    const std::vector<AudioMixInput> & inputs,
    bool normalize_peak) {
    if (inputs.empty()) {
        throw std::runtime_error("audio mix requires at least one input");
    }
    const WavData & reference = inputs.front().audio;
    const int64_t frames = frame_count(reference);
    if (reference.sample_rate <= 0 || reference.channels <= 0 || frames <= 0) {
        throw std::runtime_error("audio mix reference input is invalid");
    }
    WavData out;
    out.sample_rate = reference.sample_rate;
    out.channels = reference.channels;
    out.samples.assign(static_cast<size_t>(frames * reference.channels), 0.0F);
    for (const auto & input : inputs) {
        auto adapted = adapt_audio(input.audio, out.sample_rate, out.channels, frames);
        if (input.activity_reference.has_value()) {
            auto reference_activity = adapt_audio(*input.activity_reference, out.sample_rate, out.channels, frames);
            apply_activity_gate_in_place(
                adapted,
                reference_activity,
                out.channels,
                out.sample_rate,
                input.activity_threshold_dbfs,
                input.activity_window_seconds,
                input.activity_margin_seconds,
                input.activity_fade_seconds);
        }
        for (size_t i = 0; i < out.samples.size(); ++i) {
            out.samples[i] += adapted[i] * input.gain;
        }
    }
    if (normalize_peak) {
        normalize_peak_to_unit_range_and_clamp_in_place(out.samples);
    } else {
        for (float & sample : out.samples) {
            sample = std::clamp(sample, -1.0F, 1.0F);
        }
    }
    return out;
}

WavData read_and_mix_wavs_to_reference_shape(
    const std::vector<std::pair<std::filesystem::path, float>> & inputs,
    bool normalize_peak) {
    std::vector<AudioMixInput> loaded;
    loaded.reserve(inputs.size());
    for (const auto & [path, gain] : inputs) {
        loaded.push_back(AudioMixInput{read_wav_f32(path), gain});
    }
    return mix_audio_to_reference_shape(loaded, normalize_peak);
}

}  // namespace engine::audio
