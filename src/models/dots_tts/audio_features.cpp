#include "engine/models/dots_tts/audio_features.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/reference_audio_frontend.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace engine::models::dots_tts {
namespace {

constexpr float kEditEdgeSilenceMs = 250.0F;
constexpr float kEditEdgeSilenceTopDb = 30.0F;

std::vector<float> validated_mono_samples(const runtime::AudioBuffer & audio, const char * role) {
    if (audio.channels <= 0) {
        throw std::runtime_error(std::string("DotTTS ") + role + " audio channel count must be positive");
    }
    if (audio.sample_rate <= 0) {
        throw std::runtime_error(std::string("DotTTS ") + role + " audio sample rate must be positive");
    }
    if (audio.samples.empty()) {
        throw std::runtime_error(std::string("DotTTS ") + role + " audio must not be empty");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error(std::string("DotTTS ") + role + " audio sample count must be divisible by channel count");
    }
    return audio.channels == 1
        ? audio.samples
        : engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
}

void normalize_edit_edge_silence(std::vector<float> & waveform, int sample_rate) {
    if (waveform.empty()) {
        return;
    }
    const size_t target_samples = static_cast<size_t>(std::llround(
        static_cast<double>(sample_rate) * static_cast<double>(kEditEdgeSilenceMs) / 1000.0));
    if (target_samples == 0) {
        return;
    }
    float peak = 0.0F;
    for (const float sample : waveform) {
        peak = std::max(peak, std::fabs(sample));
    }
    if (peak <= 0.0F) {
        waveform.assign(target_samples, 0.0F);
        return;
    }
    const float threshold = peak * std::pow(10.0F, -kEditEdgeSilenceTopDb / 20.0F);
    size_t first = 0;
    while (first < waveform.size() && std::fabs(waveform[first]) <= threshold) {
        ++first;
    }
    if (first == waveform.size()) {
        waveform.assign(target_samples, 0.0F);
        return;
    }
    size_t last = waveform.size() - 1;
    while (last > first && std::fabs(waveform[last]) <= threshold) {
        --last;
    }
    const size_t leading = first;
    const size_t trailing = waveform.size() - last - 1;
    if (leading < target_samples) {
        waveform.insert(waveform.begin(), target_samples - leading, 0.0F);
    } else if (leading > target_samples) {
        waveform.erase(waveform.begin(), waveform.begin() + static_cast<std::ptrdiff_t>(leading - target_samples));
    }
    const size_t current_trailing = std::min(trailing, waveform.size());
    if (current_trailing < target_samples) {
        waveform.insert(waveform.end(), target_samples - current_trailing, 0.0F);
    } else if (current_trailing > target_samples) {
        waveform.resize(waveform.size() - (current_trailing - target_samples));
    }
}

void pad_to_multiple(std::vector<float> & waveform, int64_t multiple) {
    if (multiple <= 0) {
        throw std::runtime_error("DotTTS audio padding multiple must be positive");
    }
    const int64_t samples = static_cast<int64_t>(waveform.size());
    const int64_t remainder = samples % multiple;
    if (remainder == 0) {
        return;
    }
    waveform.resize(static_cast<size_t>(samples + multiple - remainder), 0.0F);
}

}  // namespace

DotsFbankOutput compute_dots_speaker_fbank_16k(const std::vector<float> & waveform_16k) {
    auto features = engine::audio::ReferenceAudioFrontend::campplus_fbank_16k(waveform_16k);
    return {std::move(features.values), features.frames, features.feature_dim};
}

DotsPreparedReferenceAudio prepare_dots_reference_audio(
    const runtime::AudioBuffer & audio,
    int vocoder_sample_rate,
    int64_t samples_per_patch,
    std::optional<float> max_duration_seconds) {
    if (samples_per_patch <= 0) {
        throw std::runtime_error("DotTTS reference audio samples_per_patch must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error("DotTTS reference audio channel count must be positive");
    }
    if (audio.sample_rate <= 0 || vocoder_sample_rate <= 0) {
        throw std::runtime_error("DotTTS reference resampling requires positive sample rates");
    }
    if (audio.samples.empty()) {
        throw std::runtime_error("DotTTS reference audio must not be empty");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("DotTTS reference audio sample count must be divisible by channel count");
    }
    auto mono = audio.channels == 1
        ? audio.samples
        : engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    if (max_duration_seconds.has_value()) {
        if (!std::isfinite(*max_duration_seconds) || *max_duration_seconds < 0.0F) {
            throw std::runtime_error("DotTTS reference_duration_sec must be finite and non-negative");
        }
        engine::audio::truncate_samples_to_count(
            mono,
            static_cast<size_t>(*max_duration_seconds * static_cast<float>(audio.sample_rate)));
    }
    mono = engine::audio::trim_mono_librosa_effects(mono);
    DotsPreparedReferenceAudio output;
    const auto resample_options = engine::audio::torchaudio_sinc_hann_float32_options();
    output.waveform_vocoder_rate = audio.sample_rate == vocoder_sample_rate
        ? mono
        : engine::audio::resample_mono_torchaudio_sinc_hann(
            mono,
            audio.sample_rate,
            vocoder_sample_rate,
            resample_options);
    engine::debug::trace_log_scalar("dots_tts.reference.trimmed_samples", static_cast<int64_t>(mono.size()));
    engine::debug::trace_log_scalar("dots_tts.reference.resampled_samples", static_cast<int64_t>(output.waveform_vocoder_rate.size()));
    output.waveform_16k = vocoder_sample_rate == 16000
        ? output.waveform_vocoder_rate
        : engine::audio::resample_mono_torchaudio_sinc_hann(
            output.waveform_vocoder_rate,
            vocoder_sample_rate,
            16000,
            resample_options);
    output.speaker_fbank = compute_dots_speaker_fbank_16k(output.waveform_16k);
    const int64_t vocoder_samples = static_cast<int64_t>(output.waveform_vocoder_rate.size());
    const int64_t padded_samples = ((vocoder_samples + samples_per_patch - 1) / samples_per_patch) * samples_per_patch;
    output.waveform_vocoder_rate.resize(static_cast<size_t>(padded_samples), 0.0F);
    engine::debug::trace_log_scalar("dots_tts.reference.padded_samples", padded_samples);
    return output;
}

std::vector<float> prepare_dots_edit_source_audio(
    const runtime::AudioBuffer & audio,
    int vocoder_sample_rate,
    int64_t samples_per_patch) {
    if (audio.sample_rate <= 0 || vocoder_sample_rate <= 0) {
        throw std::runtime_error("DotTTS edit source resampling requires positive sample rates");
    }
    auto mono = validated_mono_samples(audio, "edit source");
    engine::audio::TorchaudioSincHannResampleOptions options;
    options.lowpass_filter_width = 128;
    options.rolloff = 0.95;
    options.kernel_mode = engine::audio::TorchaudioSincHannKernelMode::Float32ComputationStoredAsFloat32;
    options.accumulation = engine::audio::TorchaudioSincHannAccumulation::Float32;
    if (audio.sample_rate != vocoder_sample_rate) {
        mono = engine::audio::resample_mono_torchaudio_sinc_hann(
            mono,
            audio.sample_rate,
            vocoder_sample_rate,
            options);
    }
    normalize_edit_edge_silence(mono, vocoder_sample_rate);
    pad_to_multiple(mono, samples_per_patch);
    engine::debug::trace_log_scalar("dots_tts.edit.source_samples", static_cast<int64_t>(mono.size()));
    return mono;
}

}  // namespace engine::models::dots_tts
