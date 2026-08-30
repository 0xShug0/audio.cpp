#pragma once

#include "engine/framework/audio/wav_reader.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace engine::audio {

enum class MonoMixAccumulation {
    Float32,
    Float64,
};

std::vector<float> mixdown_interleaved_to_mono_average(
    const std::vector<float> & interleaved_samples,
    int channel_count,
    MonoMixAccumulation accumulation = MonoMixAccumulation::Float32);

std::vector<float> duplicate_mono_to_interleaved_channels(
    const std::vector<float> & mono_samples,
    int target_channel_count);

std::vector<float> deinterleave_to_planar_channels(
    const std::vector<float> & interleaved_samples,
    int channel_count);

std::vector<float> interleave_planar_channels(
    const std::vector<float> & planar_samples,
    int channel_count,
    int64_t frame_count);

std::vector<float> extract_interleaved_channel(
    const std::vector<float> & interleaved_samples,
    int channel_count,
    int channel_index);

std::vector<float> convert_wav_to_mono_linear_resampled(
    const WavData & wav,
    int target_sample_rate_hz);

std::vector<float> convert_interleaved_audio_to_mono_linear_resampled(
    const std::vector<float> & interleaved_samples,
    int sample_rate_hz,
    int channel_count,
    int target_sample_rate_hz);

std::vector<float> read_wav_f32_as_mono_linear_resampled(
    const std::filesystem::path & path,
    int target_sample_rate_hz);

// Anti-aliased equivalents of the three helpers above. The `_linear_` versions
// call a two-tap interpolator with no decimation filter, so a 48 -> 16 kHz
// conversion folds a 12 kHz tone back to 4 kHz at -9.5 dBc; these route through
// soxr, or the in-tree windowed sinc at playback width, and measure -126.7 dBc
// for the same job. Use them wherever the audio is destined for a listener or
// for a model that is expected to see a clean band. The `_linear_` versions are
// kept unchanged for the call sites that assert parity against a reference
// implementation.
std::vector<float> convert_wav_to_mono_quality_resampled(
    const WavData & wav,
    int target_sample_rate_hz);

std::vector<float> convert_interleaved_audio_to_mono_quality_resampled(
    const std::vector<float> & interleaved_samples,
    int sample_rate_hz,
    int channel_count,
    int target_sample_rate_hz);

std::vector<float> read_wav_f32_as_mono_quality_resampled(
    const std::filesystem::path & path,
    int target_sample_rate_hz);

}  // namespace engine::audio
