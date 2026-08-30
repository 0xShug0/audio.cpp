#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace engine::audio {

enum class SoxrResampleProfile {
    QualityOnly,
    ExplicitFloat32Runtime,
};

enum class SoxrOutputLengthPolicy {
    ActualOutput,
    ClampToExpected,
    ExactExpected,
};

struct SoxrResampleOptions {
    SoxrResampleProfile profile = SoxrResampleProfile::QualityOnly;
    SoxrOutputLengthPolicy output_length_policy = SoxrOutputLengthPolicy::ActualOutput;
    size_t output_padding = 0;
    bool require_full_input = false;
    bool reject_empty_output = false;
    const char * warning_context = "audio";
    const char * fallback_description = "fallback resampling";
};

std::optional<std::vector<float>> try_resample_mono_soxr(
    const std::vector<float> & mono_samples,
    int source_sample_rate_hz,
    int target_sample_rate_hz,
    const SoxrResampleOptions & options);

std::vector<float> resample_mono_soxr_or_linear(
    const std::vector<float> & mono_samples,
    int source_sample_rate_hz,
    int target_sample_rate_hz,
    const SoxrResampleOptions & options);

// Same contract as resample_mono_soxr_or_linear, but the fallback is the
// in-tree windowed-sinc resampler at playback width rather than a two-tap
// linear interpolator, so output quality does not depend on whether an optional
// system library happens to be installed. Use this on any path whose output a
// listener will hear. The requested output_length_policy is applied to the
// fallback result too, which the linear fallback does not do.
std::vector<float> resample_mono_soxr_or_sinc(
    const std::vector<float> & mono_samples,
    int source_sample_rate_hz,
    int target_sample_rate_hz,
    const SoxrResampleOptions & options);

std::vector<float> resample_mono_linear(
    const std::vector<float> & mono_samples,
    int source_sample_rate_hz,
    int target_sample_rate_hz);

enum class TorchaudioSincHannKernelMode {
    Float64ComputationStoredAsFloat32,
    Float32ComputationStoredAsFloat32,
    Float64ComputationStoredAsFloat64,
};

enum class TorchaudioSincHannAccumulation {
    Float32,
    Float64,
};

struct TorchaudioSincHannResampleOptions {
    // 6 is torchaudio.transforms.Resample's own default and is deliberately
    // kept: the ~36 call sites that take it are feature-extraction paths whose
    // job is bit-parity with a Python reference, and widening the kernel there
    // would change every one of their model inputs. It is the wrong width for
    // audible output — a 44.1 -> 48 -> 44.1 kHz music round trip measures
    // 74.1 dB at width 6 against 130.5 dB at width 64, and the worst alias
    // image from a 19.5 kHz tone is -13.8 dBc against -80.8 dBc. Playback paths
    // should ask for torchaudio_sinc_hann_playback_options() instead.
    int64_t lowpass_filter_width = 6;
    double rolloff = 0.99;
    TorchaudioSincHannKernelMode kernel_mode = TorchaudioSincHannKernelMode::Float64ComputationStoredAsFloat32;
    TorchaudioSincHannAccumulation accumulation = TorchaudioSincHannAccumulation::Float64;
};

TorchaudioSincHannResampleOptions torchaudio_sinc_hann_float32_options();

// Width 64. The measured knee for music: 44.1 -> 48 -> 44.1 kHz round trip is
// 74.1 dB at width 6, 101.1 dB at 16, 130.5 dB at 64, 129.6 dB at 256, so 64
// buys 56 dB over the default and 256 buys nothing further. Cost scales roughly
// linearly with width.
TorchaudioSincHannResampleOptions torchaudio_sinc_hann_playback_options();

std::vector<float> resample_mono_torchaudio_sinc_hann(
    const std::vector<float> & mono_samples,
    int source_sample_rate_hz,
    int target_sample_rate_hz,
    const TorchaudioSincHannResampleOptions & options = {});

}  // namespace engine::audio
