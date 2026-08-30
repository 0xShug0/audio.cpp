#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"

#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

double to_db(double ratio) {
    return 20.0 * std::log10(std::max(ratio, 1e-300));
}

std::vector<float> make_sine(size_t count, double freq_hz, double rate_hz, double amplitude) {
    std::vector<float> out(count, 0.0F);
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<float>(
            amplitude * std::sin(2.0 * kPi * freq_hz * static_cast<double>(i) / rate_hz));
    }
    return out;
}

// Coherent single-bin DFT amplitude. Every call below arranges an integer
// number of cycles inside the analysis window, so no window function is needed
// and the result is the exact amplitude of that component.
double bin_amplitude(
    const std::vector<float> & samples,
    size_t begin,
    size_t count,
    double freq_hz,
    double rate_hz) {
    if (begin + count > samples.size()) {
        throw std::runtime_error("analysis window runs past the end of the signal");
    }
    double real = 0.0;
    double imaginary = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double angle = 2.0 * kPi * freq_hz * static_cast<double>(i) / rate_hz;
        real += static_cast<double>(samples[begin + i]) * std::cos(angle);
        imaginary -= static_cast<double>(samples[begin + i]) * std::sin(angle);
    }
    return 2.0 * std::sqrt(real * real + imaginary * imaginary) / static_cast<double>(count);
}

// Pink-tilted content built as a sum of sinusoids, so it is provably empty
// above the stated limit and the round trip below measures the resampler rather
// than the test signal's own filter skirt.
std::vector<float> make_music_like(size_t count, double rate_hz, double limit_hz) {
    constexpr int kPartials = 192;
    std::mt19937 rng(20260830U);
    std::uniform_real_distribution<double> phase_dist(0.0, 2.0 * kPi);
    std::vector<double> freq(kPartials, 0.0);
    std::vector<double> amplitude(kPartials, 0.0);
    std::vector<double> phase(kPartials, 0.0);
    for (int p = 0; p < kPartials; ++p) {
        const double t = static_cast<double>(p) / static_cast<double>(kPartials - 1);
        freq[p] = 30.0 * std::pow(limit_hz / 30.0, t);
        amplitude[p] = std::pow(30.0 / freq[p], 0.75);
        phase[p] = phase_dist(rng);
    }
    std::vector<float> out(count, 0.0F);
    double peak = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double time = static_cast<double>(i) / rate_hz;
        double value = 0.0;
        for (int p = 0; p < kPartials; ++p) {
            value += amplitude[p] * std::sin(2.0 * kPi * freq[p] * time + phase[p]);
        }
        out[i] = static_cast<float>(value);
        peak = std::max(peak, std::abs(value));
    }
    for (float & sample : out) {
        sample = static_cast<float>(static_cast<double>(sample) / peak * 0.5);
    }
    return out;
}

double round_trip_snr_db(
    const std::vector<float> & reference,
    const std::vector<float> & measured,
    size_t skip) {
    const size_t count = std::min(reference.size(), measured.size());
    engine::test::require(count > 2 * skip, "round trip produced too few samples to measure");
    double signal = 0.0;
    double noise = 0.0;
    for (size_t i = skip; i < count - skip; ++i) {
        const double r = static_cast<double>(reference[i]);
        const double d = static_cast<double>(measured[i]) - r;
        signal += r * r;
        noise += d * d;
    }
    return 10.0 * std::log10(signal / std::max(noise, 1e-300));
}

engine::audio::TorchaudioSincHannResampleOptions sinc_options(int64_t width) {
    engine::audio::TorchaudioSincHannResampleOptions options;
    options.lowpass_filter_width = width;
    return options;
}

std::filesystem::path scratch_dir() {
    const auto dir = std::filesystem::temp_directory_path() / "audiocpp_resample_quality_test";
    std::filesystem::create_directories(dir);
    return dir;
}

// F4.1. The audio-utility entry points (denoise, super-resolution) all read
// through read_wav_f32_as_mono_quality_resampled, and three of the five target
// rates are decimations. 48 -> 16 kHz is an exact 3:1 ratio, which degenerates
// the old two-tap linear interpolator into plain sample dropping: every third
// sample is taken and nothing is filtered, so a 12 kHz tone reappears at
// |16000 - 12000| = 4 kHz at full amplitude.
//
// The file is written as float32 rather than 16-bit PCM deliberately: a 16-bit
// container would put a ~-96 dBc quantisation floor under the measurement and
// hide the very thing being asserted.
void test_utility_path_rejects_decimation_alias() {
    const auto dir = scratch_dir();
    const auto path = dir / "tone_12k_48k.wav";
    // One second at 48 kHz: exactly 12000 cycles of the 12 kHz tone.
    const auto tone = make_sine(48000, 12000.0, 48000.0, 1.0);
    engine::audio::WavWriteOptions options;
    options.format = engine::audio::WavSampleFormat::Float32;
    engine::audio::write_wav(path, 48000, 1, tone, options);

    const auto decimated = engine::audio::read_wav_f32_as_mono_quality_resampled(path, 16000);
    engine::test::require(decimated.size() >= 14000, "16 kHz decimation returned too few samples");

    // 0.75 s of the interior: exactly 3000 cycles of 4 kHz at 16 kHz, and clear
    // of the kernel's edge transients at both ends.
    const double alias = bin_amplitude(decimated, 2000, 12000, 4000.0, 16000.0);
    const double alias_dbc = to_db(alias);

    // Threshold rationale. Measured on this exact path: the old
    // resample_mono_linear route leaves the alias at 0.00 dBc (full amplitude,
    // because 3:1 makes it a pure decimation); the in-tree windowed sinc at the
    // framework default width of 6 gives -55.1 dBc; at playback width 64 it
    // gives -117.1 dBc; libsoxr, when installed, gives -288 dBc. -60 dBc
    // therefore passes only for a genuinely anti-aliased resampler, fails the
    // old linear path by 60 dB, and also fails a regression back to the narrow
    // default kernel -- while staying insensitive to whether libsoxr happens to
    // be present on the build machine.
    constexpr double kMaxAliasDbc = -60.0;
    if (alias_dbc > kMaxAliasDbc) {
        std::ostringstream oss;
        oss << "48 -> 16 kHz decimation folded 12 kHz back to 4 kHz at " << alias_dbc
            << " dBc, which is above the " << kMaxAliasDbc << " dBc limit";
        throw std::runtime_error(oss.str());
    }

    std::filesystem::remove(path);
}

// The same assertion one level down, on the shared helper, so a caller that
// reaches for resample_mono_soxr_or_sinc directly is covered too.
void test_soxr_or_sinc_fallback_is_anti_aliased() {
    const auto tone = make_sine(48000, 12000.0, 48000.0, 1.0);
    engine::audio::SoxrResampleOptions options;
    options.warning_context = "resample quality test";
    const auto decimated = engine::audio::resample_mono_soxr_or_sinc(tone, 48000, 16000, options);
    const double alias_dbc = to_db(bin_amplitude(decimated, 2000, 12000, 4000.0, 16000.0));
    engine::test::require(
        alias_dbc <= -60.0,
        "resample_mono_soxr_or_sinc left a 4 kHz alias at " + std::to_string(alias_dbc) + " dBc");

    // The linear helper is the hazard this replaced; assert the gap is real so
    // the test fails loudly if the two are ever wired together again.
    const auto linear = engine::audio::resample_mono_linear(tone, 48000, 16000);
    const double linear_dbc = to_db(bin_amplitude(linear, 2000, 12000, 4000.0, 16000.0));
    engine::test::require(
        linear_dbc - alias_dbc > 50.0,
        "resample_mono_soxr_or_sinc is no better than resample_mono_linear here");
}

// F4.2. The fallback used to ignore output_length_policy entirely, so an
// output could be one sample shorter or longer purely on whether libsoxr
// happened to be installed. mixing.cpp throws on exactly that mismatch.
void test_fallback_honours_output_length_policy() {
    const auto source = make_music_like(11025, 44100.0, 16000.0);
    engine::audio::SoxrResampleOptions options;
    options.output_length_policy = engine::audio::SoxrOutputLengthPolicy::ExactExpected;
    options.warning_context = "resample quality test";
    const auto resampled = engine::audio::resample_mono_soxr_or_sinc(source, 44100, 48000, options);
    const size_t expected = static_cast<size_t>(
        std::ceil(static_cast<double>(source.size()) * 48000.0 / 44100.0));
    engine::test::require_eq(resampled.size(), expected, "ExactExpected output length");
}

// F4.6. The tiers, with the numbers that justify keeping the framework default
// at 6 while routing playback through 64.
void test_resampler_tier_round_trip_snr() {
    // 1 s is enough for a stable measurement and keeps the width-64 pass in
    // the low milliseconds even in a debug build.
    const size_t count = 44100;
    const auto source = make_music_like(count, 44100.0, 16000.0);
    constexpr size_t kSkip = 4096;

    const auto linear_up = engine::audio::resample_mono_linear(source, 44100, 48000);
    const auto linear_back = engine::audio::resample_mono_linear(linear_up, 48000, 44100);
    const double linear_snr = round_trip_snr_db(source, linear_back, kSkip);

    const auto narrow_up =
        engine::audio::resample_mono_torchaudio_sinc_hann(source, 44100, 48000, sinc_options(6));
    const auto narrow_back =
        engine::audio::resample_mono_torchaudio_sinc_hann(narrow_up, 48000, 44100, sinc_options(6));
    const double narrow_snr = round_trip_snr_db(source, narrow_back, kSkip);

    const auto playback = engine::audio::torchaudio_sinc_hann_playback_options();
    const auto wide_up =
        engine::audio::resample_mono_torchaudio_sinc_hann(source, 44100, 48000, playback);
    const auto wide_back =
        engine::audio::resample_mono_torchaudio_sinc_hann(wide_up, 48000, 44100, playback);
    const double wide_snr = round_trip_snr_db(source, wide_back, kSkip);

    // Measured on this signal: linear 46.3 dB, width 6 60.5 dB, width 64
    // 122.9 dB. The 53 dB divider sits about 7 dB clear of the two tiers it
    // separates, and the 110 dB floor is 13 dB under what width 64 delivers.
    engine::test::require(
        linear_snr < 53.0,
        "resample_mono_linear round trip measured " + std::to_string(linear_snr) +
            " dB, which is unexpectedly good -- the tier ordering assumed here may no longer hold");
    engine::test::require(
        narrow_snr > 53.0,
        "sinc width 6 round trip measured only " + std::to_string(narrow_snr) + " dB");
    engine::test::require(
        wide_snr > 110.0,
        "playback-width sinc round trip measured only " + std::to_string(wide_snr) + " dB");
    engine::test::require(
        wide_snr - narrow_snr > 45.0,
        "playback width bought only " + std::to_string(wide_snr - narrow_snr) +
            " dB over the framework default");
    engine::test::require(
        narrow_snr - linear_snr > 8.0,
        "sinc width 6 is not measurably better than linear interpolation");
}

// The framework default is deliberately left at torchaudio's own value of 6 so
// the ~36 feature-extraction call sites keep bit-parity with their Python
// references. Pin both so a future change is a conscious one.
void test_resampler_option_defaults() {
    const engine::audio::TorchaudioSincHannResampleOptions defaults;
    engine::test::require_eq(defaults.lowpass_filter_width, 6, "framework default filter width");
    engine::test::require_eq(
        engine::audio::torchaudio_sinc_hann_playback_options().lowpass_filter_width,
        64,
        "playback filter width");
    engine::test::require_eq(
        engine::audio::torchaudio_sinc_hann_float32_options().lowpass_filter_width,
        6,
        "float32 parity options keep the framework default width");
}

}  // namespace

int main() {
    try {
        test_utility_path_rejects_decimation_alias();
        test_soxr_or_sinc_fallback_is_anti_aliased();
        test_fallback_honours_output_length_policy();
        test_resampler_tier_round_trip_snr();
        test_resampler_option_defaults();
        std::cout << "audio_resample_quality_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "audio_resample_quality_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
