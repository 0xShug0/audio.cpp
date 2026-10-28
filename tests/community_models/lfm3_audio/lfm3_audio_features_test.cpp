// The LFM3-Audio ASR frontend: frame and token counts, the log-mel features
// against a host fp64 DFT reference, and the 16 kHz mono conversion.
#include "engine/community_models/lfm3_audio/features.h"
#include "engine/framework/runtime/errors.h"
#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace lfm3 = engine::community_models::lfm3_audio;
using engine::test::require;
using engine::test::require_eq;

constexpr double kPi = 3.14159265358979323846;

struct Diff {
    double rel = 0.0, max_abs = 0.0;
};

Diff diff(const std::vector<float> & actual, const std::vector<double> & expected) {
    require_eq(actual.size(), expected.size(), "compared sizes");
    double num = 0.0, den = 0.0;
    Diff out;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double d = actual[i] - expected[i];
        num += d * d;
        den += expected[i] * expected[i];
        out.max_abs = std::max(out.max_abs, std::fabs(d));
    }
    out.rel = std::sqrt(num / den);
    return out;
}

double slaney_mel(double hz) {
    return hz < 1000.0 ? 3.0 * hz / 200.0 : 15.0 + std::log(hz / 1000.0) * 27.0 / std::log(6.4);
}

double slaney_hz(double mel) {
    return mel < 15.0 ? 200.0 * mel / 3.0 : 1000.0 * std::exp((mel - 15.0) * std::log(6.4) / 27.0);
}

// The upstream log-mel in fp64: right-pad to whole frames, periodic Hann, |DFT|^2 over 201 bins, Slaney mel (128, 0-8 kHz,
// area-normalized), clamp(log10, -12), (x + 4) / 4. Feature-major [128][frames].
std::vector<double> reference_features(const std::vector<float> & samples) {
    const int64_t n = static_cast<int64_t>(samples.size());
    const int64_t frames = 1 + (std::max<int64_t>(n - 400, 0) + 159) / 160;
    std::vector<double> padded(static_cast<size_t>(400 + (frames - 1) * 160), 0.0);
    std::copy(samples.begin(), samples.end(), padded.begin());

    std::vector<double> filt(130);
    for (int i = 0; i < 130; ++i) filt[i] = slaney_hz(slaney_mel(8000.0) * i / 129.0);
    std::vector<double> out(static_cast<size_t>(128 * frames));
    std::vector<double> power(201);
    for (int64_t t = 0; t < frames; ++t) {
        for (int k = 0; k <= 200; ++k) {
            double re = 0.0, im = 0.0;
            for (int i = 0; i < 400; ++i) {
                const double x = padded[static_cast<size_t>(t * 160 + i)] * (0.5 - 0.5 * std::cos(2.0 * kPi * i / 400.0));
                re += x * std::cos(2.0 * kPi * k * i / 400.0);
                im -= x * std::sin(2.0 * kPi * k * i / 400.0);
            }
            power[k] = re * re + im * im;
        }
        for (int m = 0; m < 128; ++m) {
            double mel = 0.0;
            for (int k = 0; k <= 200; ++k) {
                const double f = 8000.0 * k / 200.0;
                const double w = std::max(0.0, std::min((f - filt[m]) / (filt[m + 1] - filt[m]), (filt[m + 2] - f) / (filt[m + 2] - filt[m + 1])));
                mel += w * 2.0 / (filt[m + 2] - filt[m]) * power[k];
            }
            out[static_cast<size_t>(m * frames + t)] = (std::max(std::log10(mel), -12.0) + 4.0) / 4.0;
        }
    }
    return out;
}

// A chirp plus uniform noise (like the syn fixtures). The noise keeps every mel bin far above float32 STFT
// rounding: a pure tone leaves the top bins ~1e-12 below the peak, where any float32 STFT is noise.
std::vector<float> chirp(int64_t n) {
    std::vector<float> out(static_cast<size_t>(n));
    uint32_t state = 7;
    for (int64_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / 16000.0;
        state = state * 1664525u + 1013904223u;
        const double noise = static_cast<double>(state >> 8) / static_cast<double>(1u << 24) - 0.5;
        out[static_cast<size_t>(i)] = static_cast<float>(0.3 * std::sin(2.0 * kPi * (200.0 * t + 900.0 * t * t)) + 0.1 * noise);
    }
    return out;
}

void test_counts() {
    const std::vector<std::pair<int64_t, int64_t>> frames = {{0, 1}, {400, 1}, {401, 2}, {560, 2}, {561, 3}, {16000, 99}, {16001, 99}};
    for (const auto & [samples, expected] : frames) {
        require_eq(lfm3::lfm3_feature_frames(samples), expected, "frames for " + std::to_string(samples) + " samples");
    }
    for (int64_t t = 1; t <= 17; ++t) {
        require_eq(lfm3::lfm3_audio_tokens(t), (t + 7) / 8, "tokens for " + std::to_string(t) + " frames");
    }
    std::cout << "frame and token counts: ok\n";
}

void test_reference() {
    const lfm3::Lfm3FeatureExtractor extractor;
    for (const int64_t n : {0, 399, 400, 1000, 1123}) {
        const auto samples = chirp(n);
        const auto features = extractor.extract(samples, 2);
        require_eq(features.frames, lfm3::lfm3_feature_frames(n), "extracted frames");
        const auto d = diff(features.values, reference_features(samples));
        std::printf("features n=%lld (T=%lld) vs fp64 reference: rel %.3e max_abs %.3e\n", static_cast<long long>(n),
                    static_cast<long long>(features.frames), d.rel, d.max_abs);
        require(d.rel <= 1e-5 && d.max_abs <= 1e-4, "features match the fp64 reference");
    }
    // Digital silence clamps every bin: (-12 + 4) / 4.
    const auto silent = extractor.extract(std::vector<float>(800, 0.0f));
    require(std::all_of(silent.values.begin(), silent.values.end(), [](float v) { return v == -2.0f; }), "silence gives -2");
}

void test_mono_16k() {
    engine::runtime::AudioBuffer stereo;
    stereo.sample_rate = 16000;
    stereo.channels = 2;
    stereo.samples = {0.5f, -0.25f, 1.0f, 0.0f, -1.0f, -0.5f};
    require(lfm3::lfm3_audio_mono_16k(stereo) == std::vector<float>{0.125f, 0.5f, -0.75f}, "16 kHz stereo averages the channels");

    const auto rejects = [](const engine::runtime::AudioBuffer & audio) {
        try {
            (void)lfm3::lfm3_audio_mono_16k(audio);
        } catch (const engine::runtime::InvalidRequestError &) {
            return true;  // client audio: a request error, not an internal one
        }
        return false;
    };
    engine::runtime::AudioBuffer bad = stereo;
    bad.samples[3] = std::numeric_limits<float>::quiet_NaN();
    require(rejects(bad), "non-finite audio is rejected");
    bad.samples.clear();
    require(rejects(bad), "empty audio is rejected");

    // A 1 kHz tone at 24 kHz stays a 1 kHz tone at 16 kHz, with librosa's length ceil(n * 2 / 3).
    engine::runtime::AudioBuffer tone;
    tone.sample_rate = 24000;
    tone.channels = 1;
    for (int i = 0; i < 2401; ++i) tone.samples.push_back(static_cast<float>(0.5 * std::sin(2.0 * kPi * 1000.0 * i / 24000.0)));
    const auto mono = lfm3::lfm3_audio_mono_16k(tone);
    require_eq(mono.size(), static_cast<size_t>(1601), "resampled length");
    double max_error = 0.0;
    for (size_t i = 200; i < 1400; ++i) {
        max_error = std::max(max_error, std::fabs(mono[i] - 0.5 * std::sin(2.0 * kPi * 1000.0 * static_cast<double>(i) / 16000.0)));
    }
    std::printf("24 kHz -> 16 kHz tone: max error %.3e\n", max_error);
    require(max_error < 1e-2, "resampled tone");
}

}  // namespace

int main() {
    try {
        test_counts();
        test_reference();
        test_mono_16k();
        std::cout << "lfm3_audio_features_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "lfm3_audio_features_test: " << error.what() << '\n';
        return 1;
    }
}
