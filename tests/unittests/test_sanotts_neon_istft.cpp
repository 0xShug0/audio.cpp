// SanoTtsNeonIstft against the framework HostLogMagnitudePhaseISTFT (plus the trim the sanoTTS runtime
// applies), on random log-magnitude / phase spectra. No model files needed. On builds without AArch64
// NEON it only checks that the implementation reports itself unavailable and refuses to run.

#include "engine/community_models/sanotts/neon_istft.h"

#include "engine/framework/audio/istft_graph.h"

#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using engine::models::sanotts::SanoTtsNeonIstft;

constexpr int64_t kFft = 1024;
constexpr int64_t kHop = 256;
constexpr int64_t kBins = kFft / 2 + 1;
constexpr int64_t kOutDim = 2 * kBins;

std::vector<float> hann_window() {
    std::vector<float> w(static_cast<size_t>(kFft));
    for (int64_t i = 0; i < kFft; ++i) {
        w[static_cast<size_t>(i)] = static_cast<float>(
            0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * static_cast<double>(i) / static_cast<double>(kFft)));
    }
    return w;
}

// Like the decoder output: log magnitudes mostly in [-8, 3], phases over a wide range, bins 0 and
// Nyquist at -inf (the runtime zeroes them).
std::vector<float> random_spectrum(int64_t frames, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> logmag(-8.0F, 3.0F), phase(-40.0F, 40.0F);
    std::vector<float> s(static_cast<size_t>(frames * kOutDim));
    for (int64_t f = 0; f < frames; ++f) {
        float * row = s.data() + static_cast<size_t>(f * kOutDim);
        for (int64_t b = 0; b < kBins; ++b) {
            row[b] = logmag(rng);
            row[kBins + b] = phase(rng);
        }
        row[0] = -std::numeric_limits<float>::infinity();
        row[kBins - 1] = -std::numeric_limits<float>::infinity();
    }
    return s;
}

// What the runtime does without the NEON class: framework iSTFT, then drop hop/2 per side.
std::vector<float> reference(const std::vector<float> & spectrum, int64_t frames, const std::vector<float> & window) {
    engine::audio::HostLogMagnitudePhaseISTFTConfig cfg;
    cfg.frames = frames;
    cfg.n_fft = kFft;
    cfg.hop_length = kHop;
    cfg.out_dim = kOutDim;
    cfg.threads = 1;
    engine::audio::HostLogMagnitudePhaseISTFT istft(cfg);
    const auto result = istft.compute(spectrum, window);
    const size_t edge = static_cast<size_t>(kHop / 2);
    return std::vector<float>(result.audio.begin() + static_cast<std::ptrdiff_t>(edge),
                              result.audio.end() - static_cast<std::ptrdiff_t>(edge));
}

}  // namespace

int main() try {
    const auto window = hann_window();
    if (!SanoTtsNeonIstft::available(kFft, kHop)) {
        SanoTtsNeonIstft unavailable;
        bool threw = false;
        try {
            unavailable.compute(random_spectrum(4, 1), 4, window, 1);
        } catch (const std::exception &) {
            threw = true;
        }
        engine::test::require(threw, "NEON iSTFT must refuse to run without AArch64 NEON");
        std::cout << "sanotts_neon_istft_test: skipped (no AArch64 NEON)\n";
        return 0;
    }
    engine::test::require(!SanoTtsNeonIstft::available(512, kHop), "only n_fft 1024 is supported");
    engine::test::require(!SanoTtsNeonIstft::available(kFft, 128), "only hop 256 is supported");

    SanoTtsNeonIstft istft;   // one instance for every call: its work buffer is reused and must not leak state
    // Frame counts cover the first/last-block edge cases, a partial last group of four frames,
    // and the multithreaded path (24 frames and up).
    for (const int64_t frames : {2, 3, 4, 5, 6, 7, 8, 23, 24, 25, 100, 301}) {
        const auto spectrum = random_spectrum(frames, static_cast<uint32_t>(frames));
        const auto want = reference(spectrum, frames, window);
        std::vector<float> first;
        for (const int threads : {1, 3, 4}) {
            const auto got = istft.compute(spectrum, frames, window, threads);
            engine::test::require_eq(got.size(), want.size(), "output size for frames=" + std::to_string(frames));
            double max_ref = 0.0, max_diff = 0.0;
            for (size_t i = 0; i < want.size(); ++i) {
                max_ref = std::max(max_ref, static_cast<double>(std::fabs(want[i])));
                max_diff = std::max(max_diff, static_cast<double>(std::fabs(got[i] - want[i])));
            }
            engine::test::require(max_diff <= 2.0e-4 * std::max(1.0, max_ref),
                "NEON iSTFT differs from the host iSTFT: frames=" + std::to_string(frames) +
                " threads=" + std::to_string(threads) + " max diff " + std::to_string(max_diff) +
                " (max |ref| " + std::to_string(max_ref) + ")");
            if (first.empty()) {
                first = got;
            } else {
                engine::test::require(got == first,
                    "NEON iSTFT must not depend on the thread count: frames=" + std::to_string(frames));
            }
        }
    }

    // The reused buffer must not leak state between calls of different sizes.
    {
        const auto a = random_spectrum(40, 7);
        const auto b = random_spectrum(9, 8);
        const auto a1 = istft.compute(a, 40, window, 2);
        (void)istft.compute(b, 9, window, 2);
        engine::test::require(istft.compute(a, 40, window, 2) == a1, "repeat call must give identical output");
    }

    bool threw = false;
    try {
        istft.compute(random_spectrum(4, 1), 5, window, 1);   // size does not match frames
    } catch (const std::exception &) {
        threw = true;
    }
    engine::test::require(threw, "size mismatch must be rejected");
    threw = false;
    try {
        istft.compute(random_spectrum(1, 1), 1, window, 1);   // fewer than two frames
    } catch (const std::exception &) {
        threw = true;
    }
    engine::test::require(threw, "a single frame must be rejected");

    std::cout << "sanotts_neon_istft_test: ok\n";
    return 0;
} catch (const std::exception & e) {
    std::cerr << "sanotts_neon_istft_test failed: " << e.what() << "\n";
    return 1;
}
