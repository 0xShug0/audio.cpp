#include "test_assert.h"
#include "engine/framework/audio/harmonic_vocoder.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void test_source() {
    namespace audio = engine::audio;
    const audio::HarmonicSourceConfig config{24000, 2, 0.1F, 0.003F, 10.0F};
    const std::vector<float> f0{0, 100, 220, 5, 440}, weight{0.7F, -0.2F}, phase{0.25F, 0.75F};
    const int scale = 4, samples = static_cast<int>(f0.size()) * scale;
    std::vector<float> noise(samples * 2);
    for (size_t i = 0; i < noise.size(); ++i) noise[i] = (static_cast<int>(i % 7) - 3) * 0.1F;
    const auto actual = audio::generate_harmonic_source(f0, scale, config, weight, 0.02F, phase, noise);
    engine::test::require(actual.size() == static_cast<size_t>(samples), "source length");
    constexpr float pi = 3.14159265358979323846F;
    float accumulated[2]{};
    for (int t = 0; t < samples; ++t) {
        const float frequency = f0[t / scale];
        const float voiced = frequency > 10 ? 1.0F : 0.0F;
        float expected = 0.02F;
        for (int h = 0; h < 2; ++h) {
            accumulated[h] += frequency * static_cast<float>(h + 1) / 24000.0F;
            accumulated[h] -= std::floor(accumulated[h]);
            const float initial = h == 0 ? 0.0F : phase[h] * (2 * pi) - pi;
            const float wave = 0.1F * std::sin((2 * pi) * accumulated[h] + initial) * voiced;
            const float amplitude = voiced * 0.003F + (1.0F - voiced) * 0.1F / 3.0F;
            expected += (wave + amplitude * noise[h * samples + t]) * weight[h];
        }
        engine::test::require_close(actual[t], std::tanh(expected), 1e-7F, "harmonic source reference");
    }
    bool rejected = false;
    try { audio::generate_harmonic_source(f0, scale, config, weight, 0, {}, noise); }
    catch (const std::runtime_error &) { rejected = true; }
    engine::test::require(rejected, "source random shape validation");
}

void test_spectrum(int64_t n_fft, int64_t hop) {
    namespace audio = engine::audio;
    std::vector<float> source(256);
    for (size_t t = 0; t < source.size(); ++t) source[t] = std::sin(static_cast<float>(t) * 0.1F);
    std::vector<float> window;
    const auto spectrum = audio::stft_real_imag(source, n_fft, hop, window);
    const auto bins = n_fft / 2 + 1, frames = spectrum.shape[2];
    const auto reference = audio::STFT().compute_complex(source, window, 1, source.size(), {n_fft, hop, n_fft});
    engine::test::require(spectrum.shape == std::vector<int64_t>({1, 2 * bins, frames}), "spectrum shape");
    for (int64_t f = 0; f < bins; ++f) {
        for (int64_t t = 0; t < frames; ++t) {
            engine::test::require(spectrum.values[f * frames + t] == reference.values[(f * frames + t) * 2], "real packing");
            engine::test::require(spectrum.values[(bins + f) * frames + t] == reference.values[(f * frames + t) * 2 + 1], "imaginary packing");
        }
    }
    std::vector<float> post(2 * bins * frames), complex(2 * bins * frames);
    for (int64_t f = 0; f < bins; ++f) {
        for (int64_t t = 0; t < frames; ++t) {
            const auto index = f * frames + t;
            post[index] = f == 0 ? 6.0F : -1.0F;
            post[bins * frames + index] = static_cast<float>(t % 11) * 0.2F;
            const float magnitude = std::min(std::exp(post[index]), 100.0F);
            const float phase = std::sin(post[bins * frames + index]);
            complex[index * 2] = magnitude * std::cos(phase);
            complex[index * 2 + 1] = magnitude * std::sin(phase);
        }
    }
    const auto expected = audio::ISTFT().compute(complex, window, 1, bins, frames,
        hop * (frames - 1), {n_fft, hop, n_fft});
    const auto actual = audio::sine_phase_istft(post, frames, n_fft, hop, 0.5F, window);
    engine::test::require(actual.size() == expected.values.size(), "reconstruction length");
    for (size_t i = 0; i < actual.size(); ++i)
        engine::test::require_close(actual[i], std::clamp(expected.values[i], -0.5F, 0.5F), 1e-7F, "reconstruction reference");
}

} // namespace

int main() {
    try {
        test_source();
        test_spectrum(16, 4);
        test_spectrum(32, 8);
        std::cout << "harmonic_vocoder_test passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
