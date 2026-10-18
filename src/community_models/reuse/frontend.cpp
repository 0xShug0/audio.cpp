#include "engine/community_models/reuse/frontend.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::reuse {

FrontendSpectrum analyze_spectrum(const std::vector<float> & waveform, int sample_rate, size_t threads) {
    int64_t fft = 320LL * sample_rate / 8000;
    int64_t hop = 40LL * sample_rate / 8000;
    fft += fft % 2;
    hop += hop % 2;
    const int64_t samples = static_cast<int64_t>(waveform.size());
    if (sample_rate < 8000 || sample_rate > 48000 || samples <= fft / 2) {
        throw std::runtime_error("RE-USE requires 8-48 kHz audio longer than its 20 ms STFT reflection pad");
    }

    FrontendSpectrum result;
    result.config = {fft, hop, fft, true, audio::STFTPadMode::Reflect, audio::STFTFamily::PeriodicF32};
    result.samples = samples;
    result.frames = 1 + samples / hop;
    result.bins = fft / 2 + 1;
    result.hop = hop;
    const auto & window = audio::get_cached_stft_window(result.config);
    result.spectrum = audio::STFT().compute_complex(waveform, window, 1, samples, result.config, threads);
    result.features.resize(result.frames * result.bins * 2);
#ifdef _OPENMP
#pragma omp parallel for num_threads(std::max<size_t>(1, threads)) if (threads > 1 && result.frames >= 8)
#endif
    for (int64_t t = 0; t < result.frames; ++t) {
        for (int64_t f = 0; f < result.bins; ++f) {
            const size_t index = (f * result.frames + t) * 2;
            const float real = result.spectrum.values[index];
            const float imaginary = result.spectrum.values[index + 1];
            result.features[t * result.bins + f] = std::log1p(std::hypot(real, imaginary));
            result.features[result.frames * result.bins + t * result.bins + f] = std::atan2(imaginary, real);
        }
    }
    return result;
}

std::vector<float> synthesize_spectrum(FrontendSpectrum & input,
                                      const std::vector<std::vector<float>> & heads, size_t threads) {
    const int64_t output_bins = 2 * ((input.bins + 2 - 3) / 2 + 1);
#ifdef _OPENMP
#pragma omp parallel for num_threads(std::max<size_t>(1, threads)) if (threads > 1 && input.frames >= 8)
#endif
    for (int64_t t = 0; t < input.frames; ++t) {
        int64_t zero_count = 0;
        for (int64_t f = 0; f < input.bins; ++f) {
            zero_count += heads[0][t * output_bins + f] <= 0;
        }
        for (int64_t f = 0; f < input.bins; ++f) {
            const auto index = t * output_bins + f;
            const float magnitude = zero_count * 2 > input.bins ? 0.0f : std::expm1(std::max(0.0f, heads[0][index]));
            const float real = heads[1][index];
            const float imaginary = heads[2][index];
            const float phase_norm = std::sqrt(real * real + imaginary * imaginary);
            const size_t spectrum_index = (f * input.frames + t) * 2;
            if (phase_norm == 0.0f) {
                const float phase = std::atan2(imaginary, real);
                input.spectrum.values[spectrum_index] = magnitude * std::cos(phase);
                input.spectrum.values[spectrum_index + 1] = magnitude * std::sin(phase);
            } else {
                input.spectrum.values[spectrum_index] = magnitude * real / phase_norm;
                input.spectrum.values[spectrum_index + 1] = magnitude * imaginary / phase_norm;
            }
        }
    }
    const int64_t reconstructed = (input.frames - 1) * input.hop;
    const auto & window = audio::get_cached_stft_window(input.config);
    auto output = audio::ISTFT().compute(input.spectrum.values, window, 1, input.bins, input.frames,
        reconstructed, input.config, threads).values;
    output.resize(input.samples, 1e-8f);
    return output;
}

}  // namespace engine::models::reuse
