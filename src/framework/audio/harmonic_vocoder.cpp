#include "engine/framework/audio/harmonic_vocoder.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::audio {

std::vector<float> generate_harmonic_source(const std::vector<float> & f0, int64_t scale,
    const HarmonicSourceConfig & config, const std::vector<float> & mixing_weight, float mixing_bias,
    const std::vector<float> & phase_uniform, const std::vector<float> & gaussian) {
    if (f0.empty() || scale <= 0 || config.sample_rate <= 0 || config.harmonics <= 0 ||
        mixing_weight.size() != static_cast<size_t>(config.harmonics)) {
        throw std::runtime_error("invalid harmonic source geometry");
    }
    std::vector<float> f0_upsampled(f0.size() * scale, 0.0F);
    for (size_t t = 0; t < f0.size(); ++t) {
        std::fill(f0_upsampled.begin() + static_cast<std::ptrdiff_t>(t * scale),
            f0_upsampled.begin() + static_cast<std::ptrdiff_t>((t + 1) * scale), f0[t]);
    }
    constexpr float kPi = 3.14159265358979323846F;
    constexpr float kTwoPi = 2.0F * kPi;
    const int64_t frames = static_cast<int64_t>(f0_upsampled.size());
    const int64_t harmonics = config.harmonics;
    const size_t phase_count = static_cast<size_t>(harmonics);
    const size_t gaussian_count = static_cast<size_t>(harmonics * frames);
    if (phase_uniform.size() != phase_count || gaussian.size() != gaussian_count) {
        throw std::runtime_error("harmonic source random value count mismatch");
    }
    std::vector<float> phase(static_cast<size_t>(harmonics), 0.0F);
    for (int64_t h = 0; h < harmonics; ++h) {
        phase[static_cast<size_t>(h)] = phase_uniform[static_cast<size_t>(h)] * kTwoPi - kPi;
    }
    phase[0] = 0.0F;
    std::vector<float> source(static_cast<size_t>(frames), 0.0F);
    std::vector<float> frac(static_cast<size_t>(harmonics), 0.0F);
    for (int64_t t = 0; t < frames; ++t) {
        const float base_f0 = f0_upsampled[static_cast<size_t>(t)];
        const float uv = base_f0 > config.voiced_threshold ? 1.0F : 0.0F;
        float sum = mixing_bias;
        for (int64_t h = 0; h < harmonics; ++h) {
            frac[static_cast<size_t>(h)] +=
                base_f0 * static_cast<float>(h + 1) / static_cast<float>(config.sample_rate);
            frac[static_cast<size_t>(h)] -= std::floor(frac[static_cast<size_t>(h)]);
            const float theta = kTwoPi * frac[static_cast<size_t>(h)];
            float wave =
                config.amplitude * std::sin(theta + phase[static_cast<size_t>(h)]);
            const float noise_amp =
                uv * config.noise_std + (1.0F - uv) * config.amplitude / 3.0F;
            wave = wave * uv + noise_amp * gaussian[static_cast<size_t>(h * frames + t)];
            sum += wave * mixing_weight[static_cast<size_t>(h)];
        }
        source[static_cast<size_t>(t)] = std::tanh(sum);
    }
    return source;
}

AudioTensor stft_real_imag(const std::vector<float> & source, int64_t n_fft, int64_t hop,
    std::vector<float> & window) {
    const int64_t bins = n_fft / 2 + 1;
    const audio::STFTConfig window_config{
        n_fft,
        hop,
        n_fft,
        true,
        audio::STFTPadMode::Reflect,
        audio::STFTFamily::Kokoro,
    };
    window = audio::get_cached_stft_window(window_config);
    const auto complex = audio::STFT().compute_complex(
        source,
        window,
        1,
        static_cast<int64_t>(source.size()),
        {n_fft, hop, n_fft});
    const auto stft_frames = complex.shape[2];
    std::vector<float> stft(static_cast<size_t>((2 * bins) * stft_frames), 0.0F);
    for (int64_t f = 0; f < bins; ++f) {
        for (int64_t t = 0; t < stft_frames; ++t) {
            const size_t complex_index = static_cast<size_t>((f * stft_frames + t) * 2);
            stft[static_cast<size_t>(f * stft_frames + t)] = complex.values[complex_index];
            stft[static_cast<size_t>((bins + f) * stft_frames + t)] = complex.values[complex_index + 1];
        }
    }
    return {std::move(stft), {1, 2 * bins, stft_frames}};
}

std::vector<float> sine_phase_istft(const std::vector<float> & post, int64_t frames,
    int64_t n_fft, int64_t hop, float audio_limit, const std::vector<float> & window) {
    const int64_t bins = n_fft / 2 + 1;
    if (static_cast<int64_t>(post.size()) != (2 * bins) * frames) {
        throw std::runtime_error("log-magnitude/phase shape mismatch");
    }
    std::vector<float> complex_spec(static_cast<size_t>(bins * frames * 2), 0.0F);
    for (int64_t f = 0; f < bins; ++f) {
        for (int64_t t = 0; t < frames; ++t) {
            float magnitude = std::exp(post[static_cast<size_t>(f * frames + t)]);
            magnitude = std::min(magnitude, 1.0e2F);
            const float phase = std::sin(post[static_cast<size_t>((bins + f) * frames + t)]);
            const size_t complex_index = static_cast<size_t>((f * frames + t) * 2);
            complex_spec[complex_index] = magnitude * std::cos(phase);
            complex_spec[complex_index + 1] = magnitude * std::sin(phase);
        }
    }
    const int64_t samples = hop * (frames - 1);
    auto wav = audio::ISTFT().compute(
        complex_spec,
        window,
        1,
        bins,
        frames,
        samples,
        {n_fft, hop, n_fft});
    for (float & sample : wav.values) {
        sample = std::clamp(sample, -audio_limit, audio_limit);
    }
    return wav.values;
}

} // namespace engine::audio
