#pragma once

#include "engine/framework/audio/dsp.h"

#include <cstdint>
#include <vector>

namespace engine::audio {

struct HarmonicSourceConfig {
    int64_t sample_rate = 0;
    int64_t harmonics = 1;
    float amplitude = 0.1F;
    float noise_std = 0.003F;
    float voiced_threshold = 10.0F;
};

// Repeat each F0 frame by scale, add harmonic/noise excitation, then mix and tanh.
// Caller supplies uniform phases and channel-first Gaussian noise.
std::vector<float> generate_harmonic_source(const std::vector<float> & f0, int64_t scale,
    const HarmonicSourceConfig & config, const std::vector<float> & mixing_weight, float mixing_bias,
    const std::vector<float> & phase_uniform, const std::vector<float> & gaussian);

// Centered reflect STFT with a Kokoro-family Hann window. Real then imaginary
// channels, shaped [1, 2*bins, frames].
AudioTensor stft_real_imag(const std::vector<float> & source, int64_t n_fft, int64_t hop,
    std::vector<float> & window);

// Channel-first log-magnitude/phase; phase uses sin, magnitude is capped at 100.
std::vector<float> sine_phase_istft(const std::vector<float> & post, int64_t frames,
    int64_t n_fft, int64_t hop, float audio_limit, const std::vector<float> & window);

} // namespace engine::audio
