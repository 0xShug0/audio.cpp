#pragma once

#include "engine/framework/audio/fft.h"
#include <vector>

namespace engine::models::dettato::detail {
struct Tuning {
    double cents = 0, strength = 0, ratio = 1;
};
// Harmonic-salience tuning estimate used before pitched-note transcription.
// Reuses the shared FFT implementation with bounded, session-owned buffers.
class TuningEstimator {
  public:
    explicit TuningEstimator(size_t threads);
    Tuning estimate(const std::vector<float> & audio16);
  private:
    audio::RealFFTPlan fft_{8192};
    size_t threads_;
    std::vector<float> window_, input_, magnitude_, salience_;
    std::vector<double> cents_;
    std::vector<std::complex<float>> spectrum_;
};
} // namespace engine::models::dettato::detail
