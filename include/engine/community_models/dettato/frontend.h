#pragma once

#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/fft.h"
#include "engine/community_models/dettato/assets.h"
#include <string>
#include <vector>

namespace engine::community_models::dettato::detail {

struct HarmonicFeatures {
    int64_t batch = 0, channels = 0, frames = 0, bins = 288, extended_bins = 0;
    std::vector<float> values; // [batch, channels, frames, bins]
    std::vector<float> power;  // [batch, extended_bins, frames]
};

// Uses checkpoint-specified filterbanks/windows and the shared STFT/filterbank
// implementation. Constants and FFT plans are reused, not rebuilt per window.
class HarmonicFrontend {
  public:
    HarmonicFrontend(HarmonicFrontendWeights weights, size_t threads);
    HarmonicFeatures extract(const std::vector<float> &audio, int64_t batch, bool pcen = false) const;

  private:
    audio::SparseMelFilterbank long_filter_, short_filter_;
    std::vector<float> long_window_, short_window_;
    size_t threads_;
    mutable std::shared_ptr<audio::RealFFTPlan> pcen_plan_;
    mutable std::vector<float> pcen_buffer_;
    mutable std::vector<std::complex<float>> pcen_spectrum_, pcen_kernel_;
    void apply_pcen(HarmonicFeatures &features) const;
};

class DrumFrontend {
  public:
    explicit DrumFrontend(DrumFrontendWeights weights, size_t threads);
    std::vector<float> extract(const std::vector<float> &mono44k) const;

  private:
    std::vector<audio::SparseMelFilterbank> filters_;
    std::vector<std::vector<float>> windows_;
    size_t threads_;
};

struct Tuning {
    double cents = 0, strength = 0, ratio = 1;
};
// Harmonic-salience tuning estimate used before pitched-note transcription.
// Reuses the shared FFT implementation with bounded, session-owned buffers.
class TuningEstimator {
  public:
    explicit TuningEstimator(size_t threads);
    Tuning estimate(const std::vector<float> &audio16);

  private:
    audio::RealFFTPlan fft_{8192};
    size_t threads_;
    std::vector<float> window_, input_, magnitude_, salience_;
    std::vector<double> cents_;
    std::vector<std::complex<float>> spectrum_;
};
} // namespace engine::community_models::dettato::detail
