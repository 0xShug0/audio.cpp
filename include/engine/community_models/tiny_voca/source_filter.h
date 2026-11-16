#pragma once

#include "engine/community_models/tiny_voca/assets.h"

#include <cstdint>
#include <vector>

namespace engine::models::tiny_voca {

// The waveform head of tiny-voca. It has no neural upsampling: a band-limited
// pulse train (from the pitch) and noise are each shaped by a filter that
// changes from frame to frame, given as a cepstrum and applied in the STFT
// domain. This is signal processing on the host; the transforms are the
// framework's real FFT.
class TinyVocaSourceFilter {
public:
    TinyVocaSourceFilter(
        const TinyVocaConfig & config,
        std::vector<float> harmonic_weight,
        std::vector<float> harmonic_bias,
        std::vector<float> noise_weight,
        std::vector<float> noise_bias);

    // `hidden` holds `frames` rows of `channels` values, `f0` the pitch of each
    // frame in Hz and `noise` one sample of white noise for each sample of
    // audio (frames * hop). Returns frames * hop samples. The noise is taken
    // over: it is freed as soon as its part of the sound is made.
    std::vector<float> synthesize(
        const std::vector<float> & hidden,
        const std::vector<float> & f0,
        std::vector<float> noise,
        int64_t frames) const;

    // Band-limited pulse train with unit power for a pitch contour in Hz.
    std::vector<float> pulse_train(const std::vector<float> & f0) const;

private:
    struct Branch {
        int64_t fft = 0;
        int64_t hop = 0;
        int64_t steps = 0;                  // filter frames for each acoustic frame
        int64_t coefficients = 0;           // cepstrum values of one frame
        std::vector<float> lifter;
        std::vector<double> real;           // coefficients x bins: the transform of a unit coefficient
        std::vector<double> imaginary;
        std::vector<float> window;
        std::vector<float> weight;          // the linear head: (coefficients + steps) x channels
        std::vector<float> bias;
    };

    void shape(const Branch & branch, const std::vector<float> & hidden, const std::vector<float> & source, int64_t frames,
               bool interpolate, std::vector<float> & out) const;

    TinyVocaConfig config_;
    Branch harmonic_;
    Branch noise_;
};

}  // namespace engine::models::tiny_voca
