#pragma once

#include "engine/framework/audio/dsp.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine::models::reuse {

struct FrontendSpectrum {
    audio::STFTConfig config;
    audio::AudioTensor spectrum;
    std::vector<float> features;
    int64_t samples = 0;
    int64_t frames = 0;
    int64_t bins = 0;
    int64_t hop = 0;
};

FrontendSpectrum analyze_spectrum(const std::vector<float> & waveform, int sample_rate, size_t threads);
std::vector<float> synthesize_spectrum(FrontendSpectrum & input,
    const std::vector<std::vector<float>> & heads, size_t threads);

}  // namespace engine::models::reuse
