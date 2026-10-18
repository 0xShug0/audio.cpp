#pragma once

#include "engine/models/universr/assets.h"

namespace engine::models::universr {

struct UniverSRFeatures {
    int64_t frames = 0;
    int64_t original_samples = 0;
    int low_bins = 0;
    std::vector<float> low_spectrum;
};

UniverSRFeatures analyze_audio(const UniverSRAssets & assets,
    const std::vector<float> & interleaved, int channels, int file_sample_rate,
    int effective_sample_rate, int threads);

std::vector<float> synthesize_audio(const UniverSRAssets & assets,
    const UniverSRFeatures & features, const std::vector<float> & generated_high_spectrum,
    int threads);

}  // namespace engine::models::universr
