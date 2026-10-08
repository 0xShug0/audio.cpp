#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <vector>

namespace engine::models::sherpa_kws {

struct SherpaFbankFeatures {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t feature_dim = 80;
};

SherpaFbankFeatures compute_sherpa_fbank(
    const runtime::AudioBuffer & audio);

}  // namespace engine::models::sherpa_kws
