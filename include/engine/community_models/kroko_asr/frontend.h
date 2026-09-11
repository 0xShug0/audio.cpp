#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <vector>

namespace engine::models::kroko_asr {

struct KrokoFbankFeatures {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t feature_dim = 80;
};

KrokoFbankFeatures compute_kroko_fbank(
    const runtime::AudioBuffer & audio);

}  // namespace engine::models::kroko_asr
