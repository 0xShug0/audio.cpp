#pragma once

#include "engine/framework/audio/dsp.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/hviske_asr/assets.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::hviske_asr {

struct HviskeFrontendFeatures {
    std::vector<float> values;
    int64_t feature_dim = 0;
    int64_t frames = 0;
    int64_t valid_frames = 0;
};

class HviskeFrontend {
public:
    explicit HviskeFrontend(std::shared_ptr<const HviskeAssets> assets);

    HviskeFrontendFeatures extract(const engine::runtime::AudioBuffer & audio) const;

private:
    std::shared_ptr<const HviskeAssets> assets_;
    engine::audio::SparseMelFilterbank filterbank_;
    std::vector<float> window_;
};

}  // namespace engine::models::hviske_asr
