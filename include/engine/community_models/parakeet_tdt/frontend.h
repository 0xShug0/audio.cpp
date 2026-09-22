#pragma once

#include "engine/framework/audio/nemo_mel_frontend.h"
#include "engine/framework/runtime/session.h"
#include "engine/community_models/parakeet_tdt/assets.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::parakeet_tdt {

struct ParakeetFrontendFeatures {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t valid_frames = 0;
    int64_t feature_dim = 0;
};

class ParakeetFrontend {
public:
    explicit ParakeetFrontend(std::shared_ptr<const ParakeetTDTAssets> assets);

    ParakeetFrontendFeatures extract(
        const engine::runtime::AudioBuffer & audio,
        bool center) const;
private:
    engine::audio::NemoMelFrontend frontend_;
};

}  // namespace engine::community_models::parakeet_tdt
