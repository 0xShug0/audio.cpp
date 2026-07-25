#pragma once

#include "engine/framework/audio/dsp.h"
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
    std::vector<float> prepare_waveform(const engine::runtime::AudioBuffer & audio) const;
    ParakeetFrontendFeatures extract_waveform(const std::vector<float> & waveform, bool center) const;

private:
    std::shared_ptr<const ParakeetTDTAssets> assets_;
    engine::audio::SparseMelFilterbank filterbank_;
    std::vector<float> window_;
};

}  // namespace engine::community_models::parakeet_tdt
