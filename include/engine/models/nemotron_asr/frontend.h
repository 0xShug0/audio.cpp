#pragma once

#include "engine/framework/audio/dsp.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/nemotron_asr/assets.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::nemotron_asr {

struct NemotronFrontendFeatures {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t valid_frames = 0;
    int64_t feature_dim = 0;
};

class NemotronFrontend {
public:
    explicit NemotronFrontend(std::shared_ptr<const NemotronAssets> assets);

    NemotronFrontendFeatures extract(
        const engine::runtime::AudioBuffer & audio,
        bool center) const;
    std::vector<float> prepare_waveform(const engine::runtime::AudioBuffer & audio) const;
    NemotronFrontendFeatures extract_waveform(const std::vector<float> & waveform, bool center) const;

private:
    std::shared_ptr<const NemotronAssets> assets_;
    engine::audio::SparseMelFilterbank filterbank_;
    std::vector<float> window_;
};

}  // namespace engine::models::nemotron_asr
