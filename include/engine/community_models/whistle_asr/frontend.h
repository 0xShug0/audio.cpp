#pragma once

#include "engine/framework/audio/nemo_mel_frontend.h"

#include <cstddef>
#include <vector>

namespace engine::community_models::whistle_asr {

struct MelFeatures {
    std::vector<float> values;
    size_t frames = 0;
};

// Whistle log-mel features: the shared NemoMelFrontend STFT/mel/normalization
// path with Whistle's percentile gain normalization applied locally first.
class WhistleFrontend {
public:
    explicit WhistleFrontend(std::vector<float> filterbank);

    // The caller supplies mono 16 kHz float PCM, with at most 30 seconds per pass.
    [[nodiscard]] MelFeatures extract(const std::vector<float> & samples) const;

private:
    audio::NemoMelFrontend frontend_;
};

}  // namespace engine::community_models::whistle_asr
