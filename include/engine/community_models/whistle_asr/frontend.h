#pragma once

#include <cstddef>
#include <vector>

namespace engine::community_models::whistle_asr {

struct MelFeatures {
    std::vector<float> values;
    size_t frames = 0;
};

class WhistleFrontend {
public:
    explicit WhistleFrontend(std::vector<float> filterbank);

    // The caller supplies mono 16 kHz float PCM, with at most 30 seconds per pass.
    [[nodiscard]] MelFeatures extract(const std::vector<float> & samples) const;

private:
    std::vector<float> filterbank_;
};

}  // namespace engine::community_models::whistle_asr
