#pragma once

#include "engine/framework/audio/dsp.h"

namespace engine::models::xtts_v2 {

enum class MelPurpose { Conditioning, Speaker };

// XTTS uses HTK frequency spacing, with Slaney area normalization only for
// conditioning. The speaker frontend instead uses a periodic Hamming window.
class MelFrontend {
public:
    explicit MelFrontend(MelPurpose purpose);
    std::vector<float> compute(const std::vector<float> & pcm, size_t threads) const;

private:
    MelPurpose purpose_;
    audio::STFTConfig stft_;
    std::vector<float> window_;
    audio::AudioTensor filterbank_;
};
}  // namespace engine::models::xtts_v2
