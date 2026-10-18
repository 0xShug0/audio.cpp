#pragma once

#include "engine/framework/audio/dsp.h"
#include "engine/models/open_wake_word/weights.h"

namespace engine::models::open_wake_word {

class Frontend {
public:
    explicit Frontend(const FrontendWeights &weights);
    // PCM16-scale float input. Output is time-major [frames, 32].
    audio::AudioTensor extract(const std::vector<float> &pcm, size_t threads) const;

private:
    std::vector<float> window_;
    audio::AudioTensor filterbank_;
};

} // namespace engine::models::open_wake_word
