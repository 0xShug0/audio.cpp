#pragma once

#include "engine/framework/audio/dsp.h"

namespace engine::models::gigaam_asr {

class GigaAMFrontend {
public:
    GigaAMFrontend(audio::STFTConfig config, std::vector<float> window, audio::AudioTensor filterbank);
    audio::AudioTensor extract(const std::vector<float> & mono_16khz, size_t threads) const;

private:
    audio::STFTConfig config_;
    std::vector<float> window_;
    audio::SparseMelFilterbank filterbank_;
};

}  // namespace engine::models::gigaam_asr
