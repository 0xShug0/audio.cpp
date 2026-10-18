#pragma once

#include "engine/framework/audio/dsp.h"
#include "engine/models/ced/assets.h"

namespace engine::models::ced {

class CedLogMelFrontend {
public:
    CedLogMelFrontend(const CedConfig & config, const assets::TensorSource & source);
    audio::AudioTensor extract(const std::vector<float> & mono, size_t threads) const;

private:
    audio::STFTConfig stft_;
    std::vector<float> window_;
    audio::SparseMelFilterbank filterbank_;
};

}  // namespace engine::models::ced
