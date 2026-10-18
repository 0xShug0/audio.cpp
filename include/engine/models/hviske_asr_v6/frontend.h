#pragma once

#include "engine/framework/audio/mel_spectrogram_frontend.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::hviske_asr_v6 {

struct HviskeV6Features {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t valid_frames = 0;
};

class HviskeV6WhisperFrontend {
public:
    explicit HviskeV6WhisperFrontend(int64_t mel_bins);
    HviskeV6Features extract(const std::vector<float> & mono_16k, size_t threads) const;

private:
    std::shared_ptr<const audio::MelSpectrogramFrontend> mel_;
};

}  // namespace engine::models::hviske_asr_v6
