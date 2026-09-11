#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/models/audiosr/assets.h"

#include <cstdint>
#include <vector>

namespace engine::models::audiosr {

struct AudioSRFrontendOutput {
    std::vector<float> waveform;
    std::vector<float> mel;
    std::vector<float> lowpass_waveform;
    std::vector<float> lowpass_mel;
    int64_t samples = 0;
    int64_t original_samples = 0;
    int64_t mel_frames = 0;
    int64_t latent_time = 0;
};

class AudioSRFrontend {
public:
    explicit AudioSRFrontend(const AudioSRAssets & assets);

    AudioSRFrontendOutput compute(
        const engine::runtime::AudioBuffer & audio,
        uint32_t seed,
        uint64_t lowpass_rng_offset = 0,
        size_t threads = 0) const;

private:
    AudioSRConfig config_;
    std::vector<float> lowpass_sos_;
    std::vector<float> lowpass_sos_valid_;
};

}  // namespace engine::models::audiosr
