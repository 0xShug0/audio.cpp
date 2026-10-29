#pragma once

// LFM3-Audio ASR input: 16 kHz mono audio and its 128-bin log-mel features
// (n_fft = win = 400, hop 160, Slaney, no centering), scaled as (logmel + 4) / 4.

#include "engine/framework/runtime/session.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine::community_models::lfm3_audio {

// Mixdown and resampling with libsoxr HQ (librosa soxr_hq, as HF loads audio); the torchaudio sinc resampler
// only when libsoxr is absent.
std::vector<float> lfm3_audio_mono_16k(const runtime::AudioBuffer & audio);
int64_t lfm3_feature_frames(int64_t samples);  // 1 + ceil(max(n-400,0)/160)
int64_t lfm3_audio_tokens(int64_t frames);     // ceil(frames/8)

struct Lfm3Features {  // feature-major [128][frames]
    std::vector<float> values;
    int64_t frames = 0;
};

class Lfm3FeatureExtractor {
public:
    Lfm3FeatureExtractor();
    // Right-pads to whole frames (at least one); with LFM3_AUDIO_DUMP_DIR set it writes the features dump.
    [[nodiscard]] Lfm3Features extract(const std::vector<float> & mono16k, size_t threads = 0) const;
};

}  // namespace engine::community_models::lfm3_audio
