#pragma once

#include "engine/framework/audio/nemo_mel_frontend.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <vector>

namespace engine::models::nemotron_transcribe {

struct FrontendConfig {
    int sample_rate = 16000;
    int mel_bins = 128;
    int n_fft = 512;
    int window_length = 400;
    int hop_length = 160;
    float preemphasis = 0.97f;
};

struct MelFeatures {
    int64_t frames = 0;
    int bins = 0;
    double duration_seconds = 0.0;
    // Unnormalized log-mel, [frames, bins]. ASR normalization is utterance-wide.
    std::vector<float> values;
};

class Frontend {
public:
    // Released Nemotron 3.5 constants are BF16. Production supplies the exact
    // checkpoint window; an empty vector retains the F32 synthetic-test window.
    Frontend(FrontendConfig config, std::vector<float> trained_filterbank,
        std::vector<float> trained_window = {});
    MelFeatures extract(const runtime::AudioBuffer & audio, int threads = 1) const;
    const FrontendConfig & config() const noexcept { return config_; }

private:
    FrontendConfig config_;
    audio::NemoMelFrontend frontend_;
};

// NeMo normalize_batch("per_feature"), unbiased std plus 1e-5.
std::vector<float> normalize_asr_features(const MelFeatures & features);

// Native SALM's conservative silence gate: both peak and RMS must be quiet.
bool is_near_silent(const std::vector<float> & samples);

}  // namespace engine::models::nemotron_transcribe
