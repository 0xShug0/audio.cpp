#pragma once

#include "engine/models/seed_vc/assets.h"

#include <cstdint>
#include <vector>

namespace engine::models::seed_vc {

struct SeedVcMelSpectrogramOutput {
    std::vector<float> mel;
    int64_t channels = 0;
    int64_t frames = 0;
};

struct SeedVcCampplusFbankOutput {
    std::vector<float> features;
    int64_t frames = 0;
    int64_t dims = 0;
};

struct SeedVcPreparedAudio {
    std::vector<float> waveform_22k;
    std::vector<float> waveform_16k;
};

struct SeedVcPreparedAudioForSampleRate {
    std::vector<float> waveform;
    std::vector<float> waveform_16k;
};

SeedVcPreparedAudio seed_vc_prepare_audio(
    const std::vector<float> & samples,
    int sample_rate,
    int channels,
    int64_t max_22k_samples = 0);

SeedVcPreparedAudioForSampleRate seed_vc_prepare_audio_for_sample_rate(
    const std::vector<float> & samples,
    int sample_rate,
    int channels,
    int output_sample_rate,
    int64_t max_output_samples = 0);

SeedVcMelSpectrogramOutput compute_seed_vc_mel_spectrogram(
    const std::vector<float> & waveform,
    const SeedVcMelConfig & config,
    size_t threads);

SeedVcCampplusFbankOutput compute_seed_vc_campplus_fbank_16k(const std::vector<float> & waveform_16k);

}  // namespace engine::models::seed_vc
