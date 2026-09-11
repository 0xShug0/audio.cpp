#pragma once

#include "engine/models/confucius4_tts/types.h"

#include <cstdint>
#include <vector>

namespace engine::models::confucius4_tts {

struct ConfuciusMelOutput {
    std::vector<float> values;
    int64_t channels = 0;
    int64_t frames = 0;
};

struct ConfuciusFbankOutput {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t dims = 0;
};

struct ConfuciusSemanticFeatureOutput {
    std::vector<float> values;
    std::vector<int32_t> attention_mask;
    int64_t frames = 0;
    int64_t dims = 0;
};

struct ConfuciusPreparedReferenceAudio {
    std::vector<float> waveform_16k;
    std::vector<float> waveform_target;
    ConfuciusMelOutput reference_mel;
    ConfuciusFbankOutput campplus_fbank;
    ConfuciusSemanticFeatureOutput semantic_features;
};

ConfuciusPreparedReferenceAudio prepare_confucius_reference_audio(
    const std::vector<float> & samples,
    int sample_rate,
    int channels,
    const ConfuciusAudioConfig & config,
    size_t threads);

ConfuciusMelOutput compute_confucius_mel_spectrogram(
    const std::vector<float> & waveform,
    const ConfuciusAudioConfig & config,
    size_t threads);

ConfuciusFbankOutput compute_confucius_campplus_fbank_16k(const std::vector<float> & waveform_16k);

ConfuciusSemanticFeatureOutput compute_confucius_semantic_features_16k(const std::vector<float> & waveform_16k);

}  // namespace engine::models::confucius4_tts
