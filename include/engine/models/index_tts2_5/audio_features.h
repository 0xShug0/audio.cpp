#pragma once

#include "engine/models/index_tts2_5/types.h"

#include <cstdint>
#include <vector>

namespace engine::models::index_tts2_5 {

struct IndexTTS25MelOutput {
    std::vector<float> values;
    int64_t channels = 0;
    int64_t frames = 0;
};

struct IndexTTS25FbankOutput {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t dims = 0;
};

struct IndexTTS25SemanticFeatureOutput {
    std::vector<float> values;
    std::vector<int32_t> attention_mask;
    int64_t frames = 0;
    int64_t dims = 0;
};

struct IndexTTS25PreparedReferenceAudio {
    std::vector<float> waveform_16k;
    std::vector<float> waveform_22k;
    IndexTTS25MelOutput mel;
    IndexTTS25FbankOutput campplus_fbank;
    IndexTTS25SemanticFeatureOutput semantic_features;
};

IndexTTS25PreparedReferenceAudio prepare_index_tts2_5_reference_audio(
    const std::vector<float> & samples,
    int sample_rate,
    int channels,
    const IndexTTS25S2MelConfig & mel_config,
    size_t threads,
    bool speaker_load_semantic = true);

IndexTTS25MelOutput compute_index_tts2_5_mel_spectrogram(
    const std::vector<float> & waveform,
    const IndexTTS25S2MelConfig & config,
    size_t threads);

IndexTTS25FbankOutput compute_index_tts2_5_campplus_fbank_16k(const std::vector<float> & waveform_16k);

IndexTTS25SemanticFeatureOutput compute_index_tts2_5_semantic_features_16k(const std::vector<float> & waveform_16k);

}  // namespace engine::models::index_tts2_5
