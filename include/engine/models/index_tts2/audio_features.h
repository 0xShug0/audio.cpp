#pragma once

#include "engine/models/index_tts2/types.h"

#include <cstdint>
#include <vector>

namespace engine::models::index_tts2 {

struct IndexTTS2MelOutput {
    std::vector<float> values;
    int64_t channels = 0;
    int64_t frames = 0;
};

struct IndexTTS2FbankOutput {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t dims = 0;
};

struct IndexTTS2SemanticFeatureOutput {
    std::vector<float> values;
    std::vector<int32_t> attention_mask;
    int64_t frames = 0;
    int64_t dims = 0;
};

struct IndexTTS2PreparedReferenceAudio {
    std::vector<float> waveform_16k;
    std::vector<float> waveform_22k;
    IndexTTS2MelOutput mel;
    IndexTTS2FbankOutput campplus_fbank;
    IndexTTS2SemanticFeatureOutput semantic_features;
};

IndexTTS2PreparedReferenceAudio prepare_index_tts2_reference_audio(
    const std::vector<float> & samples,
    int sample_rate,
    int channels,
    const IndexTTS2S2MelConfig & mel_config,
    size_t threads,
    bool speaker_load_semantic = true);

IndexTTS2MelOutput compute_index_tts2_mel_spectrogram(
    const std::vector<float> & waveform,
    const IndexTTS2S2MelConfig & config,
    size_t threads);

IndexTTS2FbankOutput compute_index_tts2_campplus_fbank_16k(const std::vector<float> & waveform_16k);

IndexTTS2SemanticFeatureOutput compute_index_tts2_semantic_features_16k(const std::vector<float> & waveform_16k);

}  // namespace engine::models::index_tts2
