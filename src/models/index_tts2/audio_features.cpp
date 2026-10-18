#include "engine/models/index_tts2/audio_features.h"

#include "engine/framework/audio/reference_audio_frontend.h"

#include <utility>

namespace engine::models::index_tts2 {
namespace {

engine::audio::MelSpectrogramFrontendConfig mel_config(const IndexTTS2S2MelConfig & config) {
    engine::audio::MelSpectrogramFrontendConfig mel;
    mel.sample_rate = config.sample_rate;
    mel.n_fft = config.n_fft;
    mel.hop_length = config.hop_length;
    mel.win_length = config.win_length;
    mel.n_mels = config.n_mels;
    mel.mel_fmin = config.fmin;
    mel.mel_fmax = config.fmax.value_or(static_cast<float>(config.sample_rate) / 2.0f);
    mel.magnitude_epsilon = 1.0e-9f;
    return mel;
}

}  // namespace

IndexTTS2MelOutput compute_index_tts2_mel_spectrogram(
    const std::vector<float> & waveform, const IndexTTS2S2MelConfig & config, size_t threads) {
    auto features = engine::audio::get_cached_mel_spectrogram_frontend(mel_config(config))
        ->extract_mono(waveform, threads);
    return {std::move(features.values), features.mel_bins, features.frames};
}

IndexTTS2FbankOutput compute_index_tts2_campplus_fbank_16k(const std::vector<float> & waveform_16k) {
    auto features = engine::audio::ReferenceAudioFrontend::campplus_fbank_16k(waveform_16k);
    return {std::move(features.values), features.frames, features.feature_dim};
}

IndexTTS2SemanticFeatureOutput compute_index_tts2_semantic_features_16k(
    const std::vector<float> & waveform_16k) {
    auto features = engine::audio::ReferenceAudioFrontend::semantic_features_16k(waveform_16k);
    return {std::move(features.values), std::move(features.attention_mask),
            features.frames, features.feature_dim};
}

IndexTTS2PreparedReferenceAudio prepare_index_tts2_reference_audio(
    const std::vector<float> & samples, int sample_rate, int channels,
    const IndexTTS2S2MelConfig & config, size_t threads, bool speaker_load_semantic) {
    engine::audio::ReferenceAudioFrontendConfig frontend_config;
    frontend_config.mel = mel_config(config);
    frontend_config.max_duration_seconds = 15;
    frontend_config.mel_resample = engine::audio::ReferenceAudioResampleMode::SoxrF32ExactLength;
    frontend_config.speaker_resample = speaker_load_semantic
        ? engine::audio::ReferenceAudioResampleMode::TorchaudioF32
        : engine::audio::ReferenceAudioResampleMode::SoxrF32ExactLength;
    frontend_config.speaker_source = speaker_load_semantic
        ? engine::audio::SpeakerWaveformSource::MelWaveform
        : engine::audio::SpeakerWaveformSource::OriginalInput;
    auto features = engine::audio::ReferenceAudioFrontend(frontend_config)
        .extract(samples, sample_rate, channels, threads);
    IndexTTS2PreparedReferenceAudio out;
    out.waveform_22k = std::move(features.mel_waveform);
    out.waveform_16k = std::move(features.speaker_waveform);
    out.mel = {std::move(features.mel.values), features.mel.mel_bins, features.mel.frames};
    out.campplus_fbank = {std::move(features.campplus.values),
                          features.campplus.frames, features.campplus.feature_dim};
    out.semantic_features = {std::move(features.semantic.values),
                             std::move(features.semantic.attention_mask),
                             features.semantic.frames, features.semantic.feature_dim};
    return out;
}

}  // namespace engine::models::index_tts2
