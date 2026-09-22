#include "engine/community_models/glm_tts/frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/mel_spectrogram_frontend.h"
#include "engine/framework/audio/reference_audio_frontend.h"
#include "engine/framework/audio/resampling.h"

#include <stdexcept>
#include <utility>

namespace engine::models::glm_tts {
GlmTTSMelFeatures compute_glm_tts_prompt_mel(
    const runtime::AudioBuffer & audio) {
    audio::MelSpectrogramFrontendConfig mel;
    mel.sample_rate = 24000;
    mel.n_fft = 1920;
    mel.hop_length = 480;
    mel.win_length = 1920;
    mel.n_mels = 80;
    mel.mel_fmax = 8000.0f;
    mel.stft_pad_mode = audio::STFTPadMode::Constant;
    mel.layout = audio::MelOutputLayout::TimeMajor;
    mel.resample_mode = audio::MelResampleMode::TorchaudioSincF64;
    const auto frontend = audio::get_cached_mel_spectrogram_frontend(mel);
    auto features = frontend->extract_audio(
        audio.samples, audio.sample_rate, audio.channels);
    return {std::move(features.values), features.frames, features.mel_bins};
}

GlmTTSFbankFeatures compute_glm_tts_campplus_fbank(
    const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
        throw std::runtime_error("GLM-TTS requires non-empty reference audio");
    }
    audio::TorchaudioSincHannResampleOptions options;
    options.kernel_mode = audio::TorchaudioSincHannKernelMode::Float64ComputationStoredAsFloat64;
    const auto mono = audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
        audio.samples, audio.sample_rate, audio.channels, 16000, options);
    if (mono.size() < 400) {
        throw std::runtime_error(
            "GLM-TTS reference audio is too short for CAMPPlus");
    }
    auto features = audio::ReferenceAudioFrontend::campplus_fbank_16k(mono);
    return {std::move(features.values), features.frames, features.feature_dim};
}

}  // namespace engine::models::glm_tts
