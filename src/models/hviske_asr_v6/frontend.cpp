#include "engine/models/hviske_asr_v6/frontend.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::hviske_asr_v6 {

HviskeV6WhisperFrontend::HviskeV6WhisperFrontend(int64_t mel_bins) {
    audio::MelSpectrogramFrontendConfig config;
    config.sample_rate = 16000;
    config.n_fft = config.win_length = 400;
    config.hop_length = 160;
    config.n_mels = mel_bins;
    config.mel_fmax = 8000.0f;
    config.stft_center = true;
    config.waveform_padding = audio::MelWaveformPadding::None;
    config.spectrum_mode = audio::MelSpectrumMode::PowerBeforeProjection;
    config.value_transform = audio::MelValueTransform::Log10;
    config.log_floor = 1.0e-10;
    config.log_dynamic_range = 8.0f;
    config.log_shift = 4.0f;
    config.log_divisor = 4.0f;
    config.drop_last_frames = 1;
    mel_ = audio::get_cached_mel_spectrogram_frontend(config);
}

HviskeV6Features HviskeV6WhisperFrontend::extract(
    const std::vector<float> & mono_16k, size_t threads) const {
    if (mono_16k.size() > 30 * 16000) {
        throw std::runtime_error("Hviske v6 frontend requires chunks of at most 30 seconds");
    }
    HviskeV6Features result;
    result.valid_frames = (static_cast<int64_t>(mono_16k.size()) + 159) / 160;
    result.frames = std::max<int64_t>(200, ((result.valid_frames + 99) / 100) * 100);
    // Upstream pads the waveform before STFT, not the resulting mel features.
    std::vector<float> padded(static_cast<size_t>(result.frames * 160), 0.0f);
    std::copy(mono_16k.begin(), mono_16k.end(), padded.begin());
    auto features = mel_->extract_mono(padded, threads);
    if (features.frames != result.frames) {
        throw std::runtime_error("Hviske v6 frontend frame count mismatch");
    }
    result.values = std::move(features.values);
    return result;
}

}  // namespace engine::models::hviske_asr_v6
