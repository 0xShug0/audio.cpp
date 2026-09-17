#include "engine/models/niagara_asr/frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::niagara_asr {

NiagaraFrontend::NiagaraFrontend(std::shared_ptr<const NiagaraAsrAssets> assets)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Niagara ASR frontend requires assets");
    }
    const auto & cfg = assets_->config.frontend;
    engine::audio::STFTConfig stft_cfg;
    stft_cfg.win_length = cfg.win_length;
    stft_cfg.family = engine::audio::STFTFamily::Kokoro;
    window_ = engine::audio::get_cached_stft_window(stft_cfg);
    mel_filterbank_ = engine::audio::MelFilterbank().build(
        engine::audio::MelFilterbankConfig{
            cfg.sample_rate,
            cfg.n_fft,
            cfg.n_mels,
            0.0f,
            static_cast<float>(cfg.sample_rate) / 2.0f,
            false,
        });
}

NiagaraFeatures NiagaraFrontend::extract(const runtime::AudioBuffer & audio) const {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
        throw std::runtime_error("Niagara ASR requires non-empty audio with positive sample rate/channels");
    }
    const auto & cfg = assets_->config.frontend;
    auto waveform = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples, audio.sample_rate, audio.channels, static_cast<int>(cfg.sample_rate));
    if (static_cast<int64_t>(waveform.size()) < cfg.win_length) {
        waveform.resize(static_cast<size_t>(cfg.win_length), 0.0f);
    }

    engine::audio::STFTConfig stft_cfg;
    stft_cfg.n_fft = cfg.n_fft;
    stft_cfg.win_length = cfg.win_length;
    stft_cfg.hop_length = cfg.hop_length;
    stft_cfg.center = false;
    stft_cfg.pad_mode = engine::audio::STFTPadMode::Constant;
    stft_cfg.family = engine::audio::STFTFamily::Kokoro;

    const auto mag = engine::audio::STFT().compute_magnitude(
        waveform, window_, 1, static_cast<int64_t>(waveform.size()), stft_cfg);
    const int64_t freq_bins = cfg.n_fft / 2 + 1;
    const int64_t raw_frames = mag.shape.size() >= 3
        ? mag.shape[2]
        : static_cast<int64_t>(mag.values.size()) / freq_bins;
    const int64_t padded_frames = ((raw_frames + 3) / 4) * 4;

    NiagaraFeatures out;
    out.frames = padded_frames;
    out.feature_dim = cfg.n_mels;
    out.values.assign(static_cast<size_t>(padded_frames * cfg.n_mels), 1000.0f);
    const auto mel = engine::audio::MelFilterbank().compute_custom(
        mag.values,
        1,
        freq_bins,
        raw_frames,
        mel_filterbank_);
    for (int64_t t = 0; t < raw_frames; ++t) {
        for (int64_t m = 0; m < cfg.n_mels; ++m) {
            out.values[static_cast<size_t>(t * cfg.n_mels + m)] =
                std::log(std::max(static_cast<double>(mel.values[static_cast<size_t>(m * raw_frames + t)]), 1.0e-12));
        }
    }
    return out;
}

}  // namespace engine::models::niagara_asr
