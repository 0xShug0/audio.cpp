#include "engine/models/niagara_asr/frontend.h"

#include <stdexcept>
#include <utility>

namespace engine::models::niagara_asr {

NiagaraFrontend::NiagaraFrontend(std::shared_ptr<const NiagaraAsrAssets> assets)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Niagara ASR frontend requires assets");
    }
    const auto & source = assets_->config.frontend;
    engine::audio::MelSpectrogramFrontendConfig config;
    config.sample_rate = source.sample_rate;
    config.n_fft = source.n_fft;
    config.hop_length = source.hop_length;
    config.win_length = source.win_length;
    config.n_mels = source.n_mels;
    config.mel_fmax = static_cast<float>(source.sample_rate) / 2.0f;
    config.filterbank_normalization = engine::audio::MelFilterbankNormalization::None;
    config.stft_pad_mode = engine::audio::STFTPadMode::Constant;
    config.waveform_padding = engine::audio::MelWaveformPadding::None;
    config.filterbank_projection = engine::audio::MelFilterbankProjection::DenseLongDouble;
    config.log_floor = 1.0e-12;
    config.log_precision = engine::audio::MelLogPrecision::F64;
    config.layout = engine::audio::MelOutputLayout::TimeMajor;
    config.resample_mode = engine::audio::MelResampleMode::Linear;
    config.minimum_samples = source.win_length;
    config.frame_multiple = 4;
    config.frame_pad_value = 1000.0f;
    frontend_ = engine::audio::get_cached_mel_spectrogram_frontend(config);
}

NiagaraFeatures NiagaraFrontend::extract(const runtime::AudioBuffer & audio) const {
    auto features = frontend_->extract_audio(audio.samples, audio.sample_rate, audio.channels);
    return {std::move(features.values), features.frames, features.mel_bins};
}

}  // namespace engine::models::niagara_asr
