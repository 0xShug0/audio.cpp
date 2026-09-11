#include "engine/community_models/niagara_asr/frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::community_models::niagara_asr {
namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;

std::vector<float> periodic_hann(int64_t win_length) {
    std::vector<float> window(static_cast<size_t>(win_length), 0.0f);
    for (int64_t i = 0; i < win_length; ++i) {
        window[static_cast<size_t>(i)] = static_cast<float>(
            0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) / static_cast<double>(win_length)));
    }
    return window;
}

double hz_to_mel(double hz) {
    return 2595.0 * std::log10(1.0 + hz / 700.0);
}

double mel_to_hz(double mel) {
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

std::vector<float> build_filterbank(int64_t sample_rate, int64_t n_fft, int64_t n_mels) {
    const int64_t freq_bins = n_fft / 2 + 1;
    const double f_min = 0.0;
    const double f_max = static_cast<double>(sample_rate) / 2.0;
    const double mel_min = hz_to_mel(f_min);
    const double mel_max = hz_to_mel(f_max);

    std::vector<double> hz_points(static_cast<size_t>(n_mels + 2), 0.0);
    for (int64_t i = 0; i < n_mels + 2; ++i) {
        const double mel = mel_min + (mel_max - mel_min) *
            static_cast<double>(i) / static_cast<double>(n_mels + 1);
        hz_points[static_cast<size_t>(i)] = mel_to_hz(mel);
    }

    std::vector<float> filterbank(static_cast<size_t>(n_mels * freq_bins), 0.0f);
    const double bin_hz = f_max / static_cast<double>(freq_bins - 1);
    for (int64_t mel = 0; mel < n_mels; ++mel) {
        const double left = hz_points[static_cast<size_t>(mel)];
        const double center = hz_points[static_cast<size_t>(mel + 1)];
        const double right = hz_points[static_cast<size_t>(mel + 2)];
        for (int64_t bin = 0; bin < freq_bins; ++bin) {
            const double hz = bin_hz * static_cast<double>(bin);
            const double up = center > left ? (hz - left) / (center - left) : 0.0;
            const double down = right > center ? (right - hz) / (right - center) : 0.0;
            filterbank[static_cast<size_t>(mel * freq_bins + bin)] =
                static_cast<float>(std::max(0.0, std::min(up, down)));
        }
    }
    return filterbank;
}

}  // namespace

NiagaraFrontend::NiagaraFrontend(std::shared_ptr<const NiagaraAsrAssets> assets)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Niagara ASR frontend requires assets");
    }
    const auto & cfg = assets_->config.frontend;
    window_ = periodic_hann(cfg.win_length);
    mel_filterbank_ = build_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels);
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
    for (int64_t t = 0; t < raw_frames; ++t) {
        for (int64_t m = 0; m < cfg.n_mels; ++m) {
            double energy = 0.0;
            for (int64_t f = 0; f < freq_bins; ++f) {
                const float v = mag.values[static_cast<size_t>(f * raw_frames + t)];
                energy += static_cast<double>(mel_filterbank_[static_cast<size_t>(m * freq_bins + f)]) *
                    static_cast<double>(v);
            }
            out.values[static_cast<size_t>(t * cfg.n_mels + m)] =
                std::log(std::max(energy, 1.0e-12));
        }
    }
    return out;
}

}  // namespace engine::community_models::niagara_asr
