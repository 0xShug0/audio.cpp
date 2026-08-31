#include "engine/community_models/granite5asr/frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::community_models::granite5asr {
namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<float> make_periodic_hann_window(int64_t win_length) {
    std::vector<float> window(static_cast<size_t>(win_length), 0.0f);
    if (win_length <= 1) {
        if (!window.empty()) {
            window[0] = 1.0f;
        }
        return window;
    }
    for (int64_t i = 0; i < win_length; ++i) {
        window[static_cast<size_t>(i)] = static_cast<float>(
            0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) / static_cast<double>(win_length)));
    }
    return window;
}

double hz_to_htk_mel(double hz) {
    return 2595.0 * std::log10(1.0 + hz / 700.0);
}

double htk_mel_to_hz(double mel) {
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

audio::AudioTensor build_htk_mel_filterbank(int64_t sample_rate, int64_t n_fft, int64_t n_mels) {
    const int64_t num_bins = n_fft / 2 + 1;
    const double f_min = 0.0;
    const double f_max = static_cast<double>(sample_rate) / 2.0;

    const double min_mel = hz_to_htk_mel(f_min);
    const double max_mel = hz_to_htk_mel(f_max);

    // Triangle edges stay in Hz. Snapping them to integer FFT bins collapses the narrow
    // low-frequency filters: at 16 kHz / n_fft 512 the bin spacing is 31.25 Hz while the
    // lowest mel bands are ~20 Hz wide, so several of them land on the same bin (mel[2]
    // became an all-zero row) and no longer match the torchaudio filterbank the encoder
    // was trained against.
    std::vector<double> edges_hz(static_cast<size_t>(n_mels + 2));
    for (size_t i = 0; i < edges_hz.size(); ++i) {
        const double mel = min_mel + (max_mel - min_mel) * static_cast<double>(i) / static_cast<double>(n_mels + 1);
        edges_hz[i] = htk_mel_to_hz(mel);
    }

    audio::AudioTensor fb;
    fb.shape = {n_mels, num_bins};
    fb.values.assign(static_cast<size_t>(n_mels * num_bins), 0.0f);

    // Frequency of FFT bin k, matching torch.linspace(0, sample_rate / 2, num_bins).
    const double bin_hz = f_max / static_cast<double>(num_bins - 1);

    for (int64_t m = 1; m <= n_mels; ++m) {
        const double f_left = edges_hz[static_cast<size_t>(m - 1)];
        const double f_center = edges_hz[static_cast<size_t>(m)];
        const double f_right = edges_hz[static_cast<size_t>(m + 1)];

        for (int64_t k = 0; k < num_bins; ++k) {
            const double hz = bin_hz * static_cast<double>(k);
            const double rising = f_center > f_left ? (hz - f_left) / (f_center - f_left) : 0.0;
            const double falling = f_right > f_center ? (f_right - hz) / (f_right - f_center) : 0.0;
            const double weight = std::min(rising, falling);
            if (weight > 0.0) {
                fb.values[static_cast<size_t>((m - 1) * num_bins + k)] = static_cast<float>(weight);
            }
        }
    }
    return fb;
}

}  // namespace

Granite5Frontend::Granite5Frontend(std::shared_ptr<const Granite5ASRAssets> assets)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Granite 5 ASR frontend requires valid assets");
    }
    const auto & config = assets_->config.frontend;
    mel_filterbank_ = build_htk_mel_filterbank(config.sample_rate, config.n_fft, config.n_mels);
    window_ = make_periodic_hann_window(config.win_length);
}

std::vector<float> Granite5Frontend::prepare_waveform(const runtime::AudioBuffer & audio) const {
    if (audio.sample_rate <= 0 || audio.channels <= 0) {
        throw std::runtime_error("Granite 5 ASR audio requires positive sample rate and channels");
    }
    if (audio.samples.empty()) {
        throw std::runtime_error("Granite 5 ASR audio input is empty");
    }
    const auto & config = assets_->config.frontend;
    return engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        static_cast<int>(config.sample_rate));
}

Granite5FrontendFeatures Granite5Frontend::extract(const runtime::AudioBuffer & audio) const {
    return extract_waveform(prepare_waveform(audio));
}

Granite5FrontendFeatures Granite5Frontend::extract_waveform(const std::vector<float> & waveform) const {
    if (waveform.empty()) {
        throw std::runtime_error("Granite 5 ASR waveform is empty");
    }
    const auto & config = assets_->config.frontend;
    const int64_t hop = config.hop_length;
    const int64_t s = config.stack_factor;
    const int64_t T = static_cast<int64_t>(waveform.size());
    const int64_t mel_frames = T / hop;
    const int64_t n_frames = s * ((mel_frames + s - 1) / s);
    const int64_t need_samples = (n_frames - 1) * hop + 1;

    std::vector<float> padded_waveform = waveform;
    if (static_cast<int64_t>(padded_waveform.size()) < need_samples) {
        padded_waveform.resize(static_cast<size_t>(need_samples), 0.0f);
    }

    audio::STFTConfig stft_cfg;
    stft_cfg.n_fft = config.n_fft;
    stft_cfg.hop_length = config.hop_length;
    stft_cfg.win_length = config.win_length;
    stft_cfg.center = true;
    stft_cfg.pad_mode = audio::STFTPadMode::Reflect;

    audio::STFT stft;
    const auto mag_tensor = stft.compute_magnitude(
        padded_waveform,
        window_,
        1,
        static_cast<int64_t>(padded_waveform.size()),
        stft_cfg);

    const int64_t freq_bins = config.n_fft / 2 + 1;
    const int64_t available_frames = mag_tensor.shape.size() >= 3 ? mag_tensor.shape[2] : (static_cast<int64_t>(mag_tensor.values.size()) / freq_bins);
    const int64_t frames_to_use = std::min(n_frames, available_frames);

    const int64_t n_mels = config.n_mels;
    std::vector<float> logmel(static_cast<size_t>(n_mels * n_frames), 0.0f);

    float max_logmel = -1e30f;
    for (int64_t m = 0; m < n_mels; ++m) {
        for (int64_t t = 0; t < frames_to_use; ++t) {
            float energy = 0.0f;
            for (int64_t f = 0; f < freq_bins; ++f) {
                const float mag = mag_tensor.values[static_cast<size_t>(f * available_frames + t)];
                const float power = mag * mag;
                energy += mel_filterbank_.values[static_cast<size_t>(m * freq_bins + f)] * power;
            }
            if (energy < 1e-10f) {
                energy = 1e-10f;
            }
            const float log_val = std::log10(energy);
            logmel[static_cast<size_t>(m * n_frames + t)] = log_val;
            if (log_val > max_logmel) {
                max_logmel = log_val;
            }
        }
        for (int64_t t = frames_to_use; t < n_frames; ++t) {
            logmel[static_cast<size_t>(m * n_frames + t)] = logmel[static_cast<size_t>(m * n_frames + (frames_to_use > 0 ? frames_to_use - 1 : 0))];
        }
    }

    const float floor_thresh = max_logmel - config.logmel_floor_db;
    for (float & val : logmel) {
        if (val < floor_thresh) {
            val = floor_thresh;
        }
        val = (val / 4.0f) + 1.0f;
    }

    std::vector<float> deltas(static_cast<size_t>(n_mels * n_frames), 0.0f);
    if (config.deltas) {
        for (int64_t m = 0; m < n_mels; ++m) {
            const size_t offset = static_cast<size_t>(m * n_frames);
            if (n_frames == 1) {
                deltas[offset] = 0.0f;
            } else {
                deltas[offset] = (logmel[offset + 1] - logmel[offset]) * 0.5f;
                for (int64_t t = 1; t < n_frames - 1; ++t) {
                    deltas[offset + static_cast<size_t>(t)] =
                        (logmel[offset + static_cast<size_t>(t + 1)] - logmel[offset + static_cast<size_t>(t - 1)]) * 0.5f;
                }
                deltas[offset + static_cast<size_t>(n_frames - 1)] =
                    (logmel[offset + static_cast<size_t>(n_frames - 1)] - logmel[offset + static_cast<size_t>(n_frames - 2)]) * 0.5f;
            }
        }
    }

    const int64_t out_frames = n_frames / s;
    const int64_t feature_dim = n_mels * (config.deltas ? 2 : 1) * s;

    Granite5FrontendFeatures features;
    features.frames = out_frames;
    features.feature_dim = feature_dim;
    features.values.assign(static_cast<size_t>(out_frames * feature_dim), 0.0f);

    const int64_t channels_per_step = n_mels * (config.deltas ? 2 : 1);
    for (int64_t t_out = 0; t_out < out_frames; ++t_out) {
        float * dst_frame = &features.values[static_cast<size_t>(t_out * feature_dim)];
        for (int64_t step = 0; step < s; ++step) {
            const int64_t t_in = t_out * s + step;
            float * dst_step = dst_frame + step * channels_per_step;
            for (int64_t m = 0; m < n_mels; ++m) {
                dst_step[m] = logmel[static_cast<size_t>(m * n_frames + t_in)];
                if (config.deltas) {
                    dst_step[n_mels + m] = deltas[static_cast<size_t>(m * n_frames + t_in)];
                }
            }
        }
    }

    return features;
}

}  // namespace engine::community_models::granite5asr
