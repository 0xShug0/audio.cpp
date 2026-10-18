#include "engine/framework/audio/nemo_mel_frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/sampling/noise.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

namespace engine::audio {
namespace {

std::vector<float> precise_symmetric_hann(int64_t length) {
    std::vector<float> window(static_cast<size_t>(length), 0.0f);
    if (length == 1) {
        window[0] = 1.0f;
        return window;
    }
    constexpr long double kPi = 3.14159265358979323846264338327950288L;
    for (int64_t i = 0; i < length; ++i) {
        window[static_cast<size_t>(i)] =
            0.5f - 0.5f * std::cos(2.0L * kPi * static_cast<long double>(i) / static_cast<long double>(length - 1));
    }
    return window;
}

std::vector<float> periodic_hann_double(int64_t length) {
    std::vector<float> window(static_cast<size_t>(length), 0.0f);
    if (length <= 1) {
        if (!window.empty()) window[0] = 1.0f;
        return window;
    }
    constexpr double kPi = 3.14159265358979323846;
    for (int64_t i = 0; i < length; ++i) {
        window[static_cast<size_t>(i)] = static_cast<float>(
            0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) / static_cast<double>(length)));
    }
    return window;
}

AudioTensor build_htk_filterbank(int64_t sample_rate, int64_t n_fft, int64_t n_mels) {
    const int64_t num_bins = n_fft / 2 + 1;
    const double f_max = static_cast<double>(sample_rate) / 2.0;
    const double max_mel = 2595.0 * std::log10(1.0 + f_max / 700.0);
    std::vector<double> edges_hz(static_cast<size_t>(n_mels + 2));
    for (size_t i = 0; i < edges_hz.size(); ++i) {
        const double mel = max_mel * static_cast<double>(i) / static_cast<double>(n_mels + 1);
        edges_hz[i] = 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
    }
    AudioTensor fb;
    fb.shape = {n_mels, num_bins};
    fb.values.assign(static_cast<size_t>(n_mels * num_bins), 0.0f);
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

NemoMelFrontend::NemoMelFrontend(
    NemoMelFrontendConfig config,
    std::vector<float> window_values,
    AudioTensor filterbank_values)
    : config_(std::move(config)) {
    if (config_.sample_rate <= 0 || config_.n_mels <= 0 || config_.stft.n_fft <= 0 ||
        config_.stft.hop_length <= 0 || config_.stft.win_length <= 0 || config_.frame_multiple <= 0 ||
        config_.log_zero_guard <= 0.0f || config_.dither_stddev < 0.0f ||
        config_.stack_frames <= 0 || config_.max_log_drop < 0.0f) {
        throw std::runtime_error("NemoMelFrontend invalid config");
    }
    if (config_.window == MelWindow::FromArgument) {
        if (window_values.size() != static_cast<size_t>(config_.stft.win_length)) {
            throw std::runtime_error("NemoMelFrontend window argument size mismatch");
        }
        window_ = std::move(window_values);
    } else {
        if (!window_values.empty()) {
            throw std::runtime_error("NemoMelFrontend unexpected window argument");
        }
        if (config_.window == MelWindow::SymmetricPrecise) {
            window_ = precise_symmetric_hann(config_.stft.win_length);
        } else if (config_.window == MelWindow::PeriodicF64) {
            window_ = periodic_hann_double(config_.stft.win_length);
        } else {
            window_ = get_cached_stft_window(config_.stft);
        }
    }
    if (config_.mel_bank == MelBank::FromArgument) {
        if (filterbank_values.shape != std::vector<int64_t>{config_.n_mels, config_.stft.n_fft / 2 + 1} ||
            filterbank_values.values.size() !=
                static_cast<size_t>(config_.n_mels * (config_.stft.n_fft / 2 + 1))) {
            throw std::runtime_error("NemoMelFrontend filterbank argument shape mismatch");
        }
        dense_filterbank_ = std::move(filterbank_values);
        if (config_.mel_path == MelPath::SparsePowerLn) {
            sparse_filterbank_ = MelFilterbank().prepare_sparse(dense_filterbank_);
        }
    } else {
        if (!filterbank_values.values.empty() || !filterbank_values.shape.empty()) {
            throw std::runtime_error("NemoMelFrontend unexpected filterbank argument");
        }
        const MelFilterbankConfig mel_config{
            config_.sample_rate, config_.stft.n_fft, config_.n_mels,
            0.0f, static_cast<float>(config_.sample_rate) / 2.0f, true};
        if (config_.mel_bank == MelBank::HTK) {
            dense_filterbank_ = build_htk_filterbank(
                config_.sample_rate, config_.stft.n_fft, config_.n_mels);
        } else if (config_.mel_path == MelPath::LogMelSpectrogram &&
            config_.window == MelWindow::FromArgument) {
            dense_filterbank_ = MelFilterbank().build(mel_config);
        } else if (config_.mel_path != MelPath::LogMelSpectrogram) {
            sparse_filterbank_ = MelFilterbank().build_sparse(mel_config);
        }
    }
    if (config_.mel_path == MelPath::LogMelSpectrogram && config_.log_zero_guard != 0x1p-24f) {
        throw std::runtime_error("LogMelSpectrogram requires its 2^-24 log guard");
    }
}

NemoMelFeatures NemoMelFrontend::extract_audio(
    const std::vector<float> & interleaved,
    int sample_rate,
    int channels,
    const NemoMelRunConfig & run,
    size_t threads) const {
    return extract_mono(prepare_audio(interleaved, sample_rate, channels), run, threads);
}

std::vector<float> NemoMelFrontend::prepare_audio(
    const std::vector<float> & interleaved,
    int sample_rate,
    int channels) const {
    if (sample_rate <= 0 || channels <= 0 || interleaved.empty() ||
        interleaved.size() % static_cast<size_t>(channels) != 0) {
        throw std::runtime_error("NemoMelFrontend invalid audio input");
    }
    std::vector<float> mono;
    if (config_.input_rate == MelInputRate::RequireMatch) {
        if (sample_rate != config_.sample_rate) {
            throw std::runtime_error("NemoMelFrontend unexpected sample rate");
        }
        mono = mixdown_interleaved_to_mono_average(interleaved, channels);
    } else {
        mono = convert_interleaved_audio_to_mono_linear_resampled(
            interleaved, sample_rate, channels, static_cast<int>(config_.sample_rate));
    }
    return mono;
}

NemoMelFeatures NemoMelFrontend::extract_mono(
    std::vector<float> mono,
    const NemoMelRunConfig & run,
    size_t threads) const {
    if (mono.empty()) {
        throw std::runtime_error("NemoMelFrontend empty mono waveform");
    }
    int64_t stacked_frames = -1;
    if (config_.stack_frames > 1) {
        const int64_t mel_frames = static_cast<int64_t>(mono.size()) / config_.stft.hop_length;
        stacked_frames = config_.stack_frames *
            ((mel_frames + config_.stack_frames - 1) / config_.stack_frames);
        const int64_t need_samples = (stacked_frames - 1) * config_.stft.hop_length + 1;
        if (static_cast<int64_t>(mono.size()) < need_samples) {
            mono.resize(static_cast<size_t>(need_samples), 0.0f);
        }
    }
    if (config_.wave_scale == WaveScale::DivideByMaxPlusEps) {
        const float scale = 1.0f / (*std::max_element(mono.begin(), mono.end()) + 1.0e-3f);
        for (float & sample : mono) sample *= scale;
    }
    if (config_.dither_method == DitherMethod::BoxMuller16 && config_.dither_stddev > 0.0f) {
        if (mono.size() < 16) {
            throw std::runtime_error("Cohere dither requires at least 16 samples");
        }
        std::mt19937 rng(static_cast<uint32_t>(mono.size()));
        std::vector<float> noise(mono.size());
        const auto uniform = [&]() { return static_cast<float>(rng() & 0xffffffu) * 0x1p-24f; };
        for (float & value : noise) value = uniform();
        const auto normal_block = [](float * values) {
            for (size_t i = 0; i < 8; ++i) {
                const float radius = std::sqrt(-2.0f * std::log(1.0f - values[i]));
                const float theta = 6.2831853071795864769f * values[i + 8];
                values[i] = radius * std::cos(theta);
                values[i + 8] = radius * std::sin(theta);
            }
        };
        for (size_t i = 0; i + 16 <= noise.size(); i += 16) normal_block(noise.data() + i);
        if (noise.size() % 16 != 0) {
            auto * tail = noise.data() + noise.size() - 16;
            for (size_t i = 0; i < 16; ++i) tail[i] = uniform();
            normal_block(tail);
        }
        for (size_t i = 0; i < mono.size(); ++i) mono[i] += config_.dither_stddev * noise[i];
    } else if (config_.dither_method == DitherMethod::Normal && config_.dither_stddev > 0.0f) {
        const auto noise = engine::sampling::generate_normal_noise(
            mono.size(), static_cast<uint32_t>(mono.size()));
        for (size_t i = 0; i < mono.size(); ++i) mono[i] += config_.dither_stddev * noise[i];
    } else if (config_.dither_method == DitherMethod::None && config_.dither_stddev != 0.0f) {
        throw std::runtime_error("NemoMelFrontend nonzero dither requires a dither mode");
    }
    apply_preemphasis_in_place(mono, config_.preemphasis);

    STFTConfig stft = config_.stft;
    stft.center = run.center;
    const auto mel_start = std::chrono::steady_clock::now();
    AudioTensor mel;
    if (config_.mel_path == MelPath::LogMelSpectrogram) {
        if (config_.mel_bank == MelBank::Slaney &&
            config_.window != MelWindow::FromArgument) {
            mel = LogMelSpectrogram().compute(
                mono, 1, static_cast<int64_t>(mono.size()), config_.sample_rate,
                stft, config_.n_mels, threads);
        } else {
            mel = LogMelSpectrogram().compute(
                mono, window_, 1, static_cast<int64_t>(mono.size()), stft,
                dense_filterbank_, threads);
        }
    } else if (config_.mel_path == MelPath::ComplexPowerLn) {
        const auto complex = STFT().compute_complex(
            mono, window_, 1, static_cast<int64_t>(mono.size()), stft, threads);
        const int64_t freq_bins = complex.shape.at(1);
        const int64_t frames = complex.shape.at(2);
        std::vector<float> power(static_cast<size_t>(freq_bins * frames), 0.0f);
        for (int64_t t = 0; t < frames; ++t) {
            for (int64_t f = 0; f < freq_bins; ++f) {
                const size_t idx = static_cast<size_t>((f * frames + t) * 2);
                const float re = complex.values[idx];
                const float im = complex.values[idx + 1];
                power[static_cast<size_t>(f * frames + t)] = re * re + im * im;
            }
        }
        mel = MelFilterbank().compute_custom(power, 1, freq_bins, frames,
                                             config_.mel_bank == MelBank::FromArgument
                                                 ? dense_filterbank_ : sparse_filterbank_.dense);
        for (float & value : mel.values) value = std::log(value + config_.log_zero_guard);
    } else if (config_.mel_path == MelPath::DensePowerLog10) {
        const auto magnitude = STFT().compute_magnitude(
            mono, window_, 1, static_cast<int64_t>(mono.size()), stft, threads);
        const int64_t freq_bins = magnitude.shape.at(1);
        const int64_t frames = magnitude.shape.at(2);
        mel.shape = {1, config_.n_mels, frames};
        mel.values.assign(static_cast<size_t>(config_.n_mels * frames), 0.0f);
        for (int64_t m = 0; m < config_.n_mels; ++m) {
            for (int64_t t = 0; t < frames; ++t) {
                float energy = 0.0f;
                for (int64_t f = 0; f < freq_bins; ++f) {
                    const float mag = magnitude.values[static_cast<size_t>(f * frames + t)];
                    const float power = mag * mag;
                    energy += dense_filterbank_.values[static_cast<size_t>(m * freq_bins + f)] * power;
                }
                if (energy < config_.log_zero_guard) energy = config_.log_zero_guard;
                mel.values[static_cast<size_t>(m * frames + t)] = std::log10(energy);
            }
        }
    } else {
        const auto magnitude = STFT().compute_magnitude(
            mono, window_, 1, static_cast<int64_t>(mono.size()), stft, threads);
        mel = MelFilterbank().compute_custom_sparse_from_magnitude(
            magnitude.values, 1, magnitude.shape.at(1), magnitude.shape.at(2),
            magnitude.shape.at(2), sparse_filterbank_);
        for (float & value : mel.values) value = std::log(value + config_.log_zero_guard);
    }
    const double mel_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - mel_start).count();

    if (mel.shape.size() != 3 || mel.shape[0] != 1 || mel.shape[1] != config_.n_mels) {
        throw std::runtime_error("NemoMelFrontend mel shape mismatch");
    }
    const int64_t raw = mel.shape[2];
    if (stacked_frames > raw) {
        throw std::runtime_error("NemoMelFrontend produced too few frames for stacking");
    }
    int64_t valid = raw;
    if (run.valid_frame_rule == ValidFrameRule::FloorHops) {
        valid = static_cast<int64_t>(mono.size()) / stft.hop_length;
    } else if (run.valid_frame_rule == ValidFrameRule::CeilHops) {
        valid = (static_cast<int64_t>(mono.size()) + stft.hop_length - 1) / stft.hop_length;
    } else if (run.valid_frame_rule == ValidFrameRule::FullWindows) {
        valid = static_cast<int64_t>(mono.size()) < stft.n_fft ? 0
            : (static_cast<int64_t>(mono.size()) - stft.n_fft) / stft.hop_length + 1;
    }
    if (stacked_frames >= 0) valid = stacked_frames;
    if (run.valid_frames_override >= 0) {
        if (raw < run.valid_frames_override) {
            throw std::runtime_error("NemoMelFrontend produced too few frames");
        }
        valid = run.valid_frames_override;
    }
    valid = std::clamp<int64_t>(valid, 0, raw);

    const auto norm_start = std::chrono::steady_clock::now();
    if (config_.norm == MelNorm::PerBinF32) {
        mel = FeatureNormalizer().compute(
            mel.values, {valid}, 1, config_.n_mels, raw,
            FeatureNormalizeType::PerFeature).normalized;
    } else if (config_.norm == MelNorm::PerBinF64) {
        std::vector<float> normalized(mel.values.size(), 0.0f);
        if (valid > 0) {
            for (int64_t m = 0; m < config_.n_mels; ++m) {
                double sum = 0.0;
                for (int64_t t = 0; t < valid; ++t) sum += mel.values[static_cast<size_t>(m * raw + t)];
                const double mean = sum / static_cast<double>(valid);
                double variance = 0.0;
                for (int64_t t = 0; t < valid; ++t) {
                    const double diff = mel.values[static_cast<size_t>(m * raw + t)] - mean;
                    variance += diff * diff;
                }
                double stddev = valid > 1 ? std::sqrt(variance / static_cast<double>(valid - 1)) : 0.0;
                if (std::isnan(stddev)) stddev = 0.0;
                stddev += 1.0e-5f;
                for (int64_t t = 0; t < valid; ++t) {
                    const size_t idx = static_cast<size_t>(m * raw + t);
                    normalized[idx] = static_cast<float>((mel.values[idx] - mean) / stddev);
                }
            }
        }
        mel.values = std::move(normalized);
    } else if (config_.norm == MelNorm::PerBinMixed) {
        for (int64_t m = 0; m < config_.n_mels; ++m) {
            double sum = 0.0;
            for (int64_t t = 0; t < valid; ++t) sum += mel.values[static_cast<size_t>(m * raw + t)];
            const float mean = valid > 0 ? static_cast<float>(sum / static_cast<double>(valid)) : 0.0f;
            double variance = 0.0;
            for (int64_t t = 0; t < valid; ++t) {
                const float centered = mel.values[static_cast<size_t>(m * raw + t)] - mean;
                variance += static_cast<double>(centered) * static_cast<double>(centered);
            }
            const float stddev = valid > 1
                ? std::sqrt(static_cast<float>(variance / static_cast<double>(valid - 1))) + 1.0e-5f
                : 1.0e-5f;
            for (int64_t t = 0; t < raw; ++t) {
                float & value = mel.values[static_cast<size_t>(m * raw + t)];
                value = (value - mean) / stddev;
                if (t >= valid) value = 0.0f;
            }
        }
    } else {
        for (int64_t m = 0; m < config_.n_mels; ++m) {
            for (int64_t t = valid; t < raw; ++t) {
                mel.values[static_cast<size_t>(m * raw + t)] = 0.0f;
            }
        }
    }
    const double norm_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - norm_start).count();

    if (config_.mel_path == MelPath::DensePowerLog10) {
        float max_logmel = -1e30f;
        for (int64_t m = 0; m < config_.n_mels; ++m) {
            for (int64_t t = 0; t < valid; ++t) {
                max_logmel = std::max(max_logmel, mel.values[static_cast<size_t>(m * raw + t)]);
            }
        }
        const float floor_thresh = max_logmel - config_.max_log_drop;
        std::vector<float> logmel(static_cast<size_t>(config_.n_mels * valid), 0.0f);
        for (int64_t m = 0; m < config_.n_mels; ++m) {
            for (int64_t t = 0; t < valid; ++t) {
                float value = mel.values[static_cast<size_t>(m * raw + t)];
                if (value < floor_thresh) value = floor_thresh;
                logmel[static_cast<size_t>(m * valid + t)] =
                    value * config_.log_scale + config_.log_bias;
            }
        }
        std::vector<float> deltas(static_cast<size_t>(config_.n_mels * valid), 0.0f);
        if (config_.add_deltas) {
            for (int64_t m = 0; m < config_.n_mels; ++m) {
                const size_t offset = static_cast<size_t>(m * valid);
                if (valid > 1) {
                    deltas[offset] = (logmel[offset + 1] - logmel[offset]) * 0.5f;
                    for (int64_t t = 1; t < valid - 1; ++t) {
                        deltas[offset + static_cast<size_t>(t)] =
                            (logmel[offset + static_cast<size_t>(t + 1)] -
                             logmel[offset + static_cast<size_t>(t - 1)]) * 0.5f;
                    }
                    deltas[offset + static_cast<size_t>(valid - 1)] =
                        (logmel[offset + static_cast<size_t>(valid - 1)] -
                         logmel[offset + static_cast<size_t>(valid - 2)]) * 0.5f;
                }
            }
        }
        if (valid % config_.stack_frames != 0) {
            throw std::runtime_error("NemoMelFrontend stacked frames are not aligned");
        }
        NemoMelFeatures out;
        out.frames = valid / config_.stack_frames;
        out.raw_frames = raw;
        out.valid_frames = out.frames;
        out.feature_size = config_.n_mels *
            (config_.add_deltas ? 2 : 1) * config_.stack_frames;
        out.mel_ms = mel_ms;
        out.normalize_ms = norm_ms;
        out.values.assign(static_cast<size_t>(out.frames * out.feature_size), 0.0f);
        const int64_t channels_per_step = config_.n_mels * (config_.add_deltas ? 2 : 1);
        for (int64_t t_out = 0; t_out < out.frames; ++t_out) {
            float * dst_frame = &out.values[static_cast<size_t>(t_out * out.feature_size)];
            for (int64_t step = 0; step < config_.stack_frames; ++step) {
                const int64_t t_in = t_out * config_.stack_frames + step;
                float * dst_step = dst_frame + step * channels_per_step;
                for (int64_t m = 0; m < config_.n_mels; ++m) {
                    dst_step[m] = logmel[static_cast<size_t>(m * valid + t_in)];
                    if (config_.add_deltas) {
                        dst_step[config_.n_mels + m] =
                            deltas[static_cast<size_t>(m * valid + t_in)];
                    }
                }
            }
        }
        return out;
    }

    const int64_t base_frames = config_.pad_basis == PadBasis::ValidFrames ? valid : raw;
    const int64_t frames = ((base_frames + config_.frame_multiple - 1) / config_.frame_multiple) * config_.frame_multiple;
    NemoMelFeatures out;
    out.frames = frames;
    out.raw_frames = raw;
    out.valid_frames = valid;
    out.feature_size = config_.n_mels;
    out.mel_ms = mel_ms;
    out.normalize_ms = norm_ms;
    out.values.assign(static_cast<size_t>(frames * config_.n_mels), 0.0f);
    const int64_t copy_frames = std::min(frames, raw);
    if (config_.layout == MelLayout::TimeMajor) {
        for (int64_t t = 0; t < copy_frames; ++t) {
            for (int64_t m = 0; m < config_.n_mels; ++m) {
                out.values[static_cast<size_t>(t * config_.n_mels + m)] =
                    mel.values[static_cast<size_t>(m * raw + t)];
            }
        }
    } else {
        for (int64_t m = 0; m < config_.n_mels; ++m) {
            for (int64_t t = 0; t < copy_frames; ++t) {
                out.values[static_cast<size_t>(m * frames + t)] =
                    mel.values[static_cast<size_t>(m * raw + t)];
            }
        }
    }
    return out;
}

}  // namespace engine::audio
