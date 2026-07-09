#include "engine/models/hviske_asr/frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/sampling/noise.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::hviske_asr {
namespace {

using Clock = std::chrono::steady_clock;

std::vector<float> build_hann_window(int64_t win_length) {
    std::vector<float> window(static_cast<size_t>(win_length), 0.0f);
    if (win_length == 1) {
        window[0] = 1.0f;
        return window;
    }
    constexpr long double kPi = 3.14159265358979323846264338327950288L;
    for (int64_t i = 0; i < win_length; ++i) {
        window[static_cast<size_t>(i)] =
            0.5f - 0.5f * std::cos(2.0L * kPi * static_cast<long double>(i) / static_cast<long double>(win_length - 1));
    }
    return window;
}

void validate_audio(const engine::runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("Hviske ASR audio sample_rate must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error("Hviske ASR audio channels must be positive");
    }
    if (audio.samples.empty()) {
        throw std::runtime_error("Hviske ASR audio is empty");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("Hviske ASR interleaved audio size is not divisible by channel count");
    }
}

std::vector<float> normalize_audio(const engine::runtime::AudioBuffer & audio, int sample_rate) {
    validate_audio(audio);
    return engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        sample_rate);
}

int64_t frontend_valid_frames(int64_t samples, const HviskeFrontendConfig & config) {
    return samples <= 0 ? 0 : samples / config.hop_length;
}

void apply_dither_and_preemphasis(
    std::vector<float> & samples,
    int64_t valid_samples,
    const HviskeFrontendConfig & config) {
    if (config.dither > 0.0f && valid_samples > 0) {
        auto noise = engine::sampling::generate_normal_noise(
            static_cast<size_t>(valid_samples),
            static_cast<uint32_t>(valid_samples));
        for (int64_t i = 0; i < valid_samples; ++i) {
            samples[static_cast<size_t>(i)] += config.dither * noise[static_cast<size_t>(i)];
        }
    }
    if (config.preemph != 0.0f && valid_samples > 1) {
        for (int64_t i = valid_samples - 1; i > 0; --i) {
            samples[static_cast<size_t>(i)] -= config.preemph * samples[static_cast<size_t>(i - 1)];
        }
    }
    for (int64_t i = valid_samples; i < static_cast<int64_t>(samples.size()); ++i) {
        samples[static_cast<size_t>(i)] = 0.0f;
    }
}

}  // namespace

HviskeFrontend::HviskeFrontend(std::shared_ptr<const HviskeAssets> assets)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Hviske frontend requires assets");
    }
    const auto & config = assets_->config.frontend;
    filterbank_ = engine::audio::MelFilterbank().build_sparse(
        engine::audio::MelFilterbankConfig{
            config.sample_rate,
            config.n_fft,
            config.features,
            0.0f,
            static_cast<float>(config.sample_rate) / 2.0f,
            true,
        });
    window_ = build_hann_window(config.win_length);
    if (assets_->model_weights != nullptr &&
        assets_->model_weights->has_tensor("preprocessor.featurizer.fb") &&
        assets_->model_weights->has_tensor("preprocessor.featurizer.window")) {
        const int64_t freq_bins = config.n_fft / 2 + 1;
        engine::audio::AudioTensor filterbank;
        filterbank.shape = {config.features, freq_bins};
        filterbank.values = assets_->model_weights->require_f32(
            "preprocessor.featurizer.fb",
            {1, config.features, freq_bins});
        filterbank_ = engine::audio::MelFilterbank().prepare_sparse(filterbank);
        window_ = assets_->model_weights->require_f32(
            "preprocessor.featurizer.window",
            {config.win_length});
    }
}

HviskeFrontendFeatures HviskeFrontend::extract(const engine::runtime::AudioBuffer & audio) const {
    const auto wall_start = Clock::now();
    const auto & config = assets_->config.frontend;
    auto samples = normalize_audio(audio, static_cast<int>(config.sample_rate));
    const int64_t valid_samples = static_cast<int64_t>(samples.size());
    apply_dither_and_preemphasis(samples, valid_samples, config);

    const engine::audio::STFTConfig stft_config{
        config.n_fft,
        config.hop_length,
        config.win_length,
        true,
        engine::audio::STFTPadMode::Constant,
        engine::audio::STFTFamily::Default,
    };
    auto complex = engine::audio::STFT().compute_complex(samples, window_, 1, valid_samples, stft_config);
    if (complex.shape.size() != 4 || complex.shape[0] != 1 || complex.shape[3] != 2) {
        throw std::runtime_error("Hviske frontend STFT returned unexpected rank");
    }
    const int64_t freq_bins = complex.shape[1];
    const int64_t stft_frames = complex.shape[2];
    std::vector<float> power(static_cast<size_t>(stft_frames * freq_bins), 0.0f);
    for (int64_t t = 0; t < stft_frames; ++t) {
        for (int64_t f = 0; f < freq_bins; ++f) {
            const size_t idx = static_cast<size_t>((f * stft_frames + t) * 2);
            const float real = complex.values[idx];
            const float imag = complex.values[idx + 1];
            power[static_cast<size_t>(f * stft_frames + t)] = real * real + imag * imag;
        }
    }

    auto mel = engine::audio::MelFilterbank().compute_custom(
        power,
        1,
        freq_bins,
        stft_frames,
        filterbank_.dense);
    if (mel.shape.size() != 3 || mel.shape[1] != config.features || mel.shape[2] != stft_frames) {
        throw std::runtime_error("Hviske frontend mel shape mismatch");
    }
    for (float & value : mel.values) {
        value = std::log(value + config.log_zero_guard);
    }

    const int64_t valid_frames = frontend_valid_frames(valid_samples, config);
    for (int64_t mel_bin = 0; mel_bin < config.features; ++mel_bin) {
        double sum = 0.0;
        for (int64_t t = 0; t < valid_frames; ++t) {
            sum += mel.values[static_cast<size_t>(mel_bin * stft_frames + t)];
        }
        const float mean = valid_frames > 0 ? static_cast<float>(sum / static_cast<double>(valid_frames)) : 0.0f;
        double variance = 0.0;
        for (int64_t t = 0; t < valid_frames; ++t) {
            const float centered = mel.values[static_cast<size_t>(mel_bin * stft_frames + t)] - mean;
            variance += static_cast<double>(centered) * static_cast<double>(centered);
        }
        const float stddev = valid_frames > 1
            ? std::sqrt(static_cast<float>(variance / static_cast<double>(valid_frames - 1))) + 1.0e-5f
            : 1.0e-5f;
        for (int64_t t = 0; t < stft_frames; ++t) {
            float & value = mel.values[static_cast<size_t>(mel_bin * stft_frames + t)];
            value = (value - mean) / stddev;
            if (t >= valid_frames) {
                value = 0.0f;
            }
        }
    }

    int64_t padded_frames = stft_frames;
    if (config.pad_to > 0 && padded_frames % config.pad_to != 0) {
        padded_frames += config.pad_to - (padded_frames % config.pad_to);
    }
    if (padded_frames != stft_frames) {
        std::vector<float> padded(static_cast<size_t>(config.features * padded_frames), 0.0f);
        for (int64_t mel_bin = 0; mel_bin < config.features; ++mel_bin) {
            for (int64_t t = 0; t < stft_frames; ++t) {
                padded[static_cast<size_t>(mel_bin * padded_frames + t)] =
                    mel.values[static_cast<size_t>(mel_bin * stft_frames + t)];
            }
        }
        mel.values = std::move(padded);
    }

    HviskeFrontendFeatures out;
    out.values = std::move(mel.values);
    out.feature_dim = config.features;
    out.frames = padded_frames;
    out.valid_frames = valid_frames;
    debug::timing_log_scalar("hviske_asr.frontend_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    debug::trace_log_scalar("hviske_asr.frontend_frames", out.frames);
    debug::trace_log_scalar("hviske_asr.frontend_valid_frames", out.valid_frames);
    return out;
}

}  // namespace engine::models::hviske_asr
