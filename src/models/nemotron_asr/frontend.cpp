#include "engine/models/nemotron_asr/frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::nemotron_asr {
namespace {

using Clock = std::chrono::steady_clock;

void validate_audio(const engine::runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0) {
        throw std::runtime_error("Nemotron ASR audio requires positive sample_rate and channels");
    }
    if (audio.samples.empty()) {
        throw std::runtime_error("Nemotron ASR audio is empty");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("Nemotron ASR interleaved audio size is not divisible by channel count");
    }
}

std::vector<float> make_hann_window(int64_t win_length) {
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

int64_t valid_feature_frames(int64_t samples, const NemotronFrontendConfig & config, bool center) {
    if (samples <= 0) {
        return 0;
    }
    if (center) {
        return (samples + config.n_fft - config.n_fft) / config.hop_length;
    }
    if (samples < config.n_fft) {
        return 0;
    }
    return (samples - config.n_fft) / config.hop_length + 1;
}

std::vector<float> prepare_waveform_impl(const engine::runtime::AudioBuffer & audio, const NemotronFrontendConfig & config) {
    validate_audio(audio);
    return engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        static_cast<int>(config.sample_rate));
}

}  // namespace

NemotronFrontend::NemotronFrontend(std::shared_ptr<const NemotronAssets> assets)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Nemotron ASR frontend requires assets");
    }
    const auto & config = assets_->config.frontend;
    filterbank_ = engine::audio::MelFilterbank().build_sparse({
        config.sample_rate,
        config.n_fft,
        config.feature_size,
        0.0f,
        static_cast<float>(config.sample_rate) / 2.0f,
        true,
    });
    window_ = make_hann_window(config.win_length);
}

NemotronFrontendFeatures NemotronFrontend::extract(
    const engine::runtime::AudioBuffer & audio,
    bool center) const {
    return extract_waveform(prepare_waveform(audio), center);
}

std::vector<float> NemotronFrontend::prepare_waveform(const engine::runtime::AudioBuffer & audio) const {
    return prepare_waveform_impl(audio, assets_->config.frontend);
}

NemotronFrontendFeatures NemotronFrontend::extract_waveform(
    const std::vector<float> & waveform,
    bool center) const {
    const auto wall_start = Clock::now();
    const auto & config = assets_->config.frontend;
    std::vector<float> processed = waveform;
    if (config.preemphasis != 0.0f && processed.size() > 1) {
        for (int64_t i = static_cast<int64_t>(processed.size()) - 1; i > 0; --i) {
            processed[static_cast<size_t>(i)] -= config.preemphasis * processed[static_cast<size_t>(i - 1)];
        }
    }
    const int64_t valid_samples = static_cast<int64_t>(processed.size());
    const engine::audio::STFTConfig stft_config{
        config.n_fft,
        config.hop_length,
        config.win_length,
        center,
        engine::audio::STFTPadMode::Constant,
        engine::audio::STFTFamily::Default,
    };
    const auto magnitude = engine::audio::STFT().compute_magnitude(processed, window_, 1, valid_samples, stft_config);
    if (magnitude.shape.size() != 3 || magnitude.shape[0] != 1) {
        throw std::runtime_error("Nemotron ASR frontend STFT returned unexpected rank");
    }
    const int64_t freq_bins = magnitude.shape[1];
    const int64_t frames = magnitude.shape[2];
    auto mel = engine::audio::MelFilterbank().compute_custom_sparse_from_magnitude(
        magnitude.values,
        1,
        freq_bins,
        frames,
        frames,
        filterbank_);
    if (mel.shape.size() != 3 || mel.shape[1] != config.feature_size || mel.shape[2] != frames) {
        throw std::runtime_error("Nemotron ASR frontend mel shape mismatch");
    }
    const int64_t valid_frames = std::clamp<int64_t>(valid_feature_frames(valid_samples, config, center), 0, frames);
    std::vector<float> time_major(static_cast<size_t>(frames * config.feature_size), 0.0f);
    for (int64_t t = 0; t < frames; ++t) {
        for (int64_t m = 0; m < config.feature_size; ++m) {
            float value = std::log(mel.values[static_cast<size_t>(m * frames + t)] + config.log_zero_guard);
            if (t >= valid_frames) {
                value = 0.0f;
            }
            time_major[static_cast<size_t>(t * config.feature_size + m)] = value;
        }
    }

    NemotronFrontendFeatures out;
    out.values = std::move(time_major);
    out.frames = frames;
    out.valid_frames = valid_frames;
    out.feature_dim = config.feature_size;
    debug::timing_log_scalar("nemotron_asr.frontend_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    debug::trace_log_scalar("nemotron_asr.frontend.frames", out.frames);
    debug::trace_log_scalar("nemotron_asr.frontend.valid_frames", out.valid_frames);
    debug::trace_log_scalar("nemotron_asr.frontend.center", center);
    return out;
}

}  // namespace engine::models::nemotron_asr
