#include "engine/models/lfm_d1_omni/audio.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/errors.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

namespace engine::models::lfm_d1_omni {

D1AudioFrontend::D1AudioFrontend(LfmD1OmniAudioSpec config)
    : config_(std::move(config)), frontend_(config_.frontend) {}

audio::MelSpectrogramFeatures D1AudioFrontend::extract(const runtime::AudioBuffer & input, int threads) {
    if (input.sample_rate <= 0 || input.channels <= 0 || input.samples.empty() ||
        input.samples.size() % static_cast<size_t>(input.channels) != 0) {
        throw runtime::InvalidRequestError("d1: audio input requires samples, a positive rate/channel count and whole frames");
    }
    if (!std::all_of(input.samples.begin(), input.samples.end(), [](float value) { return std::isfinite(value); })) {
        throw runtime::InvalidRequestError("d1: input audio contains non-finite samples");
    }
    const auto feature_start = std::chrono::steady_clock::now();
    // Upstream media.resampled: float32 channel average and torchaudio-default Hann sinc.
    auto samples = audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
        input.samples, input.sample_rate, input.channels, static_cast<int>(config_.frontend.sample_rate),
        audio::torchaudio_sinc_hann_float32_options(), audio::MonoMixAccumulation::Float32);
    const int64_t n_fft = config_.frontend.n_fft, hop = config_.frontend.hop_length;
    const int64_t length = static_cast<int64_t>(samples.size());
    const int64_t frames = 1 + (std::max<int64_t>(0, length - n_fft) + hop - 1) / hop;
    samples.resize(static_cast<size_t>(n_fft + (frames - 1) * hop), 0.0f);
    auto features = frontend_.extract_mono(samples, static_cast<size_t>(std::max(1, threads)));
    if (features.frames != frames || features.mel_bins != config_.frontend.n_mels) {
        throw std::runtime_error("d1: mel frontend output dimensions disagree with whole-window padding");
    }
    for (float & value : features.values) { value = (value + config_.normalization_offset) / config_.normalization_divisor; }
    debug::timing_log_scalar("lfm_d1_omni.audio.features_ms", debug::elapsed_ms(feature_start));

    return features;
}

} // namespace engine::models::lfm_d1_omni
