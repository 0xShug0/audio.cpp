#include "engine/community_models/confucius4_r2t2/frontend_whisper.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::community_models::confucius4_r2t2 {
namespace {

constexpr int kMinInputSamples = 8000;
using Clock = std::chrono::steady_clock;

void validate_audio_input(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("R2T2 ASR audio sample_rate must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error("R2T2 ASR audio channels must be positive");
    }
    if (audio.samples.empty()) {
        throw std::runtime_error("R2T2 ASR audio is empty");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("R2T2 ASR interleaved audio size is not divisible by channel count");
    }
}

std::vector<float> normalize_audio(const runtime::AudioBuffer & audio, int sample_rate) {
    validate_audio_input(audio);
    auto mono = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        sample_rate);
    engine::audio::normalize_peak_to_unit_range_and_clamp_in_place(mono);
    if (mono.size() < kMinInputSamples) {
        mono.resize(kMinInputSamples, 0.0F);
    }
    return mono;
}

engine::audio::WhisperLogMelConfig make_extractor_config(const std::shared_ptr<const R2T2ASRAssets> & assets) {
    if (assets == nullptr) {
        throw std::runtime_error("R2T2 ASR Whisper frontend requires assets");
    }
    const auto & config = assets->config.frontend;
    return {
        config.sample_rate,
        config.n_fft,
        config.hop_length,
        config.feature_size,
        engine::audio::STFTFamily::Default,
    };
}

}  // namespace

R2T2ASRWhisperFrontend::R2T2ASRWhisperFrontend(std::shared_ptr<const R2T2ASRAssets> assets)
    : assets_(std::move(assets)),
      extractor_(make_extractor_config(assets_)) {}

R2T2ASRAudioFeatures R2T2ASRWhisperFrontend::extract(const runtime::AudioBuffer & audio) const {
    const auto normalize_start = Clock::now();
    const auto & config = assets_->config.frontend;
    if (config.sample_rate <= 0 || config.feature_size <= 0 || config.hop_length <= 0 || config.n_fft <= 0) {
        throw std::runtime_error("R2T2 ASR Whisper frontend config is invalid");
    }
    auto samples = normalize_audio(audio, config.sample_rate);
    const auto normalize_end = Clock::now();

    const auto feature_start = Clock::now();
    auto features = extractor_.compute(samples);
    const auto feature_end = Clock::now();

    R2T2ASRAudioFeatures out;
    out.values = std::move(features.values);
    out.attention_mask.assign(static_cast<size_t>(features.frames), 1);
    out.mel_bins = features.mel_bins;
    out.frames = features.frames;
    out.encoder_tokens = confucius4_r2t2_audio_encoder_token_count(out.frames);
    debug::timing_log_scalar("confucius4_r2t2.frontend.normalize_ms", engine::debug::elapsed_ms(normalize_start, normalize_end));
    debug::timing_log_scalar("confucius4_r2t2.frontend.log_mel_ms", engine::debug::elapsed_ms(feature_start, feature_end));
    return out;
}

}  // namespace engine::community_models::confucius4_r2t2
