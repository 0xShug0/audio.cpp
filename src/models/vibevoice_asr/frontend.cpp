#include "engine/models/vibevoice_asr/frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::vibevoice_asr {
namespace {

void validate_audio(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0) {
        throw std::runtime_error("VibeVoice-ASR audio sample_rate and channels must be positive");
    }
    if (audio.samples.empty()) {
        throw std::runtime_error("VibeVoice-ASR audio is empty");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("VibeVoice-ASR interleaved audio is not divisible by channel count");
    }
}

}  // namespace

VibeVoiceASRFrontend::VibeVoiceASRFrontend(std::shared_ptr<const VibeVoiceAssets> assets)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("VibeVoice-ASR frontend requires assets");
    }
}

runtime::AudioBuffer VibeVoiceASRFrontend::normalize(const runtime::AudioBuffer & audio) const {
    validate_audio(audio);
    auto mono = engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    const auto & config = assets_->processor.audio_processor;
    if (audio.sample_rate != config.sample_rate) {
        engine::audio::SoxrResampleOptions options;
        options.profile = engine::audio::SoxrResampleProfile::QualityOnly;
        options.output_length_policy = engine::audio::SoxrOutputLengthPolicy::ExactExpected;
        options.output_padding = 256;
        options.reject_empty_output = true;
        options.warning_context = "VibeVoice-ASR audio";
        options.fallback_description = "linear resampling";
        mono = engine::audio::resample_mono_soxr_or_linear(mono, audio.sample_rate, config.sample_rate, options);
    }
    if (config.normalize_audio) {
        double sum = 0.0;
        for (const float sample : mono) {
            sum += static_cast<double>(sample) * static_cast<double>(sample);
        }
        const float rms = std::sqrt(static_cast<float>(sum / std::max<size_t>(mono.size(), 1)));
        const float target = std::pow(10.0f, config.target_db_fs / 20.0f);
        const float gain = target / (rms + config.eps);
        float max_abs = 0.0f;
        for (float & sample : mono) {
            sample *= gain;
            max_abs = std::max(max_abs, std::abs(sample));
        }
        if (max_abs > 1.0f) {
            const float scale = max_abs + config.eps;
            for (float & sample : mono) {
                sample /= scale;
            }
        }
    }
    return runtime::AudioBuffer{config.sample_rate, 1, std::move(mono)};
}

}  // namespace engine::models::vibevoice_asr
