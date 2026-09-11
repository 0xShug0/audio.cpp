#include "engine/models/seed_vc/whisper_content.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/waveform_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace engine::models::seed_vc {
namespace {

std::vector<float> compute_whisper_log_mel(const std::vector<float> & waveform_16k, size_t threads) {
    constexpr int64_t kSampleRate = 16000;
    constexpr int64_t kNfft = 400;
    constexpr int64_t kHop = 160;
    constexpr int64_t kMels = 80;
    constexpr int64_t kOutputFrames = 3000;
    const engine::audio::STFTConfig stft_config{
        kNfft,
        kHop,
        kNfft,
        true,
        engine::audio::STFTPadMode::Reflect,
        engine::audio::STFTFamily::Kokoro,
    };
    const auto & window = engine::audio::get_cached_stft_window(stft_config);
    const auto magnitude = engine::audio::STFT().compute_magnitude(
        waveform_16k,
        window,
        1,
        static_cast<int64_t>(waveform_16k.size()),
        stft_config,
        threads);
    const int64_t freq_bins = magnitude.shape[1];
    const int64_t stft_frames = magnitude.shape[2];
    if (freq_bins != (kNfft / 2 + 1) || stft_frames <= kOutputFrames) {
        throw std::runtime_error("Seed-VC Whisper frontend STFT shape mismatch");
    }
    static const auto mel_filter = engine::audio::MelFilterbank().build(
        engine::audio::MelFilterbankConfig{kSampleRate, kNfft, kMels, 0.0F, 0.0F, true});
    std::vector<float> log_mel(static_cast<size_t>(kMels * kOutputFrames), 0.0F);
    float max_log = -std::numeric_limits<float>::infinity();
    for (int64_t mel = 0; mel < kMels; ++mel) {
        for (int64_t frame = 0; frame < kOutputFrames; ++frame) {
            float sum = 0.0F;
            for (int64_t freq = 0; freq < freq_bins; ++freq) {
                const float mag = magnitude.values[static_cast<size_t>(freq * stft_frames + frame)];
                sum += mel_filter.values[static_cast<size_t>(mel * freq_bins + freq)] * mag * mag;
            }
            const float value = std::log10(std::max(sum, 1.0e-10F));
            log_mel[static_cast<size_t>(mel * kOutputFrames + frame)] = value;
            max_log = std::max(max_log, value);
        }
    }
    const float floor = max_log - 8.0F;
    for (float & value : log_mel) {
        value = (std::max(value, floor) + 4.0F) / 4.0F;
    }
    return log_mel;
}

}  // namespace

SeedVcWhisperContentEncoder::SeedVcWhisperContentEncoder(
    std::shared_ptr<const engine::assets::TensorSource> source,
    engine::core::BackendConfig backend,
    engine::assets::TensorStorageType storage_type) {
    engine::modules::WhisperFrontendComponentConfig component_config;
    component_config.name = "seed_vc.whisper.encoder";
    component_config.matmul_weight_storage_type = storage_type;
    component_config.conv_weight_storage_type = storage_type;
    frontend_ = engine::modules::WhisperFrontendComponent::load_hf_encoder_layout(
        std::move(source),
        std::move(backend),
        std::move(component_config));
}

SeedVcWhisperContentEncoder::~SeedVcWhisperContentEncoder() = default;
SeedVcWhisperContentEncoder::SeedVcWhisperContentEncoder(SeedVcWhisperContentEncoder &&) noexcept = default;
SeedVcWhisperContentEncoder & SeedVcWhisperContentEncoder::operator=(SeedVcWhisperContentEncoder &&) noexcept = default;

int64_t SeedVcWhisperContentEncoder::channels() const noexcept {
    return frontend_.channels();
}

std::vector<float> SeedVcWhisperContentEncoder::extract_16k_mono(
    const std::vector<float> & waveform_16k,
    size_t threads) const {
    const auto & config = frontend_.config();
    const int64_t wanted_frames = static_cast<int64_t>(waveform_16k.size()) / 320 + 1;
    constexpr size_t kWhisperSamples = 480000;
    const auto log_mel = compute_whisper_log_mel(
        engine::audio::copy_or_zero_pad_samples_to_count(waveform_16k, kWhisperSamples),
        threads);
    const auto full = frontend_.encode_log_mel(log_mel);
    const int64_t frames = std::min<int64_t>(wanted_frames, config.n_audio_ctx);
    std::vector<float> out(static_cast<size_t>(frames * config.n_audio_state), 0.0F);
    std::copy_n(full.begin(), out.size(), out.begin());
    return out;
}

}  // namespace engine::models::seed_vc
