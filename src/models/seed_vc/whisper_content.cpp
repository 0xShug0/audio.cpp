#include "engine/models/seed_vc/whisper_content.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/mel_spectrogram_frontend.h"
#include "engine/framework/audio/waveform_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace engine::models::seed_vc {

std::vector<float> compute_whisper_log_mel(const std::vector<float> & waveform_16k, size_t threads) {
    constexpr int64_t kOutputFrames = 3000;
    if (1 + static_cast<int64_t>(waveform_16k.size()) / 160 <= kOutputFrames) {
        throw std::runtime_error("Seed-VC Whisper frontend STFT shape mismatch");
    }
    engine::audio::MelSpectrogramFrontendConfig config;
    config.sample_rate = 16000;
    config.n_fft = 400;
    config.hop_length = 160;
    config.win_length = 400;
    config.n_mels = 80;
    config.stft_center = true;
    config.waveform_padding = engine::audio::MelWaveformPadding::None;
    config.spectrum_mode = engine::audio::MelSpectrumMode::PowerDuringProjection;
    config.value_transform = engine::audio::MelValueTransform::Log10;
    config.log_floor = 1.0e-10;
    config.max_frames = kOutputFrames;
    config.log_dynamic_range = 8.0f;
    config.log_shift = 4.0f;
    config.log_divisor = 4.0f;
    return engine::audio::get_cached_mel_spectrogram_frontend(config)
        ->extract_mono(waveform_16k, threads).values;
}

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
