#include "engine/models/audio_flamingo/frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"

#include <algorithm>
#include <stdexcept>

namespace engine::models::audio_flamingo {

AudioFlamingoFrontend::AudioFlamingoFrontend(std::shared_ptr<const AudioFlamingoAssets> assets)
    : assets_(std::move(assets)) {
    const auto & c = assets_->config.frontend;
    audio::MelSpectrogramFrontendConfig mel;
    mel.sample_rate = c.sample_rate;
    mel.n_fft = c.n_fft;
    mel.hop_length = c.hop_length;
    mel.win_length = c.n_fft;
    mel.n_mels = c.feature_size;
    mel.mel_fmax = c.sample_rate / 2.0F;
    mel.stft_center = true;
    mel.waveform_padding = audio::MelWaveformPadding::None;
    mel.filterbank_projection = audio::MelFilterbankProjection::SparseF32;
    mel.spectrum_mode = audio::MelSpectrumMode::PowerDuringProjection;
    mel.value_transform = audio::MelValueTransform::Log10;
    mel.log_floor = 1e-10;
    mel.log_dynamic_range = 8.0F;
    mel.log_shift = 4.0F;
    mel.log_divisor = 4.0F;
    mel.drop_last_frames = 1;
    mel_ = audio::get_cached_mel_spectrogram_frontend(mel);
}

AudioFlamingoAudioFeatures AudioFlamingoFrontend::extract(const runtime::AudioBuffer & audio) const {
    const auto & c = assets_->config.frontend;
    if (audio.samples.empty() || audio.sample_rate <= 0 || audio.channels <= 0 ||
        audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("Audio Flamingo requires non-empty, valid audio");
    }
    auto samples = engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    if (audio.sample_rate != c.sample_rate) {
        engine::audio::SoxrResampleOptions options;
        options.output_length_policy = engine::audio::SoxrOutputLengthPolicy::ExactExpected;
        options.require_full_input = true;
        auto converted = engine::audio::try_resample_mono_soxr(samples, audio.sample_rate, c.sample_rate, options);
        if (!converted) {
            throw std::runtime_error("Audio Flamingo requires SOXR to resample non-16-kHz audio");
        }
        samples = std::move(*converted);
    }
    if (samples.size() > static_cast<size_t>(c.max_audio_length_sec * c.sample_rate)) {
        throw std::runtime_error("Audio Flamingo input exceeds the processor's maximum audio duration");
    }
    const int64_t window_samples = c.chunk_length_sec * c.sample_rate;
    const int64_t total = static_cast<int64_t>(samples.size());
    AudioFlamingoAudioFeatures out;
    out.batch = out.windows = (total + window_samples - 1) / window_samples;
    out.frames = window_samples / c.hop_length;
    out.mel_bins = c.feature_size;
    out.values.resize(static_cast<size_t>(out.batch * out.frames * out.mel_bins));
    out.attention_mask.assign(static_cast<size_t>(out.batch * out.frames), 0);
    for (int64_t w = 0; w < out.windows; ++w) {
        const int64_t valid = std::min(window_samples, total - w * window_samples);
        std::vector<float> chunk(static_cast<size_t>(window_samples), 0.0F);
        std::copy_n(samples.data() + w * window_samples, valid, chunk.data());
        const int64_t frames = (valid + c.hop_length - 1) / c.hop_length;
        out.post_lengths.push_back(audio_flamingo_post_length(frames));
        std::fill_n(out.attention_mask.data() + w * out.frames, frames, 1);
        const auto mel = mel_->extract_mono(chunk);
        if (mel.frames != out.frames) {
            throw std::runtime_error("Audio Flamingo mel frame count mismatch");
        }
        std::copy(mel.values.begin(), mel.values.end(), out.values.begin() + w * out.frames * out.mel_bins);
    }
    return out;
}

}  // namespace engine::models::audio_flamingo
