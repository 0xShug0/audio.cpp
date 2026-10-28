#include "engine/community_models/lfm3_audio/features.h"

#include "engine/community_models/lfm3_audio/debug.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/mel_spectrogram_frontend.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/runtime/errors.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::community_models::lfm3_audio {
namespace {

constexpr int kSampleRate = 16000;
constexpr int64_t kWindow = 400;  // n_fft = win_length
constexpr int64_t kHop = 160;
constexpr int64_t kMelBins = 128;
constexpr int64_t kTokenFrames = 8;  // the downsampler's three stride-2 convs
// Lfm3AudioFeatureExtractor: clamp(log10(mel), -12), then (x + 4) / 4.
constexpr double kLogFloor = 1e-12;
constexpr float kLogShift = 4.0f;
constexpr float kLogDivisor = 4.0f;

audio::MelSpectrogramFrontendConfig frontend_config() {
    audio::MelSpectrogramFrontendConfig config;
    config.sample_rate = kSampleRate;
    config.n_fft = config.win_length = kWindow;
    config.hop_length = kHop;
    config.n_mels = kMelBins;
    config.mel_fmax = 8000.0f;
    config.window = audio::MelHannWindow::Periodic;
    config.stft_center = false;
    config.waveform_padding = audio::MelWaveformPadding::None;
    config.spectrum_mode = audio::MelSpectrumMode::PowerDuringProjection;
    config.value_transform = audio::MelValueTransform::Log10;
    config.log_floor = kLogFloor;
    return config;
}

}  // namespace

std::vector<float> lfm3_audio_mono_16k(const runtime::AudioBuffer & audio) {
    if (audio.samples.empty() || audio.channels <= 0 || audio.sample_rate <= 0 ||
        audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw runtime::InvalidRequestError("LFM3-Audio requires non-empty audio");
    }

    // One NaN or Inf sample turns every feature of its chunk into NaN.
    if (!std::all_of(audio.samples.begin(), audio.samples.end(), [](float value) { return std::isfinite(value); })) {
        throw runtime::InvalidRequestError("LFM3-Audio input audio has non-finite samples");
    }

    auto mono = audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    if (audio.sample_rate == kSampleRate) {
        return mono;
    }

    // HF loads audio with librosa.load(sr=16000), i.e. soxr_hq: the same libsoxr HQ recipe, with librosa's
    // fix_length to ceil(n * 16000 / rate). The torchaudio sinc differs by ~6% in the top mel bands, so it
    // is only the fallback for a host without libsoxr.
    audio::SoxrResampleOptions options;
    options.output_length_policy = audio::SoxrOutputLengthPolicy::ExactExpected;
    options.warning_context = "LFM3-Audio input";
    options.fallback_description = "the torchaudio sinc resampler";
    if (auto resampled = audio::try_resample_mono_soxr(mono, audio.sample_rate, kSampleRate, options)) {
        return std::move(*resampled);
    }

    return audio::resample_mono_torchaudio_sinc_hann(mono, audio.sample_rate, kSampleRate, audio::torchaudio_sinc_hann_float32_options());
}

int64_t lfm3_feature_frames(int64_t samples) {
    return 1 + (std::max<int64_t>(samples - kWindow, 0) + kHop - 1) / kHop;
}

int64_t lfm3_audio_tokens(int64_t frames) {
    return (frames + kTokenFrames - 1) / kTokenFrames;
}

Lfm3FeatureExtractor::Lfm3FeatureExtractor() {
    (void)audio::get_cached_mel_spectrogram_frontend(frontend_config());  // validates the config and builds the filterbank once
}

Lfm3Features Lfm3FeatureExtractor::extract(const std::vector<float> & mono16k, size_t threads) const {
    // Right-pad with zeros to whole frames, at least one (no centering).
    const int64_t samples = static_cast<int64_t>(mono16k.size());
    Lfm3Features out;
    out.frames = lfm3_feature_frames(samples);
    std::vector<float> padded(static_cast<size_t>(kWindow + (out.frames - 1) * kHop), 0.0f);
    std::copy(mono16k.begin(), mono16k.end(), padded.begin());

    auto mel = audio::get_cached_mel_spectrogram_frontend(frontend_config())->extract_mono(padded, threads);
    if (mel.frames != out.frames || mel.mel_bins != kMelBins) {
        throw std::runtime_error("LFM3-Audio features have " + std::to_string(mel.frames) + " frames, expected " +
                                 std::to_string(out.frames));
    }

    out.values = std::move(mel.values);
    for (auto & value : out.values) {
        value = (value + kLogShift) / kLogDivisor;
    }

    if (lfm3_dump_enabled()) {
        lfm3_dump("features", out.values.data(), {kMelBins, out.frames});
    }
    return out;
}

}  // namespace engine::community_models::lfm3_audio
