#include "engine/models/xtts_v2/audio_features.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/resampling.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::xtts_v2 {
namespace {

std::vector<float> mono(const runtime::AudioBuffer & audio) {
    if (audio.channels <= 0 || audio.samples.empty() ||
        audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("XTTS v2 reference audio has an invalid shape");
    }
    return audio.channels == 1
        ? audio.samples
        : engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
}

std::vector<float> resample(const std::vector<float> & input, int source_rate, int target_rate) {
    if (source_rate == target_rate) return input;
    auto options = engine::audio::torchaudio_sinc_hann_float32_options();
    return engine::audio::resample_mono_torchaudio_sinc_hann(input, source_rate, target_rate, options);
}

XttsV2MelFeatures power_mel(
    const std::vector<float> & waveform,
    int64_t sample_rate,
    int64_t n_fft,
    int64_t hop,
    int64_t win,
    int64_t n_mels,
    float fmax,
    const std::vector<float> & window,
    const std::vector<float> & filterbank,
    size_t threads) {
    const engine::audio::STFTConfig stft{
        n_fft, hop, win, true, engine::audio::STFTPadMode::Reflect,
        engine::audio::STFTFamily::Default,
    };
    const auto magnitude = engine::audio::STFT{}.compute_magnitude(
        waveform, window, 1, static_cast<int64_t>(waveform.size()), stft, threads);
    if (magnitude.shape.size() != 3) throw std::runtime_error("XTTS v2 STFT returned an invalid shape");
    const int64_t bins = n_fft / 2 + 1;
    const int64_t frames = magnitude.shape[2];
    if (static_cast<int64_t>(filterbank.size()) != bins * n_mels) {
        throw std::runtime_error("XTTS v2 mel filterbank shape mismatch");
    }
    (void) sample_rate;
    (void) fmax;
    XttsV2MelFeatures out;
    out.channels = n_mels;
    out.frames = frames;
    out.values.assign(static_cast<size_t>(n_mels * frames), 0.0F);
    for (int64_t m = 0; m < n_mels; ++m) {
        for (int64_t t = 0; t < frames; ++t) {
            double energy = 0.0;
            for (int64_t f = 0; f < bins; ++f) {
                const float value = magnitude.values[static_cast<size_t>(f * frames + t)];
                energy += static_cast<double>(filterbank[static_cast<size_t>(m * bins + f)]) * value * value;
            }
            out.values[static_cast<size_t>(m * frames + t)] = static_cast<float>(energy);
        }
    }
    return out;
}

}  // namespace

XttsV2PreparedReference prepare_xtts_v2_reference(const runtime::AudioBuffer & audio) {
    auto waveform = mono(audio);
    for (float & value : waveform) value = std::max(-1.0F, std::min(1.0F, value));
    // Coqui caps each input reference at 30 seconds before joining references.
    const size_t max_source = static_cast<size_t>(audio.sample_rate) * 30U;
    if (waveform.size() > max_source) waveform.resize(max_source);
    XttsV2PreparedReference out;
    out.waveform_22050 = resample(waveform, audio.sample_rate, 22050);
    out.waveform_16000 = resample(waveform, audio.sample_rate, 16000);
    return out;
}

XttsV2MelFeatures compute_xtts_v2_conditioning_mel(
    const std::vector<float> & waveform_22050,
    const std::vector<float> & mel_stats,
    size_t threads) {
    if (mel_stats.size() != 80) throw std::runtime_error("XTTS v2 conditioning requires 80 mel norms");
    const engine::audio::STFTConfig stft{2048, 256, 1024, true, engine::audio::STFTPadMode::Reflect, engine::audio::STFTFamily::Kokoro};
    const auto & window = engine::audio::get_cached_stft_window(stft);
    const auto filterbank = engine::audio::MelFilterbank{}.build({22050, 2048, 80, 0.0F, 8000.0F, true});

    // The caller supplies one chunk. XTTS v2 runs this frontend and the Perceiver
    // independently for each (normally four-second) chunk, then averages latents.
    auto out = power_mel(waveform_22050, 22050, 2048, 256, 1024, 80, 8000.0F, window, filterbank.values, threads);
    for (int64_t m = 0; m < out.channels; ++m) {
        const float norm = mel_stats[static_cast<size_t>(m)];
        if (norm == 0.0F) throw std::runtime_error("XTTS v2 mel norm contains zero");
        for (int64_t t = 0; t < out.frames; ++t) {
            auto & value = out.values[static_cast<size_t>(m * out.frames + t)];
            value = std::log(std::max(value, 1.0e-5F)) / norm;
        }
    }
    return out;
}

XttsV2MelFeatures compute_xtts_v2_speaker_mel(
    const std::vector<float> & waveform_16000,
    const std::vector<float> & window,
    const std::vector<float> & mel_filterbank,
    size_t threads) {
    if (waveform_16000.size() < 2 || window.size() != 400 || mel_filterbank.size() != 257U * 64U) {
        throw std::runtime_error("XTTS v2 speaker frontend inputs are invalid");
    }
    // ResNetSpeakerEncoder's embedded torchaudio frontend applies reflection-pad
    // pre-emphasis [-0.97, 1] before a power mel spectrogram.
    std::vector<float> emphasized(waveform_16000.size());
    emphasized[0] = waveform_16000[0] - 0.97F * waveform_16000[1];
    for (size_t i = 1; i < waveform_16000.size(); ++i) {
        emphasized[i] = waveform_16000[i] - 0.97F * waveform_16000[i - 1];
    }
    auto out = power_mel(emphasized, 16000, 512, 160, 400, 64, 8000.0F, window, mel_filterbank, threads);
    for (auto & value : out.values) value = std::log(value + 1.0e-6F);

    // InstanceNorm1d normalizes each mel channel across time without affine terms.
    for (int64_t m = 0; m < out.channels; ++m) {
        double mean = 0.0;
        for (int64_t t = 0; t < out.frames; ++t) mean += out.values[static_cast<size_t>(m * out.frames + t)];
        mean /= static_cast<double>(out.frames);
        double variance = 0.0;
        for (int64_t t = 0; t < out.frames; ++t) {
            const double delta = out.values[static_cast<size_t>(m * out.frames + t)] - mean;
            variance += delta * delta;
        }
        variance /= static_cast<double>(out.frames);
        const float inv = static_cast<float>(1.0 / std::sqrt(variance + 1.0e-5));
        for (int64_t t = 0; t < out.frames; ++t) {
            auto & value = out.values[static_cast<size_t>(m * out.frames + t)];
            value = (value - static_cast<float>(mean)) * inv;
        }
    }
    return out;
}

}  // namespace engine::models::xtts_v2
