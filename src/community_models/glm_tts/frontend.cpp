#include "engine/community_models/glm_tts/frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/resampling.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace engine::models::glm_tts {
namespace {

std::vector<float> reflect_pad(
    const std::vector<float> & input,
    int64_t left,
    int64_t right) {
    if (input.size() < 2) {
        throw std::runtime_error(
            "GLM-TTS prompt audio is too short for reflect padding");
    }
    std::vector<float> out(
        static_cast<size_t>(
            left + static_cast<int64_t>(input.size()) + right),
        0.0F);
    const int64_t size = static_cast<int64_t>(input.size());
    auto reflected_index = [size](int64_t index) {
        while (index < 0 || index >= size) {
            if (index < 0) {
                index = -index;
            } else {
                index = 2 * size - 2 - index;
            }
        }
        return index;
    };
    for (int64_t index = -left; index < size + right; ++index) {
        out[static_cast<size_t>(index + left)] =
            input[static_cast<size_t>(reflected_index(index))];
    }
    return out;
}

struct KaldiFilterbankKey {
    int64_t sample_rate = 0;
    int64_t n_fft = 0;
    int64_t n_mels = 0;

    bool operator==(const KaldiFilterbankKey & other) const noexcept {
        return sample_rate == other.sample_rate &&
            n_fft == other.n_fft &&
            n_mels == other.n_mels;
    }
};

struct KaldiFilterbankKeyHash {
    size_t operator()(const KaldiFilterbankKey & key) const noexcept {
        size_t seed = std::hash<int64_t>{}(key.sample_rate);
        seed ^= std::hash<int64_t>{}(key.n_fft) +
            0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int64_t>{}(key.n_mels) +
            0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

std::vector<float> make_povey_window(int64_t size) {
    constexpr float kPi = 3.14159265358979323846F;
    std::vector<float> out(static_cast<size_t>(size), 0.0F);
    for (int64_t index = 0; index < size; ++index) {
        const float hann =
            0.5F -
            0.5F * std::cos(
                       2.0F * kPi * static_cast<float>(index) /
                       static_cast<float>(size - 1));
        out[static_cast<size_t>(index)] = std::pow(hann, 0.85F);
    }
    return out;
}

std::vector<float> make_kaldi_filterbank(
    int64_t sample_rate,
    int64_t n_fft,
    int64_t n_mels) {
    const int64_t bins = n_fft / 2 + 1;
    const float nyquist = static_cast<float>(sample_rate) * 0.5F;
    const float low_mel =
        1127.0F * std::log(1.0F + 20.0F / 700.0F);
    const float high_mel =
        1127.0F * std::log(1.0F + nyquist / 700.0F);
    const float delta =
        (high_mel - low_mel) / static_cast<float>(n_mels + 1);
    std::vector<float> out(
        static_cast<size_t>(n_mels * bins), 0.0F);
    for (int64_t mel = 0; mel < n_mels; ++mel) {
        const float left = low_mel + static_cast<float>(mel) * delta;
        const float center =
            low_mel + static_cast<float>(mel + 1) * delta;
        const float right =
            low_mel + static_cast<float>(mel + 2) * delta;
        for (int64_t bin = 0; bin < bins - 1; ++bin) {
            const float hz =
                static_cast<float>(bin * sample_rate) /
                static_cast<float>(n_fft);
            const float point =
                1127.0F * std::log(1.0F + hz / 700.0F);
            out[static_cast<size_t>(mel * bins + bin)] =
                std::max(
                    0.0F,
                    std::min(
                        (point - left) / (center - left),
                        (right - point) / (right - center)));
        }
    }
    return out;
}

const std::vector<float> & cached_povey_window(int64_t size) {
    static std::mutex mutex;
    static std::unordered_map<int64_t, std::vector<float>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = cache.find(size);
    if (found != cache.end()) {
        return found->second;
    }
    return cache.emplace(size, make_povey_window(size)).first->second;
}

const std::vector<float> & cached_kaldi_filterbank(
    int64_t sample_rate,
    int64_t n_fft,
    int64_t n_mels) {
    static std::mutex mutex;
    static std::unordered_map<
        KaldiFilterbankKey,
        std::vector<float>,
        KaldiFilterbankKeyHash>
        cache;
    const KaldiFilterbankKey key{sample_rate, n_fft, n_mels};
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = cache.find(key);
    if (found != cache.end()) {
        return found->second;
    }
    return cache
        .emplace(key, make_kaldi_filterbank(sample_rate, n_fft, n_mels))
        .first->second;
}

}  // namespace

std::vector<float> glm_tts_audio_mono_resampled(
    const runtime::AudioBuffer & audio,
    int sample_rate) {
    if (audio.sample_rate <= 0 || audio.channels <= 0 ||
        audio.samples.empty()) {
        throw std::runtime_error(
            "GLM-TTS requires non-empty reference audio");
    }
    auto mono = audio::mixdown_interleaved_to_mono_average(
        audio.samples, audio.channels);
    if (audio.sample_rate == sample_rate) {
        return mono;
    }
    audio::TorchaudioSincHannResampleOptions options;
    options.kernel_mode =
        audio::TorchaudioSincHannKernelMode::
            Float64ComputationStoredAsFloat64;
    return audio::resample_mono_torchaudio_sinc_hann(
        mono, audio.sample_rate, sample_rate, options);
}

GlmTTSMelFeatures compute_glm_tts_prompt_mel(
    const runtime::AudioBuffer & audio) {
    constexpr int64_t kSampleRate = 24000;
    constexpr int64_t kNfft = 1920;
    constexpr int64_t kHop = 480;
    constexpr int64_t kMels = 80;
    constexpr int64_t kPad = (kNfft - kHop) / 2;
    const auto mono =
        glm_tts_audio_mono_resampled(audio, kSampleRate);
    const auto padded = reflect_pad(mono, kPad, kPad);
    const audio::STFTConfig config{
        kNfft,
        kHop,
        kNfft,
        false,
        audio::STFTPadMode::Constant,
        audio::STFTFamily::Kokoro};
    const auto & window = audio::get_cached_stft_window(config);
    const auto magnitude = audio::STFT().compute_magnitude(
        padded,
        window,
        1,
        static_cast<int64_t>(padded.size()),
        config);
    const int64_t frames = magnitude.shape[2];
    const int64_t bins = kNfft / 2 + 1;
    const auto filterbank = audio::MelFilterbank().build(
        {kSampleRate, kNfft, kMels, 0.0F, 8000.0F, true});
    GlmTTSMelFeatures out;
    out.frames = frames;
    out.values.assign(
        static_cast<size_t>(frames * kMels), 0.0F);
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t mel = 0; mel < kMels; ++mel) {
            float value = 0.0F;
            for (int64_t bin = 0; bin < bins; ++bin) {
                value +=
                    filterbank.values[
                        static_cast<size_t>(mel * bins + bin)] *
                    magnitude.values[
                        static_cast<size_t>(bin * frames + frame)];
            }
            out.values[
                static_cast<size_t>(frame * kMels + mel)] =
                std::log(std::max(value, 1.0e-5F));
        }
    }
    return out;
}

GlmTTSFbankFeatures compute_glm_tts_campplus_fbank(
    const runtime::AudioBuffer & audio) {
    constexpr int64_t kSampleRate = 16000;
    constexpr int64_t kWindow = 400;
    constexpr int64_t kHop = 160;
    constexpr int64_t kNfft = 512;
    constexpr int64_t kMels = 80;
    constexpr float kPreemphasis = 0.97F;
    const auto mono =
        glm_tts_audio_mono_resampled(audio, kSampleRate);
    if (static_cast<int64_t>(mono.size()) < kWindow) {
        throw std::runtime_error(
            "GLM-TTS reference audio is too short for CAMPPlus");
    }
    const int64_t frames =
        1 + (static_cast<int64_t>(mono.size()) - kWindow) / kHop;
    const auto & window = cached_povey_window(kWindow);
    const auto & filterbank =
        cached_kaldi_filterbank(kSampleRate, kNfft, kMels);
    std::vector<float> batches(
        static_cast<size_t>(frames * kNfft), 0.0F);
    for (int64_t frame = 0; frame < frames; ++frame) {
        const int64_t offset = frame * kHop;
        float mean = 0.0F;
        for (int64_t index = 0; index < kWindow; ++index) {
            mean += mono[static_cast<size_t>(offset + index)];
        }
        mean /= static_cast<float>(kWindow);
        for (int64_t index = 0; index < kWindow; ++index) {
            const float current =
                mono[static_cast<size_t>(offset + index)] - mean;
            const float previous = index == 0
                ? current
                : mono[static_cast<size_t>(offset + index - 1)] - mean;
            batches[static_cast<size_t>(frame * kNfft + index)] =
                (current - kPreemphasis * previous) *
                window[static_cast<size_t>(index)];
        }
    }
    std::vector<float> ones(static_cast<size_t>(kNfft), 1.0F);
    const audio::STFTConfig config{
        kNfft,
        kNfft,
        kNfft,
        false,
        audio::STFTPadMode::Constant,
        audio::STFTFamily::Default};
    const auto magnitude = audio::STFT().compute_magnitude(
        batches, ones, frames, kNfft, config);
    const int64_t bins = kNfft / 2 + 1;
    GlmTTSFbankFeatures out;
    out.frames = frames;
    out.values.assign(
        static_cast<size_t>(frames * kMels), 0.0F);
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t mel = 0; mel < kMels; ++mel) {
            float energy = 0.0F;
            for (int64_t bin = 0; bin < bins; ++bin) {
                const float value = magnitude.values[
                    static_cast<size_t>(frame * bins + bin)];
                energy += value * value *
                    filterbank[
                        static_cast<size_t>(mel * bins + bin)];
            }
            out.values[
                static_cast<size_t>(frame * kMels + mel)] =
                std::log(
                    std::max(
                        energy,
                        std::numeric_limits<float>::epsilon()));
        }
    }
    for (int64_t mel = 0; mel < kMels; ++mel) {
        float mean = 0.0F;
        for (int64_t frame = 0; frame < frames; ++frame) {
            mean += out.values[
                static_cast<size_t>(frame * kMels + mel)];
        }
        mean /= static_cast<float>(frames);
        for (int64_t frame = 0; frame < frames; ++frame) {
            out.values[
                static_cast<size_t>(frame * kMels + mel)] -= mean;
        }
    }
    return out;
}

}  // namespace engine::models::glm_tts
