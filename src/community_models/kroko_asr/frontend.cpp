#include "engine/community_models/kroko_asr/frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace engine::models::kroko_asr {
namespace {

constexpr int64_t kSampleRate = 16000;
constexpr int64_t kWindow = 400;
constexpr int64_t kHop = 160;
constexpr int64_t kNfft = 512;
constexpr int64_t kBins = kNfft / 2;
constexpr int64_t kMels = 80;
constexpr float kLowFreq = 20.0F;
constexpr float kHighFreq = 7600.0F;
constexpr float kPreemphasis = 0.97F;

int64_t reflect_index(int64_t index, int64_t size) {
    while (index < 0 || index >= size) {
        index = index < 0 ? -index - 1 : 2 * size - 1 - index;
    }
    return index;
}

const std::vector<float> & povey_window() {
    static const std::vector<float> value = [] {
        constexpr float kPi = 3.14159265358979323846F;
        std::vector<float> result(static_cast<size_t>(kWindow), 0.0F);
        for (int64_t index = 0; index < kWindow; ++index) {
            const float hann =
                0.5F - 0.5F * std::cos(
                    2.0F * kPi * static_cast<float>(index) /
                    static_cast<float>(kWindow - 1));
            result[static_cast<size_t>(index)] = std::pow(hann, 0.85F);
        }
        return result;
    }();
    return value;
}

const std::vector<float> & filterbank() {
    static const std::vector<float> value = [] {
        auto mel = [](float hz) {
            return 1127.0F * std::log(1.0F + hz / 700.0F);
        };
        const float low = mel(kLowFreq);
        const float high = mel(kHighFreq);
        const float delta = (high - low) / static_cast<float>(kMels + 1);
        std::vector<float> result(static_cast<size_t>(kMels * kBins), 0.0F);
        for (int64_t band = 0; band < kMels; ++band) {
            const float left = low + static_cast<float>(band) * delta;
            const float center = low + static_cast<float>(band + 1) * delta;
            const float right = low + static_cast<float>(band + 2) * delta;
            for (int64_t bin = 0; bin < kBins; ++bin) {
                const float point = mel(
                    static_cast<float>(bin * kSampleRate) /
                    static_cast<float>(kNfft));
                if (point > left && point < right) {
                    result[static_cast<size_t>(band * kBins + bin)] =
                        point <= center
                            ? (point - left) / (center - left)
                            : (right - point) / (right - center);
                }
            }
        }
        return result;
    }();
    return value;
}

}  // namespace

KrokoFbankFeatures compute_kroko_fbank(
    const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
        throw std::runtime_error("Kroko ASR requires non-empty audio");
    }
    const auto waveform =
        engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
            audio.samples,
            audio.sample_rate,
            audio.channels,
            static_cast<int>(kSampleRate));
    if (waveform.empty()) {
        throw std::runtime_error("Kroko ASR resampling produced empty audio");
    }

    // Kaldi snip_edges=false: round sample_count / frame_shift to nearest.
    const int64_t frames =
        (static_cast<int64_t>(waveform.size()) + kHop / 2) / kHop;
    std::vector<float> framed(
        static_cast<size_t>(frames * kNfft), 0.0F);
    const auto & window = povey_window();
    const int64_t samples = static_cast<int64_t>(waveform.size());
    for (int64_t frame = 0; frame < frames; ++frame) {
        const int64_t start = frame * kHop + kHop / 2 - kWindow / 2;
        float mean = 0.0F;
        for (int64_t index = 0; index < kWindow; ++index) {
            mean += waveform[
                static_cast<size_t>(reflect_index(start + index, samples))];
        }
        mean /= static_cast<float>(kWindow);
        float previous = 0.0F;
        for (int64_t index = 0; index < kWindow; ++index) {
            float current = waveform[
                static_cast<size_t>(reflect_index(start + index, samples))] - mean;
            const float emphasized =
                index == 0
                    ? current - kPreemphasis * current
                    : current - kPreemphasis * previous;
            previous = current;
            framed[static_cast<size_t>(frame * kNfft + index)] =
                emphasized * window[static_cast<size_t>(index)];
        }
    }

    std::vector<float> unit_window(static_cast<size_t>(kNfft), 1.0F);
    const engine::audio::STFTConfig stft{
        kNfft,
        kNfft,
        kNfft,
        false,
        engine::audio::STFTPadMode::Constant,
        engine::audio::STFTFamily::Default,
    };
    const auto magnitude = engine::audio::STFT().compute_magnitude(
        framed,
        unit_window,
        frames,
        kNfft,
        stft);
    if (magnitude.shape.size() != 3 ||
        magnitude.shape[0] != frames ||
        magnitude.shape[1] != kNfft / 2 + 1 ||
        magnitude.shape[2] != 1) {
        throw std::runtime_error("Kroko ASR STFT returned an unexpected shape");
    }

    KrokoFbankFeatures result;
    result.frames = frames;
    result.feature_dim = kMels;
    result.values.assign(static_cast<size_t>(frames * kMels), 0.0F);
    const auto & banks = filterbank();
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t band = 0; band < kMels; ++band) {
            float energy = 0.0F;
            for (int64_t bin = 0; bin < kBins; ++bin) {
                const float value = magnitude.values[
                    static_cast<size_t>(frame * (kNfft / 2 + 1) + bin)];
                energy += value * value *
                    banks[static_cast<size_t>(band * kBins + bin)];
            }
            result.values[static_cast<size_t>(frame * kMels + band)] =
                std::log(std::max(
                    energy,
                    std::numeric_limits<float>::epsilon()));
        }
    }
    return result;
}

}  // namespace engine::models::kroko_asr
