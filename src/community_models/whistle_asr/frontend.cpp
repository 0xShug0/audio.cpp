#include "engine/community_models/whistle_asr/frontend.h"

#include "engine/framework/audio/fft.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::community_models::whistle_asr {
namespace {

constexpr size_t kRate = 16000;
constexpr size_t kMaximumSamples = 30 * kRate;
constexpr size_t kFft = 512;
constexpr size_t kWindow = 400;
constexpr size_t kHop = 160;
constexpr size_t kBins = kFft / 2 + 1;
constexpr size_t kMels = 80;

}  // namespace

WhistleFrontend::WhistleFrontend(std::vector<float> filterbank)
    : filterbank_(std::move(filterbank)) {
    if (filterbank_.size() != kBins * kMels ||
        !std::all_of(filterbank_.begin(), filterbank_.end(), [](float value) {
            return std::isfinite(value) && value >= 0.0f;
        })) {
        throw std::invalid_argument("Whistle requires a 257-by-80 nonnegative mel filterbank");
    }
}

MelFeatures WhistleFrontend::extract(const std::vector<float> & samples) const {
    if (samples.size() > kMaximumSamples ||
        !std::all_of(samples.begin(), samples.end(), [](float sample) { return std::isfinite(sample); })) {
        throw std::invalid_argument("Whistle requires finite 16 kHz mono PCM no longer than 30 seconds");
    }
    const size_t frames = samples.size() / kHop;
    if (frames < 4) {
        return {};
    }

    double gain = 1.0;
    if (const size_t blocks = samples.size() / 320; blocks > 0) {
        std::vector<double> rms(blocks);
        for (size_t block = 0; block < blocks; ++block) {
            double energy = 0.0;
            for (size_t index = 0; index < 320; ++index) {
                const double value = samples[block * 320 + index];
                energy += value * value;
            }
            rms[block] = std::sqrt(energy / 320.0);
        }
        const size_t percentile = static_cast<size_t>((blocks - 1) * 0.99);
        std::nth_element(rms.begin(), rms.begin() + percentile, rms.end());
        if (rms[percentile] > 1.0e-6) {
            gain = 0.1 / rms[percentile];
        }
    }

    std::array<double, kWindow> window{};
    const double pi = std::acos(-1.0);
    for (size_t index = 0; index < kWindow; ++index) {
        window[index] = 0.5 - 0.5 * std::cos(2.0 * pi * index / (kWindow - 1));
    }
    engine::audio::RealFFTPlan fft(kFft);
    std::array<float, kFft> input{};
    std::array<std::complex<float>, kBins> spectrum{};
    std::array<double, kBins> power{};
    std::vector<double> log_mel(frames * kMels);

    for (size_t frame = 0; frame < frames; ++frame) {
        input.fill(0.0f);
        for (size_t index = 0; index < kWindow; ++index) {
            const auto sample = static_cast<int64_t>(frame * kHop + index) - 200;
            if (sample >= 0 && static_cast<size_t>(sample) < samples.size()) {
                input[index] = static_cast<float>(samples[static_cast<size_t>(sample)] * gain * window[index]);
            }
        }
        fft.forward({kFft}, {sizeof(float)}, {sizeof(std::complex<float>)}, 0,
            input.data(), spectrum.data());
        for (size_t bin = 0; bin < kBins; ++bin) {
            power[bin] = std::norm(spectrum[bin]);
        }
        for (size_t mel = 0; mel < kMels; ++mel) {
            double value = 0.0;
            for (size_t bin = 0; bin < kBins; ++bin) {
                value += power[bin] * filterbank_[bin * kMels + mel];
            }
            log_mel[frame * kMels + mel] = std::log(value + 0x1p-24);
        }
    }

    MelFeatures result;
    result.frames = frames;
    result.values.resize(log_mel.size());
    for (size_t mel = 0; mel < kMels; ++mel) {
        double mean = 0.0;
        for (size_t frame = 0; frame < frames; ++frame) {
            mean += log_mel[frame * kMels + mel];
        }
        mean /= frames;
        double variance = 0.0;
        for (size_t frame = 0; frame < frames; ++frame) {
            const double difference = log_mel[frame * kMels + mel] - mean;
            variance += difference * difference;
        }
        const double deviation = std::sqrt(variance / (frames - 1)) + 1.0e-5;
        for (size_t frame = 0; frame < frames; ++frame) {
            result.values[frame * kMels + mel] =
                static_cast<float>((log_mel[frame * kMels + mel] - mean) / deviation);
        }
    }
    return result;
}

}  // namespace engine::community_models::whistle_asr
