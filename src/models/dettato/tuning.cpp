#include "tuning.h"
#include <algorithm>
#include <cmath>

namespace engine::models::dettato::detail {
namespace { constexpr double pi = 3.14159265358979323846; }
TuningEstimator::TuningEstimator(size_t threads) : threads_(threads), window_(8192) {
    for (size_t i = 0; i < window_.size(); ++i)
        window_[i] = static_cast<float>(.5 - .5 * std::cos(2 * pi * i / 8191));
    const double lo = 1200 * std::log2(65. / 440.), hi = 1200 * std::log2(1000. / 440.);
    for (size_t i = 0; lo + i * 5 < hi; ++i) cents_.push_back(lo + i * 5);
}
Tuning TuningEstimator::estimate(const std::vector<float> & audio16) {
    if (audio16.size() < 8192) return {};
    const size_t frames = 1 + (audio16.size() - 8192) / 1024;
    std::complex<double> sum{};
    double weight_sum = 0;
    for (size_t first = 0; first < frames; first += 512) {
        const size_t count = std::min<size_t>(512, frames - first);
        input_.resize(count * 8192); spectrum_.resize(count * 4097);
        magnitude_.resize(count * 4097); salience_.assign(cents_.size() * count, 0);
        for (size_t t = 0; t < count; ++t)
            for (size_t i = 0; i < 8192; ++i)
                input_[t * 8192 + i] = audio16[(first + t) * 1024 + i] * window_[i];
        fft_.forward({count, 8192}, {8192 * sizeof(float), sizeof(float)},
            {4097 * sizeof(std::complex<float>), sizeof(std::complex<float>)}, 1,
            input_.data(), spectrum_.data(), 1, threads_);
        for (size_t i = 0; i < magnitude_.size(); ++i) magnitude_[i] = std::sqrt(std::abs(spectrum_[i]));
        for (int h = 1; h <= 6; ++h) {
            const float harmonic_weight = static_cast<float>(std::pow(.85, h - 1));
            for (size_t j = 0; j < cents_.size(); ++j) {
                const double bin = h * 440 * std::pow(2., cents_[j] / 1200) * 8192 / 16000;
                if (bin >= 4095) continue;
                const size_t i = static_cast<size_t>(bin);
                const float w = static_cast<float>(bin - i);
                for (size_t t = 0; t < count; ++t)
                    salience_[j * count + t] += harmonic_weight * ((1 - w) * magnitude_[t * 4097 + i] + w * magnitude_[t * 4097 + i + 1]);
            }
        }
        std::vector<float> maxima(count, 0);
        for (size_t j = 0; j < cents_.size(); ++j)
            for (size_t t = 0; t < count; ++t) maxima[t] = std::max(maxima[t], salience_[j * count + t]);
        for (size_t j = 1; j + 1 < cents_.size(); ++j) {
            for (size_t t = 0; t < count; ++t) {
                const float a = salience_[(j - 1) * count + t], b = salience_[j * count + t], c = salience_[(j + 1) * count + t];
                if (!(b > a && b >= c && b > .5f * maxima[t] && maxima[t] > 0)) continue;
                const float d = std::clamp(.5f * (a - c) / (a - 2 * b + c - 1e-12f), -1.f, 1.f);
                const double angle = 2 * pi * (cents_[j] + d * 5) / 100;
                sum += static_cast<double>(b) * std::complex<double>(std::cos(angle), std::sin(angle));
                weight_sum += b;
            }
        }
    }
    if (weight_sum <= 0) return {};
    sum /= weight_sum;
    Tuning result;
    result.cents = std::arg(sum) * 100 / (2 * pi);
    result.strength = std::abs(sum);
    if (std::abs(result.cents) >= 10 && result.strength >= .12) result.ratio = std::pow(2., -result.cents / 1200);
    return result;
}
} // namespace engine::models::dettato::detail
