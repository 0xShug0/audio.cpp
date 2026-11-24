#include "engine/community_models/dettato/frontend.h"
#include "engine/framework/audio/resampling.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace engine::community_models::dettato::detail {

namespace {
audio::AudioTensor spectrum(const std::vector<float> &samples, int64_t batch, const std::vector<float> &window,
                            int64_t hop, const audio::SparseMelFilterbank &bank, size_t threads) {
    audio::STFTConfig config;
    config.n_fft = config.win_length = static_cast<int64_t>(window.size());
    config.hop_length = hop;
    config.center = false;
    config.pad_mode = audio::STFTPadMode::Constant;
    const int64_t length = static_cast<int64_t>(samples.size()) / batch;
    const int64_t pad = config.n_fft / 2;
    const int64_t frames = 1 + (length + 2 * pad - config.n_fft) / hop;
    constexpr int64_t block_frames = 32;
    const int64_t blocks = (frames + block_frames - 1) / block_frames;
    const int64_t bands = bank.dense.shape[0];
    audio::AudioTensor output;
    // Model frontends consume time-major rows. Keep filtered blocks in that
    // layout rather than writing a full frequency-major recording first.
    output.shape = {batch, frames, bands};
    output.values.resize(static_cast<size_t>(batch * bands * frames));
    // Filter each FFT block before moving on. The large unfiltered magnitude
    // and complex arrays never need to cover the complete recording.
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
    for (int64_t job = 0; job < batch * blocks; ++job) {
        const int64_t b = job / blocks, first = (job % blocks) * block_frames;
        const int64_t count = std::min(block_frames, frames - first);
        const int64_t size = config.n_fft + (count - 1) * hop, origin = first * hop - pad;
        std::vector<float> segment(static_cast<size_t>(size), 0);
        const int64_t lo = std::max<int64_t>(origin, 0), hi = std::min(origin + size, length);
        if (hi > lo)
            std::copy_n(samples.begin() + b * length + lo, hi - lo, segment.begin() + lo - origin);
        const auto magnitude = audio::STFT{}.compute_magnitude(segment, window, 1, size, config, 1);
        const auto filtered = audio::MelFilterbank{}.compute_custom_sparse_from_magnitude(
            magnitude.values, 1, config.n_fft / 2 + 1, count, count, bank);
        for (int64_t t = 0; t < count; ++t)
            for (int64_t band = 0; band < bands; ++band)
                output.values[static_cast<size_t>(((b * frames) + first + t) * bands + band)] =
                    filtered.values[static_cast<size_t>(band * count + t)];
    }
    return output;
}

void standardise(HarmonicFeatures &out, size_t threads) {
    (void)threads; // Used by the parallel loop when OpenMP is enabled.
    const int64_t channels = out.batch * out.channels;
    const int64_t frames = out.frames, bins = out.bins;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
    for (int64_t ch = 0; ch < channels; ++ch) {
        auto first = out.values.begin() + ch * frames * bins;
        const float cutoff = *std::max_element(first, first + frames * bins) - 12;
        double sum = 0, variance = 0;
        int64_t count = 0;
        for (int64_t i = 0; i < frames * bins; ++i)
            if (first[i] > cutoff) {
                sum += first[i];
                ++count;
            }
        const float mean = static_cast<float>(sum / std::max<int64_t>(count, 1));
        for (int64_t i = 0; i < frames * bins; ++i)
            if (first[i] > cutoff)
                variance += static_cast<double>(first[i] - mean) * (first[i] - mean);
        const float sd = std::max(0.5f, static_cast<float>(std::sqrt(variance / std::max<int64_t>(count, 1))));
        for (int64_t i = 0; i < frames * bins; ++i)
            first[i] = (first[i] - mean) / sd;
    }
}
} // namespace

HarmonicFrontend::HarmonicFrontend(HarmonicFrontendWeights weights, size_t threads)
    : long_filter_(std::move(weights.long_filter)), short_filter_(std::move(weights.short_filter)),
      long_window_(std::move(weights.long_window)), short_window_(std::move(weights.short_window)), threads_(threads) {}

void HarmonicFrontend::apply_pcen(HarmonicFeatures &features) const {
    // Match upstream's finite 200-tap FFT convolution. Replacing it with an
    // IIR/subtraction recurrence changes low-level values in padded windows;
    // PCEN and window standardisation amplify those changes.
    size_t n = 1;
    while (n < static_cast<size_t>(features.frames + 200))
        n *= 2;
    const size_t frequencies = n / 2 + 1;
    if (!pcen_plan_ || pcen_plan_->fft_size() != n) {
        pcen_plan_ = audio::get_real_fft_plan(n);
        std::vector<float> kernel(n, 0);
        for (size_t k = 0; k < 200; ++k)
            kernel[k] = .025f * std::pow(.975f, static_cast<float>(k));
        pcen_kernel_.resize(frequencies);
        pcen_plan_->forward({n}, {sizeof(float)}, {sizeof(std::complex<float>)}, 0, kernel.data(), pcen_kernel_.data(),
                            1, threads_);
    }
    const size_t rows = static_cast<size_t>(features.batch * features.channels * features.bins);
    pcen_buffer_.assign(rows * n, 0);
    pcen_spectrum_.resize(rows * frequencies);
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_)
#endif
    for (int64_t row = 0; row < static_cast<int64_t>(rows); ++row) {
        const size_t ch = row / features.bins, bin = row % features.bins;
        for (int64_t t = 0; t < features.frames; ++t) {
            const auto index = (ch * features.frames + t) * features.bins + bin;
            const float energy = std::exp(features.values[index]);
            features.values[index] = energy;
            pcen_buffer_[row * n + t] = energy;
        }
    }
    const audio::TensorStrideBytes real_strides{static_cast<std::ptrdiff_t>(n * sizeof(float)), sizeof(float)};
    const audio::TensorStrideBytes complex_strides{
        static_cast<std::ptrdiff_t>(frequencies * sizeof(std::complex<float>)), sizeof(std::complex<float>)};
    pcen_plan_->forward({rows, n}, real_strides, complex_strides, 1, pcen_buffer_.data(), pcen_spectrum_.data(), 1,
                        threads_);
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_)
#endif
    for (int64_t row = 0; row < static_cast<int64_t>(rows); ++row)
        for (size_t k = 0; k < frequencies; ++k)
            pcen_spectrum_[row * frequencies + k] *= pcen_kernel_[k];
    pcen_plan_->inverse({rows, n}, complex_strides, real_strides, 1, pcen_spectrum_.data(), pcen_buffer_.data(),
                        1.f / n, threads_);
    // Each frequency sequence is independent. Harmonic channels that read the
    // same frequency share this work before channel-specific standardisation.
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_)
#endif
    for (int64_t row = 0; row < static_cast<int64_t>(rows); ++row) {
        const int64_t channel = row / features.bins, bin = row % features.bins;
        for (int64_t t = 0; t < features.frames; ++t) {
            const auto index = static_cast<size_t>((channel * features.frames + t) * features.bins + bin);
            const float energy = features.values[index];
            const float level = pcen_buffer_[static_cast<size_t>(row) * n + t];
            const float compressed =
                std::sqrt(energy / std::pow(1e-6f + std::max(level, 0.0f), 0.98f) + 2.0f) - std::sqrt(2.0f);
            features.values[index] = std::log(compressed + 1e-4f);
        }
    }
}

HarmonicFeatures HarmonicFrontend::extract(const std::vector<float> &samples, int64_t batch, bool pcen) const {
    if (batch <= 0 || samples.empty() || samples.size() % static_cast<size_t>(batch))
        throw std::runtime_error("Dettato frontend expects equally sized, nonempty audio windows");
    auto raw = spectrum(samples, batch, long_window_, 160, long_filter_, threads_);
    const auto high = spectrum(samples, batch, short_window_, 160, short_filter_, threads_);
    HarmonicFeatures out;
    out.batch = batch;
    out.channels = 9;
    out.frames = static_cast<int64_t>(samples.size()) / batch / 160;
    out.extended_bins = long_filter_.dense.shape[0];
    if (out.frames <= 0)
        throw std::runtime_error("Dettato frontend needs at least 10 ms of audio");
    out.values.resize(static_cast<size_t>(batch * 9 * out.frames * out.bins));
    out.power.resize(static_cast<size_t>(batch * out.extended_bins * out.frames));
    const std::array<double, 8> harmonics{0.5, 1, 2, 3, 4, 5, 6, 7};
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_)
#endif
    for (int64_t row = 0; row < batch * out.frames; ++row) {
        const int64_t b = row / out.frames, t = row % out.frames;
        for (int64_t f = 0; f < out.extended_bins; ++f) {
            const auto index = static_cast<size_t>((b * raw.shape[1] + t) * out.extended_bins + f);
            out.power[static_cast<size_t>((b * out.extended_bins + f) * out.frames + t)] = raw.values[index];
            raw.values[index] = std::log(raw.values[index] + 1e-5f);
        }
    }
    HarmonicFeatures compact;
    if (pcen) {
        compact.batch = batch;
        compact.channels = 1;
        compact.frames = out.frames;
        compact.bins = out.extended_bins + out.bins;
        compact.values.resize(static_cast<size_t>(batch * out.frames * compact.bins));
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_)
#endif
        for (int64_t row_index = 0; row_index < batch * out.frames; ++row_index) {
            const int64_t b = row_index / out.frames, t = row_index % out.frames;
            auto *row = compact.values.data() + (b * out.frames + t) * compact.bins;
            std::copy_n(raw.values.data() + (b * raw.shape[1] + t) * out.extended_bins, out.extended_bins, row);
            for (int64_t f = 0; f < out.bins; ++f)
                row[out.extended_bins + f] =
                    std::log(high.values[static_cast<size_t>((b * high.shape[1] + t) * out.bins + f)] + 1e-5f);
        }
        apply_pcen(compact);
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_)
#endif
    for (int64_t row = 0; row < batch * 9; ++row) {
        const int64_t b = row / 9, ch = row % 9;
        const int64_t shift =
            ch < 8 ? static_cast<int64_t>(std::nearbyint(36 * std::log2(harmonics[static_cast<size_t>(ch)]))) + 36 : 0;
        for (int64_t t = 0; t < out.frames; ++t) {
            auto *destination = out.values.data() + ((b * 9 + ch) * out.frames + t) * out.bins;
            if (pcen) {
                const auto offset = ch < 8 ? shift : out.extended_bins;
                std::copy_n(compact.values.data() + (b * out.frames + t) * compact.bins + offset, out.bins,
                            destination);
            } else if (ch < 8) {
                std::copy_n(raw.values.data() + (b * raw.shape[1] + t) * out.extended_bins + shift, out.bins,
                            destination);
            } else {
                for (int64_t f = 0; f < out.bins; ++f)
                    destination[f] =
                        std::log(high.values[static_cast<size_t>((b * high.shape[1] + t) * out.bins + f)] + 1e-5f);
            }
        }
    }
    if (pcen)
        standardise(out, threads_);
    return out;
}

DrumFrontend::DrumFrontend(DrumFrontendWeights weights, size_t threads)
    : filters_(std::move(weights.filters)), windows_(std::move(weights.windows)), threads_(threads) {}

std::vector<float> DrumFrontend::extract(const std::vector<float> &mono44k) const {
    double energy = 0;
    for (const auto value : mono44k)
        energy += static_cast<double>(value) * value;
    const double gain = std::pow(10.0, -18.0 / 20) / (std::sqrt(energy / mono44k.size()) + 1e-9);
    auto high_audio = mono44k;
    for (auto &value : high_audio)
        value = static_cast<float>(value * gain);
    audio::SoxrResampleOptions options;
    options.warning_context = "Dettato drum frontend";
    options.fallback_description = "linear resampling";
    const auto low_audio = audio::resample_mono_soxr_or_linear(high_audio, 44100, 16000, options);
    std::array<audio::AudioTensor, 3> features{
        spectrum(low_audio, 1, windows_[0], 160, filters_[0], threads_),
        spectrum(high_audio, 1, windows_[1], 441, filters_[1], threads_),
        spectrum(low_audio, 1, windows_[2], 160, filters_[2], threads_),
    };
    const int64_t frames = std::min({static_cast<int64_t>(low_audio.size() / 160), features[0].shape[1],
                                     features[1].shape[1], features[2].shape[1]});
    std::vector<float> output(static_cast<size_t>(frames * 280));
    int64_t base = 0;
    for (const auto &value : features) {
        for (int64_t t = 0; t < frames; ++t)
            for (int64_t f = 0; f < value.shape[2]; ++f) {
                const float log = std::log(value.values[static_cast<size_t>(t * value.shape[2] + f)] + 1e-5f);
                output[static_cast<size_t>(t * 280 + base + f)] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(log));
            }
        base += value.shape[2];
    }
    return output;
}

namespace {
constexpr double pi = 3.14159265358979323846;
}
TuningEstimator::TuningEstimator(size_t threads) : threads_(threads), window_(8192) {
    for (size_t i = 0; i < window_.size(); ++i)
        window_[i] = static_cast<float>(.5 - .5 * std::cos(2 * pi * i / 8191));
    const double lo = 1200 * std::log2(65. / 440.), hi = 1200 * std::log2(1000. / 440.);
    for (size_t i = 0; lo + i * 5 < hi; ++i)
        cents_.push_back(lo + i * 5);
}
Tuning TuningEstimator::estimate(const std::vector<float> &audio16) {
    if (audio16.size() < 8192)
        return {};
    const size_t frames = 1 + (audio16.size() - 8192) / 1024;
    std::complex<double> sum{};
    double weight_sum = 0;
    for (size_t first = 0; first < frames; first += 512) {
        const size_t count = std::min<size_t>(512, frames - first);
        input_.resize(count * 8192);
        spectrum_.resize(count * 4097);
        magnitude_.resize(count * 4097);
        salience_.assign(cents_.size() * count, 0);
        for (size_t t = 0; t < count; ++t)
            for (size_t i = 0; i < 8192; ++i)
                input_[t * 8192 + i] = audio16[(first + t) * 1024 + i] * window_[i];
        fft_.forward({count, 8192}, {8192 * sizeof(float), sizeof(float)},
                     {4097 * sizeof(std::complex<float>), sizeof(std::complex<float>)}, 1, input_.data(),
                     spectrum_.data(), 1, threads_);
        for (size_t i = 0; i < magnitude_.size(); ++i)
            magnitude_[i] = std::sqrt(std::abs(spectrum_[i]));
        for (int h = 1; h <= 6; ++h) {
            const float harmonic_weight = static_cast<float>(std::pow(.85, h - 1));
            for (size_t j = 0; j < cents_.size(); ++j) {
                const double bin = h * 440 * std::pow(2., cents_[j] / 1200) * 8192 / 16000;
                if (bin >= 4095)
                    continue;
                const size_t i = static_cast<size_t>(bin);
                const float w = static_cast<float>(bin - i);
                for (size_t t = 0; t < count; ++t)
                    salience_[j * count + t] +=
                        harmonic_weight * ((1 - w) * magnitude_[t * 4097 + i] + w * magnitude_[t * 4097 + i + 1]);
            }
        }
        std::vector<float> maxima(count, 0);
        for (size_t j = 0; j < cents_.size(); ++j)
            for (size_t t = 0; t < count; ++t)
                maxima[t] = std::max(maxima[t], salience_[j * count + t]);
        for (size_t j = 1; j + 1 < cents_.size(); ++j) {
            for (size_t t = 0; t < count; ++t) {
                const float a = salience_[(j - 1) * count + t], b = salience_[j * count + t],
                            c = salience_[(j + 1) * count + t];
                if (!(b > a && b >= c && b > .5f * maxima[t] && maxima[t] > 0))
                    continue;
                const float d = std::clamp(.5f * (a - c) / (a - 2 * b + c - 1e-12f), -1.f, 1.f);
                const double angle = 2 * pi * (cents_[j] + d * 5) / 100;
                sum += static_cast<double>(b) * std::complex<double>(std::cos(angle), std::sin(angle));
                weight_sum += b;
            }
        }
    }
    if (weight_sum <= 0)
        return {};
    sum /= weight_sum;
    Tuning result;
    result.cents = std::arg(sum) * 100 / (2 * pi);
    result.strength = std::abs(sum);
    if (std::abs(result.cents) >= 10 && result.strength >= .12)
        result.ratio = std::pow(2., -result.cents / 1200);
    return result;
}
} // namespace engine::community_models::dettato::detail
