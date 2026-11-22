#include "frontend.h"
#include "engine/framework/audio/resampling.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace engine::models::dettato::detail {
namespace {
audio::SparseMelFilterbank filter(const assets::TensorSource &source, const std::string &name) {
    const auto data = source.require_f32_tensor(name);
    auto bank = audio::MelFilterbank{}.prepare_sparse({data.values, {data.shape.dims[0], data.shape.dims[1]}});
    // Harmonics above Nyquist have exactly zero support. Represent those rows
    // as empty ranges; the generic sparse builder assumes nonempty mel rows.
    for (size_t row = 0; row < bank.starts.size(); ++row)
        if (bank.starts[row] == data.shape.dims[1] && bank.ends[row] == 0)
            bank.starts[row] = 0;
    return bank;
}

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

HarmonicFrontend::HarmonicFrontend(const assets::TensorSource &source, const std::string &prefix, size_t threads)
    : long_filter_(filter(source, prefix + ".fb")), short_filter_(filter(source, prefix + ".fb_x0")),
      long_window_(source.require_f32(prefix + ".win")), short_window_(source.require_f32(prefix + ".win_x0")),
      threads_(threads) {

}

void HarmonicFrontend::apply_pcen(HarmonicFeatures & features) const {
    // Match upstream's finite 200-tap FFT convolution. Replacing it with an
    // IIR/subtraction recurrence changes low-level values in padded windows;
    // PCEN and window standardisation amplify those changes.
    size_t n = 1;
    while (n < static_cast<size_t>(features.frames + 200)) n *= 2;
    const size_t frequencies = n / 2 + 1;
    if (!pcen_plan_ || pcen_plan_->fft_size() != n) {
        pcen_plan_ = audio::get_real_fft_plan(n);
        std::vector<float> kernel(n, 0);
        for (size_t k = 0; k < 200; ++k) kernel[k] = .025f * std::pow(.975f, static_cast<float>(k));
        pcen_kernel_.resize(frequencies);
        pcen_plan_->forward({n}, {sizeof(float)}, {sizeof(std::complex<float>)}, 0,
            kernel.data(), pcen_kernel_.data(), 1, threads_);
    }
    const size_t rows = static_cast<size_t>(features.batch * features.channels * features.bins);
    pcen_buffer_.assign(rows * n, 0);
    pcen_spectrum_.resize(rows * frequencies);
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_)
#endif
    for (int64_t row = 0; row < static_cast<int64_t>(rows); ++row) {
        const size_t ch = row / features.bins, bin = row % features.bins;
        for (int64_t t = 0; t < features.frames; ++t)
            pcen_buffer_[row * n + t] = std::exp(features.values[(ch * features.frames + t) * features.bins + bin]);
    }
    const audio::TensorStrideBytes real_strides{static_cast<std::ptrdiff_t>(n * sizeof(float)), sizeof(float)};
    const audio::TensorStrideBytes complex_strides{static_cast<std::ptrdiff_t>(frequencies * sizeof(std::complex<float>)), sizeof(std::complex<float>)};
    pcen_plan_->forward({rows, n}, real_strides, complex_strides, 1, pcen_buffer_.data(), pcen_spectrum_.data(), 1, threads_);
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_)
#endif
    for (int64_t row = 0; row < static_cast<int64_t>(rows); ++row) for (size_t k = 0; k < frequencies; ++k)
        pcen_spectrum_[row * frequencies + k] *= pcen_kernel_[k];
    pcen_plan_->inverse({rows, n}, complex_strides, real_strides, 1, pcen_spectrum_.data(), pcen_buffer_.data(), 1.f / n, threads_);
    // Each frequency sequence is independent. Harmonic channels that read the
    // same frequency share this work before channel-specific standardisation.
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_)
#endif
    for (int64_t row = 0; row < static_cast<int64_t>(rows); ++row) {
        const int64_t channel = row / features.bins, bin = row % features.bins;
        for (int64_t t = 0; t < features.frames; ++t) {
            const auto index = static_cast<size_t>((channel * features.frames + t) * features.bins + bin);
            const float energy = std::exp(features.values[index]);
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
    for (int64_t b = 0; b < batch; ++b)
        for (int64_t t = 0; t < out.frames; ++t)
            for (int64_t f = 0; f < out.extended_bins; ++f) {
                const auto index = static_cast<size_t>((b * raw.shape[1] + t) * out.extended_bins + f);
                out.power[static_cast<size_t>((b * out.extended_bins + f) * out.frames + t)] = raw.values[index];
                raw.values[index] = std::log(raw.values[index] + 1e-5f);
            }
    HarmonicFeatures compact;
    if (pcen) {
        compact.batch = batch;
        compact.channels = 1;
        compact.frames = out.frames;
        compact.bins = out.extended_bins + out.bins;
        compact.values.resize(static_cast<size_t>(batch * out.frames * compact.bins));
        for (int64_t b = 0; b < batch; ++b)
            for (int64_t t = 0; t < out.frames; ++t) {
                auto *row = compact.values.data() + (b * out.frames + t) * compact.bins;
                std::copy_n(raw.values.data() + (b * raw.shape[1] + t) * out.extended_bins, out.extended_bins, row);
                for (int64_t f = 0; f < out.bins; ++f)
                    row[out.extended_bins + f] = std::log(high.values[static_cast<size_t>((b * high.shape[1] + t) * out.bins + f)] + 1e-5f);
            }
        apply_pcen(compact);
    }
    for (int64_t b = 0; b < batch; ++b)
        for (int64_t ch = 0; ch < 9; ++ch) {
            const int64_t shift = ch < 8
                ? static_cast<int64_t>(std::nearbyint(36 * std::log2(harmonics[static_cast<size_t>(ch)]))) + 36 : 0;
            for (int64_t t = 0; t < out.frames; ++t) {
                auto *destination = out.values.data() + ((b * 9 + ch) * out.frames + t) * out.bins;
                if (pcen) {
                    const auto offset = ch < 8 ? shift : out.extended_bins;
                    std::copy_n(compact.values.data() + (b * out.frames + t) * compact.bins + offset, out.bins, destination);
                } else if (ch < 8) {
                    std::copy_n(raw.values.data() + (b * raw.shape[1] + t) * out.extended_bins + shift, out.bins, destination);
                } else {
                    for (int64_t f = 0; f < out.bins; ++f)
                        destination[f] = std::log(high.values[static_cast<size_t>((b * high.shape[1] + t) * out.bins + f)] + 1e-5f);
                }
            }
        }
    if (pcen) standardise(out, threads_);
    return out;
}

DrumFrontend::DrumFrontend(const assets::TensorSource &source, size_t threads) : threads_(threads) {
    for (const auto name : {"lo", "hi", "cq"}) {
        const auto prefix = std::string("front.") + name;
        filters_.push_back(filter(source, prefix + ".filterbank"));
        windows_.push_back(source.require_f32(prefix + ".window"));
    }

}

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
} // namespace engine::models::dettato::detail
