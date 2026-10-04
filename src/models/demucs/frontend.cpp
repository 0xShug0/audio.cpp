#include "engine/models/demucs/frontend.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/fft.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace engine::models::demucs {
namespace {

int32_t reflect_index(int64_t index, int64_t size) {
    while (index < 0 || index >= size) {
        if (index < 0) {
            index = -index;
        } else {
            index = 2 * size - index - 2;
        }
    }
    return static_cast<int32_t>(index);
}

std::vector<int32_t> build_reflect_indices(int64_t samples, int64_t pad_left, int64_t pad_right) {
    if (samples <= 0) {
        throw std::runtime_error("HTDemucs pad1d requires positive length");
    }
    const int64_t max_pad = std::max(pad_left, pad_right);
    int64_t extra_left = 0;
    int64_t extra_right = 0;
    int64_t adj_left = pad_left;
    int64_t adj_right = pad_right;
    if (samples <= max_pad) {
        const int64_t extra = max_pad - samples + 1;
        extra_right = std::min<int64_t>(pad_right, extra);
        extra_left = extra - extra_right;
        adj_left -= extra_left;
        adj_right -= extra_right;
    }

    if (extra_left == 0 && extra_right == 0) {
        const int64_t out_len = samples + adj_left + adj_right;
        std::vector<int32_t> indices(static_cast<size_t>(out_len));
        for (int64_t i = 0; i < out_len; ++i) {
            indices[static_cast<size_t>(i)] = reflect_index(i - adj_left, samples);
        }
        return indices;
    }

    const int64_t mid_len = samples + extra_left + extra_right;
    const int64_t out_len = mid_len + adj_left + adj_right;
    std::vector<int32_t> indices(static_cast<size_t>(out_len));
    for (int64_t i = 0; i < out_len; ++i) {
        const int64_t mid_index = reflect_index(i - adj_left, mid_len);
        if (mid_index < extra_left || mid_index >= extra_left + samples) {
            indices[static_cast<size_t>(i)] = -1;
        } else {
            indices[static_cast<size_t>(i)] = static_cast<int32_t>(mid_index - extra_left);
        }
    }
    return indices;
}

void pad1d_reflect_fast(
    std::vector<float> & out,
    const std::vector<float> & signal,
    int64_t channels,
    int64_t samples,
    const std::vector<int32_t> & indices,
    size_t threads) {
    const int64_t out_len = static_cast<int64_t>(indices.size());
    out.resize(static_cast<size_t>(channels * out_len));
#ifdef _OPENMP
    #pragma omp parallel for num_threads(threads) if(channels >= 2)
#endif
    for (int64_t ch = 0; ch < channels; ++ch) {
        const float * src = signal.data() + static_cast<size_t>(ch * samples);
        float * dst = out.data() + static_cast<size_t>(ch * out_len);
        for (int64_t i = 0; i < out_len; ++i) {
            dst[i] = src[indices[static_cast<size_t>(i)]];
        }
    }
}

void compute_stft_complex_normalized(
    std::vector<float> & framed,
    std::vector<std::complex<float>> & spectrum,
    const std::vector<float> & signal,
    const std::vector<float> & window,
    int64_t batch,
    int64_t samples,
    int64_t frames,
    int64_t n_fft,
    const std::vector<int32_t> & frame_indices,
    size_t fft_threads) {
    const int64_t freq_bins = (n_fft / 2) + 1;
    framed.resize(static_cast<size_t>(batch * frames * n_fft));
#ifdef _OPENMP
    #pragma omp parallel for num_threads(fft_threads) if(batch * frames >= 8)
#endif
    for (int64_t b = 0; b < batch; ++b) {
        const float * src = signal.data() + static_cast<size_t>(b * samples);
        for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
            float * frame = framed.data() + static_cast<size_t>((b * frames + frame_index) * n_fft);
            const int32_t * mapping = frame_indices.data() + static_cast<size_t>(frame_index * n_fft);
            for (int64_t i = 0; i < n_fft; ++i) {
                frame[i] = src[mapping[i]] * window[static_cast<size_t>(i)];
            }
        }
    }

    engine::audio::TensorShape shape_in{
        static_cast<size_t>(batch),
        static_cast<size_t>(frames),
        static_cast<size_t>(n_fft),
    };
    engine::audio::TensorStrideBytes stride_in{
        static_cast<std::ptrdiff_t>(frames * n_fft * static_cast<int64_t>(sizeof(float))),
        static_cast<std::ptrdiff_t>(n_fft * static_cast<int64_t>(sizeof(float))),
        static_cast<std::ptrdiff_t>(sizeof(float)),
    };
    engine::audio::TensorStrideBytes stride_out{
        static_cast<std::ptrdiff_t>(frames * freq_bins * static_cast<int64_t>(sizeof(std::complex<float>))),
        static_cast<std::ptrdiff_t>(freq_bins * static_cast<int64_t>(sizeof(std::complex<float>))),
        static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)),
    };

    spectrum.resize(static_cast<size_t>(batch * frames * freq_bins));
    engine::audio::real_fft_forward(
        shape_in,
        stride_in,
        stride_out,
        2,
        framed.data(),
        spectrum.data(),
        1.0f / std::sqrt(static_cast<float>(n_fft)),
        fft_threads);
}

std::pair<float, float> normalize_in_place_with_stats(
    std::vector<float> & values,
    double sum,
    double sumsq,
    size_t threads) {
    const double count = static_cast<double>(values.size());
    const double mean = sum / count;
    const double denom = values.size() > 1 ? static_cast<double>(values.size() - 1) : 1.0;
    const double centered = std::max(0.0, sumsq - count * mean * mean);
    const float mean_f32 = static_cast<float>(mean);
    const float stddev = static_cast<float>(std::sqrt(centered / denom) + 1.0e-5);
#ifdef _OPENMP
    #pragma omp parallel for num_threads(threads) if(values.size() >= 1 << 16)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i) {
        values[static_cast<size_t>(i)] = (values[static_cast<size_t>(i)] - mean_f32) / stddev;
    }
    return {mean_f32, stddev};
}

std::pair<float, float> normalize_in_place(std::vector<float> & values, size_t threads) {
    double sum = 0.0;
    double sumsq = 0.0;
    for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i) {
        const double value = static_cast<double>(values[static_cast<size_t>(i)]);
        sum += value;
        sumsq += value * value;
    }
    return normalize_in_place_with_stats(values, sum, sumsq, threads);
}

std::pair<double, double> build_demucs_complex_input(
    std::vector<float> & out,
    const std::vector<std::complex<float>> & spectrum,
    int64_t channels,
    int64_t freq_bins,
    int64_t frames,
    int64_t full_frames) {
    out.resize(static_cast<size_t>(channels * 2 * freq_bins * frames));
    double sum = 0.0;
    double sumsq = 0.0;
    for (int64_t ch = 0; ch < channels; ++ch) {
        for (int64_t f = 0; f < freq_bins; ++f) {
            for (int64_t t = 0; t < frames; ++t) {
                const auto value = spectrum[static_cast<size_t>((ch * full_frames + (t + 2)) * (freq_bins + 1) + f)];
                const size_t dst_real = static_cast<size_t>((((ch * 2 + 0) * freq_bins) + f) * frames + t);
                const size_t dst_imag = static_cast<size_t>((((ch * 2 + 1) * freq_bins) + f) * frames + t);
                const float real = value.real();
                const float imag = value.imag();
                out[dst_real] = real;
                out[dst_imag] = imag;
                sum += static_cast<double>(real) + static_cast<double>(imag);
                sumsq += static_cast<double>(real) * static_cast<double>(real) +
                         static_cast<double>(imag) * static_cast<double>(imag);
            }
        }
    }
    return {sum, sumsq};
}

}  // namespace

HTDemucsFrontend::HTDemucsFrontend(const HTDemucsConfig & config, size_t fft_threads)
    : config_(config),
      fft_threads_(fft_threads) {
    engine::audio::STFTConfig stft_config;
    stft_config.n_fft = config_.n_fft;
    stft_config.hop_length = config_.hop_length;
    stft_config.win_length = config_.n_fft;
    stft_config.center = true;
    stft_config.pad_mode = engine::audio::STFTPadMode::Reflect;
    stft_config.family = engine::audio::STFTFamily::Kokoro;
    stft_window_ = engine::audio::get_cached_stft_window(stft_config);
    const int64_t le = static_cast<int64_t>(
        std::ceil(static_cast<double>(config_.segment_samples) / static_cast<double>(config_.hop_length)));
    const int64_t pad = (config_.hop_length / 2) * 3;
    const int64_t right_pad = pad + le * config_.hop_length - config_.segment_samples;
    if (config_.segment_samples <= std::max(pad, right_pad)) {
        throw std::runtime_error("HTDemucs frontend static reflect map requires segment longer than pad");
    }
    padded_samples_ = config_.segment_samples + pad + right_pad;
    stft_full_frames_ = le + 4;
    pad_indices_ = build_reflect_indices(config_.segment_samples, pad, right_pad);
    stft_frame_indices_.resize(static_cast<size_t>(stft_full_frames_ * config_.n_fft));
    const int64_t stft_pad = config_.n_fft / 2;
    for (int64_t frame_index = 0; frame_index < stft_full_frames_; ++frame_index) {
        const int64_t start = frame_index * config_.hop_length - stft_pad;
        int32_t * mapping = stft_frame_indices_.data() + static_cast<size_t>(frame_index * config_.n_fft);
        for (int64_t i = 0; i < config_.n_fft; ++i) {
            mapping[i] = reflect_index(start + i, padded_samples_);
        }
    }
    padded_.resize(static_cast<size_t>(config_.audio_channels * padded_samples_));
    stft_framed_.resize(static_cast<size_t>(config_.audio_channels * stft_full_frames_ * config_.n_fft));
    stft_spectrum_.resize(static_cast<size_t>(
        config_.audio_channels * stft_full_frames_ * ((config_.n_fft / 2) + 1)));
    freq_input_.resize(static_cast<size_t>(
        config_.input_freq_channels * config_.stft_freq_bins * config_.stft_frames));
}

void HTDemucsFrontend::prepare_chunk(std::vector<float> & chunk_planar) {
    if (chunk_planar.empty() || static_cast<int64_t>(chunk_planar.size()) % config_.audio_channels != 0) {
        throw std::runtime_error("HTDemucs chunk size mismatch");
    }
    input_samples_ = static_cast<int64_t>(chunk_planar.size()) / config_.audio_channels;
    if (input_samples_ > config_.segment_samples) {
        throw std::runtime_error("HTDemucs chunk length exceeds training segment length");
    }
    if (input_samples_ != config_.segment_samples) {
        throw std::runtime_error("HTDemucs frontend expects session chunks padded to segment length");
    }
    time_input_ = &chunk_planar;

    pad1d_reflect_fast(
        padded_,
        *time_input_,
        config_.audio_channels,
        config_.segment_samples,
        pad_indices_, fft_threads_);
    compute_stft_complex_normalized(
        stft_framed_,
        stft_spectrum_,
        padded_,
        stft_window_,
        config_.audio_channels,
        padded_samples_,
        stft_full_frames_,
        config_.n_fft,
        stft_frame_indices_,
        fft_threads_);
    const auto [freq_sum, freq_sumsq] = build_demucs_complex_input(
        freq_input_,
        stft_spectrum_,
        config_.audio_channels,
        config_.stft_freq_bins,
        config_.stft_frames,
        stft_full_frames_);
    std::tie(freq_mean_, freq_std_) = normalize_in_place_with_stats(freq_input_, freq_sum, freq_sumsq, fft_threads_);
    std::tie(time_mean_, time_std_) = normalize_in_place(*time_input_, fft_threads_);
}

const std::vector<float> & HTDemucsFrontend::freq_input() const noexcept { return freq_input_; }
const std::vector<float> & HTDemucsFrontend::time_input() const noexcept { return *time_input_; }
const std::vector<float> & HTDemucsFrontend::stft_window() const noexcept { return stft_window_; }
int64_t HTDemucsFrontend::input_samples() const noexcept { return input_samples_; }
float HTDemucsFrontend::freq_mean() const noexcept { return freq_mean_; }
float HTDemucsFrontend::freq_std() const noexcept { return freq_std_; }
float HTDemucsFrontend::time_mean() const noexcept { return time_mean_; }
float HTDemucsFrontend::time_std() const noexcept { return time_std_; }

}  // namespace engine::models::demucs
