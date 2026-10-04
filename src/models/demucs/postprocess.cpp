#include "engine/models/demucs/postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace engine::models::demucs {
HTDemucsPostprocessor::HTDemucsPostprocessor(
    const HTDemucsConfig & config,
    size_t fft_threads,
    const std::vector<float> & stft_window)
    : config_(config),
      fft_threads_(fft_threads),
      stft_window_(stft_window),
      fft_plan_(engine::audio::get_real_fft_plan(static_cast<size_t>(config.n_fft))),
      fft_scale_(std::sqrt(static_cast<float>(config.n_fft))) {
    window_sq_.resize(stft_window_.size());
    for (size_t i = 0; i < stft_window_.size(); ++i) {
        const float w = stft_window_[i];
        window_sq_[i] = w * w;
    }
    const int64_t le = static_cast<int64_t>(
        std::ceil(static_cast<double>(config_.segment_samples) / static_cast<double>(config_.hop_length)));
    const int64_t pad = (config_.hop_length / 2) * 3;
    const int64_t istft_length = config_.hop_length * le + 2 * pad;
    const int64_t full_frames = config_.stft_frames + 4;
    const int64_t padded_samples = istft_length + config_.n_fft;
    const int64_t istft_pad = config_.n_fft / 2;
    window_sums_.assign(static_cast<size_t>(padded_samples), 0.0f);
    for (int64_t frame_index = 0; frame_index < full_frames; ++frame_index) {
        const int64_t start = frame_index * config_.hop_length;
        float * window_row = window_sums_.data() + static_cast<size_t>(start);
        for (int64_t n = 0; n < config_.n_fft; ++n) {
            window_row[n] += window_sq_[static_cast<size_t>(n)];
        }
    }
    window_inverse_.resize(static_cast<size_t>(config_.segment_samples));
    const float * trimmed_window = window_sums_.data() + static_cast<size_t>(istft_pad + pad);
    for (int64_t i = 0; i < config_.segment_samples; ++i) {
        const float denom = trimmed_window[i] > 1e-8f ? trimmed_window[i] : 1.0f;
        window_inverse_[static_cast<size_t>(i)] = 1.0f / denom;
    }
    const int64_t batch = static_cast<int64_t>(config_.sources.size() * config_.audio_channels);
    const int64_t full_freq_bins = config_.stft_freq_bins + 1;
    spectrum_.resize(static_cast<size_t>(batch * full_frames * full_freq_bins));
    framed_.resize(static_cast<size_t>(batch * full_frames * config_.n_fft));
    accum_.resize(static_cast<size_t>(batch * padded_samples));
}

void HTDemucsPostprocessor::combine_chunk_into(
    const float * freq_output,
    const float * time_output,
    int64_t input_samples,
    float freq_mean,
    float freq_std,
    float time_mean,
    float time_std,
    float * output) {
    if (input_samples <= 0 || input_samples > config_.segment_samples) {
        throw std::runtime_error("HTDemucs postprocessor input sample count is invalid");
    }
    if (freq_output == nullptr || time_output == nullptr || output == nullptr) {
        throw std::runtime_error("HTDemucs postprocessor received null buffers");
    }

    const int64_t model_samples = config_.segment_samples;
    const int64_t le = static_cast<int64_t>(std::ceil(static_cast<double>(model_samples) / static_cast<double>(config_.hop_length)));
    const int64_t stft_frames = config_.stft_frames;
    const int64_t pad = (config_.hop_length / 2) * 3;
    const int64_t istft_length = config_.hop_length * le + 2 * pad;
    const int64_t full_freq_bins = config_.stft_freq_bins + 1;
    const int64_t full_frames = stft_frames + 4;
    const int64_t batch = static_cast<int64_t>(config_.sources.size() * config_.audio_channels);
    float * spectrum_data = reinterpret_cast<float *>(spectrum_.data());
    const float freq_scale = freq_std * fft_scale_;
    const float freq_bias = freq_mean * fft_scale_;
    const int64_t source_count = static_cast<int64_t>(config_.sources.size());

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) num_threads(fft_threads_) if(static_cast<int64_t>(config_.sources.size()) * config_.audio_channels >= 4)
#endif
    for (int64_t source = 0; source < source_count; ++source) {
        for (int ch = 0; ch < config_.audio_channels; ++ch) {
            const int64_t batch_index = static_cast<int64_t>(source * config_.audio_channels + ch);
            float * batch_dst = spectrum_data +
                static_cast<size_t>(batch_index * full_frames * full_freq_bins * 2);
            const int64_t real_channel = static_cast<int64_t>(source * config_.audio_channels * 2 + ch * 2);
            const int64_t imag_channel = real_channel + 1;
            for (int64_t f = 0; f < config_.stft_freq_bins; ++f) {
                for (int64_t t = 0; t < stft_frames; ++t) {
                    const size_t src_real = static_cast<size_t>(((real_channel * config_.stft_freq_bins + f) * stft_frames + t));
                    const size_t src_imag = static_cast<size_t>(((imag_channel * config_.stft_freq_bins + f) * stft_frames + t));
                    float * dst = batch_dst + static_cast<size_t>((((t + 2) * full_freq_bins) + f) * 2);
                    dst[0] = freq_output[src_real] * freq_scale + freq_bias;
                    dst[1] = freq_output[src_imag] * freq_scale + freq_bias;
                }
            }
        }
    }

    const int64_t padded_samples = istft_length + config_.n_fft;
    std::fill(accum_.begin(), accum_.end(), 0.0f);

    const engine::audio::TensorShape output_shape{
        static_cast<size_t>(batch),
        static_cast<size_t>(full_frames),
        static_cast<size_t>(config_.n_fft),
    };
    const engine::audio::TensorStrideBytes input_strides{
        static_cast<std::ptrdiff_t>(full_frames * full_freq_bins * static_cast<int64_t>(sizeof(std::complex<float>))),
        static_cast<std::ptrdiff_t>(full_freq_bins * static_cast<int64_t>(sizeof(std::complex<float>))),
        static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)),
    };
    const engine::audio::TensorStrideBytes output_strides{
        static_cast<std::ptrdiff_t>(full_frames * config_.n_fft * static_cast<int64_t>(sizeof(float))),
        static_cast<std::ptrdiff_t>(config_.n_fft * static_cast<int64_t>(sizeof(float))),
        static_cast<std::ptrdiff_t>(sizeof(float)),
    };
    fft_plan_->inverse(
        output_shape,
        input_strides,
        output_strides,
        2,
        spectrum_.data(),
        framed_.data(),
        1.0f / static_cast<float>(config_.n_fft),
        fft_threads_);
#ifdef _OPENMP
    #pragma omp parallel for num_threads(fft_threads_) if(batch >= 4)
#endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t frame_index = 0; frame_index < full_frames; ++frame_index) {
            const int64_t start = frame_index * config_.hop_length;
            const float * frame = framed_.data() + static_cast<size_t>((b * full_frames + frame_index) * config_.n_fft);
            float * accum_row = accum_.data() + static_cast<size_t>(b * padded_samples + start);
            for (int64_t n = 0; n < config_.n_fft; ++n) {
                accum_row[n] += frame[n] * stft_window_[static_cast<size_t>(n)];
            }
        }
    }

    const int64_t istft_pad = config_.n_fft / 2;
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) num_threads(fft_threads_) if(static_cast<int64_t>(config_.sources.size()) * config_.audio_channels >= 4)
#endif
    for (int64_t source = 0; source < source_count; ++source) {
        for (int ch = 0; ch < config_.audio_channels; ++ch) {
            const int64_t batch_index = static_cast<int64_t>(source * config_.audio_channels + ch);
            const float * accum_row =
                accum_.data() + static_cast<size_t>(batch_index * padded_samples + istft_pad + pad);
            const float * time_src = time_output + static_cast<size_t>((source * config_.audio_channels + ch) * model_samples);
            float * dst = output + static_cast<size_t>((source * config_.audio_channels + ch) * input_samples);
            for (int64_t i = 0; i < input_samples; ++i) {
                dst[i] = accum_row[i] * window_inverse_[static_cast<size_t>(i)] + (time_src[i] * time_std + time_mean);
            }
        }
    }
}
}  // namespace engine::models::demucs
