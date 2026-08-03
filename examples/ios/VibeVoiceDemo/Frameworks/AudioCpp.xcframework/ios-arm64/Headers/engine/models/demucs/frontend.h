#pragma once

#include "engine/models/demucs/assets.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine::models::demucs {

class HTDemucsFrontend {
public:
    HTDemucsFrontend(const HTDemucsConfig & config, size_t fft_threads);

    void prepare_chunk(std::vector<float> & chunk_planar);

    const std::vector<float> & freq_input() const noexcept;
    const std::vector<float> & time_input() const noexcept;
    const std::vector<float> & stft_window() const noexcept;

    int64_t input_samples() const noexcept;
    float freq_mean() const noexcept;
    float freq_std() const noexcept;
    float time_mean() const noexcept;
    float time_std() const noexcept;

private:
    HTDemucsConfig config_;
    size_t fft_threads_ = 1;
    int64_t input_samples_ = 0;
    int64_t padded_samples_ = 0;
    int64_t stft_full_frames_ = 0;
    std::vector<float> stft_window_;
    std::vector<float> padded_;
    std::vector<int32_t> pad_indices_;
    std::vector<float> stft_framed_;
    std::vector<std::complex<float>> stft_spectrum_;
    std::vector<int32_t> stft_frame_indices_;
    std::vector<float> freq_input_;
    std::vector<float> * time_input_ = nullptr;
    float freq_mean_ = 0.0f;
    float freq_std_ = 1.0f;
    float time_mean_ = 0.0f;
    float time_std_ = 1.0f;
};

}  // namespace engine::models::demucs
