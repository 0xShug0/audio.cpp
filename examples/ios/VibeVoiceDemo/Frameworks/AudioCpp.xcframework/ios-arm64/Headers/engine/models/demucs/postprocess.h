#pragma once

#include "engine/models/demucs/assets.h"
#include "engine/framework/audio/fft.h"

#include <cstddef>
#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::demucs {

class HTDemucsPostprocessor {
public:
    HTDemucsPostprocessor(
        const HTDemucsConfig & config,
        size_t fft_threads,
        const std::vector<float> & stft_window);

    void combine_chunk_into(
        const float * freq_output,
        const float * time_output,
        int64_t input_samples,
        float freq_mean,
        float freq_std,
        float time_mean,
        float time_std,
        float * output);

private:
    HTDemucsConfig config_;
    size_t fft_threads_ = 1;
    std::vector<float> stft_window_;
    std::shared_ptr<engine::audio::RealFFTPlan> fft_plan_;
    std::vector<float> window_sq_;
    std::vector<float> window_sums_;
    std::vector<float> window_inverse_;
    std::vector<std::complex<float>> spectrum_;
    std::vector<float> framed_;
    std::vector<float> accum_;
    float fft_scale_ = 1.0f;
};

}  // namespace engine::models::demucs
