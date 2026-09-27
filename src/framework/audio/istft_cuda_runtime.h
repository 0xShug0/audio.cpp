#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::audio::detail {

struct CudaIstftRuntimeConfig {
    int64_t frames = 0;
    int64_t n_fft = 0;
    int64_t hop_length = 0;
    int64_t out_dim = 0;
    int device = 0;
    bool deterministic_overlap_add = false;
};

struct CudaIstftRuntimeTiming {
    double input_upload_ms = 0.0;
    double spectrum_kernel_ms = 0.0;
    double fft_inverse_ms = 0.0;
    double overlap_add_ms = 0.0;
    double normalize_ms = 0.0;
    double audio_read_ms = 0.0;
    double total_ms = 0.0;
};

struct CudaIstftRuntimeResult {
    std::vector<float> audio;
    CudaIstftRuntimeTiming timing;
};

class CudaIstftRuntime {
public:
    explicit CudaIstftRuntime(const CudaIstftRuntimeConfig & config);
    ~CudaIstftRuntime();

    CudaIstftRuntime(const CudaIstftRuntime &) = delete;
    CudaIstftRuntime & operator=(const CudaIstftRuntime &) = delete;
    CudaIstftRuntime(CudaIstftRuntime &&) noexcept;
    CudaIstftRuntime & operator=(CudaIstftRuntime &&) noexcept;

    CudaIstftRuntimeResult compute(
        const std::vector<float> & log_magnitude_phase,
        const std::vector<float> & window);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::audio::detail
