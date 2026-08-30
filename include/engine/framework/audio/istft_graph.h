#pragma once

#include "engine/framework/core/module.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::audio {

// Which log-magnitude/phase iSTFT implementation a call site should build.
enum class LogMagnitudePhaseISTFTPath {
    Host,
    Cuda,
};

// True when this build carries the CUDA iSTFT runtime (ENGINE_HAS_CUDA_ISTFT).
// Resolved out of line, in engine_core, because the macro is PRIVATE to that
// target: a model translation unit cannot test it with #ifdef and would always
// read false. Every caller must ask through this function.
bool cuda_log_magnitude_phase_istft_available() noexcept;

// Chooses the iSTFT implementation for a backend. This is the only place that
// decides, so the four call sites cannot drift apart again.
//
// Metal deliberately maps to Host: there is no Metal iSTFT kernel, and the
// ggml graph builders in this file express only the CUDA and host paths. This
// is a missing implementation, not a Metal defect — see the note in
// istft_graph.cpp. HIP maps to Host too, because istft_cuda_runtime.cu is
// compiled only under `ENGINE_ENABLE_CUDA AND NOT ENGINE_ENABLE_HIP`.
//
// Setting ENGINE_ISTFT_HOST=1 forces the host path on every backend, which
// keeps the pre-existing behaviour of any call site this selector newly moves
// onto the CUDA kernel.
LogMagnitudePhaseISTFTPath select_log_magnitude_phase_istft_path(core::BackendType backend) noexcept;

struct HostLogMagnitudePhaseISTFTConfig {
    int64_t frames = 0;
    int64_t n_fft = 0;
    int64_t hop_length = 0;
    int64_t out_dim = 0;
    size_t threads = 1;
};

struct CudaLogMagnitudePhaseISTFTConfig {
    int64_t frames = 0;
    int64_t n_fft = 0;
    int64_t hop_length = 0;
    int64_t out_dim = 0;
    int device = 0;
};

struct HostLogMagnitudePhaseISTFTTiming {
    double spectrum_ms = 0.0;
    double framed_clear_ms = 0.0;
    double fft_inverse_ms = 0.0;
    // Always 0: the overlap-add is a gather that writes every output element,
    // so there is no zero-fill pass left to time. Kept so the timing schema
    // and its consumers do not change.
    double fold_clear_ms = 0.0;
    double overlap_add_ms = 0.0;
    double normalize_ms = 0.0;
    double total_ms = 0.0;
};

struct CudaLogMagnitudePhaseISTFTTiming {
    double input_upload_ms = 0.0;
    double spectrum_kernel_ms = 0.0;
    double fft_inverse_ms = 0.0;
    double overlap_add_ms = 0.0;
    double normalize_ms = 0.0;
    double audio_read_ms = 0.0;
    double total_ms = 0.0;
};

struct HostLogMagnitudePhaseISTFTResult {
    std::vector<float> audio;
    HostLogMagnitudePhaseISTFTTiming timing;
};

struct CudaLogMagnitudePhaseISTFTResult {
    std::vector<float> audio;
    CudaLogMagnitudePhaseISTFTTiming timing;
};

class HostLogMagnitudePhaseISTFT {
public:
    explicit HostLogMagnitudePhaseISTFT(const HostLogMagnitudePhaseISTFTConfig & config);
    ~HostLogMagnitudePhaseISTFT();

    HostLogMagnitudePhaseISTFT(const HostLogMagnitudePhaseISTFT &) = delete;
    HostLogMagnitudePhaseISTFT & operator=(const HostLogMagnitudePhaseISTFT &) = delete;
    HostLogMagnitudePhaseISTFT(HostLogMagnitudePhaseISTFT &&) noexcept;
    HostLogMagnitudePhaseISTFT & operator=(HostLogMagnitudePhaseISTFT &&) noexcept;

    HostLogMagnitudePhaseISTFTResult compute(
        const std::vector<float> & log_magnitude_phase,
        const std::vector<float> & window);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class CudaLogMagnitudePhaseISTFT {
public:
    explicit CudaLogMagnitudePhaseISTFT(const CudaLogMagnitudePhaseISTFTConfig & config);
    ~CudaLogMagnitudePhaseISTFT();

    CudaLogMagnitudePhaseISTFT(const CudaLogMagnitudePhaseISTFT &) = delete;
    CudaLogMagnitudePhaseISTFT & operator=(const CudaLogMagnitudePhaseISTFT &) = delete;
    CudaLogMagnitudePhaseISTFT(CudaLogMagnitudePhaseISTFT &&) noexcept;
    CudaLogMagnitudePhaseISTFT & operator=(CudaLogMagnitudePhaseISTFT &&) noexcept;

    CudaLogMagnitudePhaseISTFTResult compute(
        const std::vector<float> & log_magnitude_phase,
        const std::vector<float> & window);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::audio
