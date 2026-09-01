#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::audio {

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

    // --- 增量 overlap-add iSTFT（用于流式）---
    // 按块喂入 log-magnitude+phase 帧，内部累积 overlap-add，
    // 返回"已能被窗口包络完整覆盖"的音频样本（块间平滑衔接）。
    // 首次 append 自动初始化；finish 收尾 flush 尾部并复位。
    std::vector<float> append_incremental(
        const std::vector<float> & log_magnitude_phase,
        int64_t frames,
        const std::vector<float> & window);
    std::vector<float> finish_incremental();

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
