#include "engine/framework/audio/istft_graph.h"

#include "engine/framework/audio/fft.h"
#ifdef ENGINE_HAS_CUDA_ISTFT
#include "istft_cuda_runtime.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace engine::audio {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct Workspace {
    int64_t frames = 0;
    int64_t freq_bins = 0;
    int64_t n_fft = 0;
    int64_t output_size = 0;
    std::vector<std::complex<float>> spectrum;
    std::vector<float> framed;
    std::vector<float> folded;
    std::vector<float> envelope;
    std::vector<float> audio;
    std::vector<float> window;
};

struct WorkspaceConfig {
    int64_t frames = 0;
    int64_t n_fft = 0;
    int64_t hop_length = 0;
};

void ensure_workspace(
    Workspace & workspace,
    const WorkspaceConfig & config,
    int64_t freq_bins,
    const std::vector<float> & window,
    const char * label) {
    const int64_t pad = (config.n_fft - config.hop_length) / 2;
    const int64_t output_size = (config.frames - 1) * config.hop_length + config.n_fft;
    const int64_t samples = output_size - 2 * pad;
    require(samples > 0, label);
    if (workspace.frames == config.frames &&
        workspace.freq_bins == freq_bins &&
        workspace.n_fft == config.n_fft &&
        workspace.output_size == output_size &&
        workspace.window == window) {
        return;
    }

    workspace.frames = config.frames;
    workspace.freq_bins = freq_bins;
    workspace.n_fft = config.n_fft;
    workspace.output_size = output_size;
    workspace.window = window;
    workspace.spectrum.resize(static_cast<size_t>(config.frames * freq_bins));
    workspace.framed.resize(static_cast<size_t>(config.frames * config.n_fft));
    workspace.folded.resize(static_cast<size_t>(output_size));
    workspace.envelope.assign(static_cast<size_t>(output_size), 0.0F);
    workspace.audio.resize(static_cast<size_t>(samples));
    for (int64_t frame = 0; frame < config.frames; ++frame) {
        const int64_t start = frame * config.hop_length;
        for (int64_t i = 0; i < config.n_fft; ++i) {
            const float w = window[static_cast<size_t>(i)];
            workspace.envelope[static_cast<size_t>(start + i)] += w * w;
        }
    }
}

template <typename Timing>
void finish_istft_from_spectrum(
    Workspace & workspace,
    const WorkspaceConfig & config,
    int64_t freq_bins,
    const std::vector<float> & window,
    size_t threads,
    Timing & timing,
    const char * label) {
    ensure_workspace(workspace, config, freq_bins, window, label);

    auto timing_start = Clock::now();
    std::fill(workspace.framed.begin(), workspace.framed.end(), 0.0F);
    timing.framed_clear_ms = elapsed_ms(timing_start, Clock::now());

    timing_start = Clock::now();
    real_fft_inverse(
        {static_cast<size_t>(config.frames), static_cast<size_t>(config.n_fft)},
        {
            static_cast<std::ptrdiff_t>(freq_bins * static_cast<int64_t>(sizeof(std::complex<float>))),
            static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)),
        },
        {
            static_cast<std::ptrdiff_t>(config.n_fft * static_cast<int64_t>(sizeof(float))),
            static_cast<std::ptrdiff_t>(sizeof(float)),
        },
        1,
        workspace.spectrum.data(),
        workspace.framed.data(),
        1.0F / static_cast<float>(config.n_fft),
        threads);
    timing.fft_inverse_ms = elapsed_ms(timing_start, Clock::now());

    timing_start = Clock::now();
    std::fill(workspace.folded.begin(), workspace.folded.end(), 0.0F);
    timing.fold_clear_ms = elapsed_ms(timing_start, Clock::now());

    timing_start = Clock::now();
    for (int64_t frame = 0; frame < config.frames; ++frame) {
        const int64_t start = frame * config.hop_length;
        const float * src = workspace.framed.data() + static_cast<size_t>(frame * config.n_fft);
        for (int64_t i = 0; i < config.n_fft; ++i) {
            const float w = window[static_cast<size_t>(i)];
            workspace.folded[static_cast<size_t>(start + i)] += src[i] * w;
        }
    }
    timing.overlap_add_ms = elapsed_ms(timing_start, Clock::now());

    const int64_t pad = (config.n_fft - config.hop_length) / 2;
    const int64_t samples = workspace.output_size - 2 * pad;
    timing_start = Clock::now();
    for (int64_t i = 0; i < samples; ++i) {
        const int64_t src = i + pad;
        const float denom = workspace.envelope[static_cast<size_t>(src)];
        if (denom <= 1.0e-11F) {
            throw std::runtime_error("ISTFT window envelope underflow");
        }
        workspace.audio[static_cast<size_t>(i)] = workspace.folded[static_cast<size_t>(src)] / denom;
    }
    timing.normalize_ms = elapsed_ms(timing_start, Clock::now());
}

}  // namespace

class HostLogMagnitudePhaseISTFT::Impl {
public:
    explicit Impl(const HostLogMagnitudePhaseISTFTConfig & config)
        : config_(config),
          freq_bins_(config.n_fft / 2 + 1) {
        require(config_.frames > 0, "host ISTFT requires positive frames");
        require(config_.n_fft > 0, "host ISTFT requires positive n_fft");
        require(config_.hop_length > 0, "host ISTFT requires positive hop length");
        require(config_.out_dim == freq_bins_ * 2, "host ISTFT expects log-magnitude and phase halves");
    }

    HostLogMagnitudePhaseISTFTResult compute(
        const std::vector<float> & log_magnitude_phase,
        const std::vector<float> & window) {
        require(
            static_cast<int64_t>(log_magnitude_phase.size()) == config_.frames * config_.out_dim,
            "host ISTFT input size mismatch");
        require(
            static_cast<int64_t>(window.size()) == config_.n_fft,
            "host ISTFT window size mismatch");

        HostLogMagnitudePhaseISTFTResult result;
        auto & timing = result.timing;
        const auto total_start = Clock::now();
        const WorkspaceConfig workspace_config{config_.frames, config_.n_fft, config_.hop_length};
        ensure_workspace(workspace_, workspace_config, freq_bins_, window, "host ISTFT produced non-positive samples");

        auto timing_start = Clock::now();
#ifdef _OPENMP
        const int omp_threads = static_cast<int>(std::max<size_t>(1, config_.threads));
#pragma omp parallel for num_threads(omp_threads) if (config_.frames >= 8)
#endif
        for (int64_t frame = 0; frame < config_.frames; ++frame) {
            const float * row = log_magnitude_phase.data() + static_cast<size_t>(frame * config_.out_dim);
            auto * spectrum_row = workspace_.spectrum.data() + static_cast<size_t>(frame * freq_bins_);
            for (int64_t freq = 0; freq < freq_bins_; ++freq) {
                const float mag = std::min(std::exp(row[freq]), 100.0F);
                const float phase = row[freq_bins_ + freq];
                spectrum_row[static_cast<size_t>(freq)] = {
                    mag * std::cos(phase),
                    mag * std::sin(phase),
                };
            }
        }
        timing.spectrum_ms = elapsed_ms(timing_start, Clock::now());

        finish_istft_from_spectrum(
            workspace_,
            workspace_config,
            freq_bins_,
            window,
            config_.threads,
            timing,
            "host ISTFT produced non-positive samples");
        result.audio = workspace_.audio;
        timing.total_ms = elapsed_ms(total_start, Clock::now());
        return result;
    }

private:
    HostLogMagnitudePhaseISTFTConfig config_;
    int64_t freq_bins_ = 0;
    Workspace workspace_;

    // --- 增量状态 ---
    bool inc_initialized_ = false;
    std::vector<float> inc_window_;
    std::vector<float> inc_folded_;
    std::vector<float> inc_envelope_;
    int64_t inc_frames_emitted_ = 0;    // 已折叠的 frame 绝对计数
    int64_t inc_emitted_samples_ = 0;   // 已输出的样本数（累计）

    // 初始化滚动累加器：buffer 长度 = (本块 frame 数 - 1) * hop + n_fft（可增长），
    // envelope 按已加入的 frame 计算。
    void inc_ensure(int64_t total_frames, const std::vector<float> & window) {
        const int64_t need = (total_frames - 1) * config_.hop_length + config_.n_fft;
        if (static_cast<int64_t>(inc_folded_.size()) < need) {
            inc_folded_.resize(static_cast<size_t>(need), 0.0F);
            inc_envelope_.resize(static_cast<size_t>(need), 0.0F);
        }
        inc_window_ = window;
    }

public:
    std::vector<float> append_incremental(
        const std::vector<float> & log_magnitude_phase,
        int64_t frames,
        const std::vector<float> & window) {
        require(static_cast<int64_t>(window.size()) == config_.n_fft, "host incremental ISTFT window size mismatch");
        require(static_cast<int64_t>(log_magnitude_phase.size()) == frames * config_.out_dim,
                "host incremental ISTFT input size mismatch");
        require(frames > 0, "host incremental ISTFT requires positive frames");
        if (!inc_initialized_) {
            inc_folded_.clear();
            inc_envelope_.clear();
            inc_frames_emitted_ = 0;
            inc_emitted_samples_ = 0;
            inc_initialized_ = true;
        }
        // 目标总帧数（含本块）
        const int64_t new_total = inc_frames_emitted_ + frames;
        inc_ensure(new_total, window);

        // 折叠本块 spectrum（frame 绝对索引从 inc_frames_emitted_ 起）
        fold_frames(log_magnitude_phase, frames, window);

        // 重新计算 envelope（基于已加入的帧数）
        std::fill(inc_envelope_.begin(), inc_envelope_.end(), 0.0F);
        for (int64_t f = 0; f < new_total; ++f) {
            const int64_t start = f * config_.hop_length;
            for (int64_t i = 0; i < config_.n_fft; ++i) {
                const int64_t pos = start + i;
                if (pos >= static_cast<int64_t>(inc_envelope_.size())) {
                    break;
                }
                const float w = window[static_cast<size_t>(i)];
                inc_envelope_[static_cast<size_t>(pos)] += w * w;
            }
        }

        // 可输出样本：从 pad 到 (buffer 末尾 - pad)，且只输出"自上次以来新增"的部分。
        // 首块从 pad 起；后续从上次输出的绝对位置 inc_emitted_samples_ 起。
        const int64_t pad = (config_.n_fft - config_.hop_length) / 2;
        const int64_t total_samples = (new_total - 1) * config_.hop_length + config_.n_fft;
        const int64_t out_end = total_samples - pad;
        const int64_t out_start = inc_emitted_samples_ == 0 ? pad : inc_emitted_samples_;
        std::vector<float> out;
        if (out_end > out_start) {
            out.resize(static_cast<size_t>(out_end - out_start));
            for (int64_t i = out_start; i < out_end; ++i) {
                const float denom = inc_envelope_[static_cast<size_t>(i)];
                out[static_cast<size_t>(i - out_start)] = denom <= 1.0e-11F ? 0.0F
                                                          : inc_folded_[static_cast<size_t>(i)] / denom;
            }
        }
        // 记录已输出的绝对样本数（不含 pad 前缀，保持与输出对齐）
        inc_emitted_samples_ = out_end;
        return out;
    }

    std::vector<float> finish_incremental() {
        if (!inc_initialized_) {
            return {};
        }
        // append_incremental 已输出 [pad, total_samples-pad) 的完整内部覆盖区，
        // 与离线 compute() 的裁剪语义（输出 = output_size - 2*pad）一致。
        // 这里只输出"尚未输出"的尾部（若有），避免把整段折叠缓冲区重复倒出
        // （此前从索引 0 全量重放会导致流式输出 2 倍时长）。
        const int64_t start = inc_emitted_samples_;
        const int64_t end = static_cast<int64_t>(inc_folded_.size());
        std::vector<float> out;
        if (end > start) {
            out.resize(static_cast<size_t>(end - start));
            for (int64_t i = start; i < end; ++i) {
                const float denom = inc_envelope_[static_cast<size_t>(i)];
                out[static_cast<size_t>(i - start)] =
                    denom <= 1.0e-11F ? 0.0F
                                      : inc_folded_[static_cast<size_t>(i)] / denom;
            }
        }
        inc_folded_.clear();
        inc_envelope_.clear();
        inc_frames_emitted_ = 0;
        inc_emitted_samples_ = 0;
        inc_initialized_ = false;
        return out;
    }

    // 折叠一批 log-magnitude+phase 帧到滚动累加器（frame 绝对索引从 inc_frames_emitted_ 起）。
    void fold_frames(const std::vector<float> & log_magnitude_phase, int64_t frames, const std::vector<float> & window) {
        // 1) 逐帧把 log-magnitude+phase 转成复数 spectrum
        std::vector<std::complex<float>> spectrum(static_cast<size_t>(frames * freq_bins_));
        for (int64_t frame = 0; frame < frames; ++frame) {
            const float * row = log_magnitude_phase.data() + static_cast<size_t>(frame * config_.out_dim);
            auto * spectrum_row = spectrum.data() + static_cast<size_t>(frame * freq_bins_);
            for (int64_t freq = 0; freq < freq_bins_; ++freq) {
                const float mag = std::min(std::exp(row[freq]), 100.0F);
                const float phase = row[freq_bins_ + freq];
                spectrum_row[static_cast<size_t>(freq)] = {
                    mag * std::cos(phase),
                    mag * std::sin(phase),
                };
            }
        }
        // 2) inverse FFT
        std::vector<float> framed(static_cast<size_t>(frames * config_.n_fft), 0.0F);
        real_fft_inverse(
            {static_cast<size_t>(frames), static_cast<size_t>(config_.n_fft)},
            {
                static_cast<std::ptrdiff_t>(freq_bins_ * static_cast<int64_t>(sizeof(std::complex<float>))),
                static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)),
            },
            {
                static_cast<std::ptrdiff_t>(config_.n_fft * static_cast<int64_t>(sizeof(float))),
                static_cast<std::ptrdiff_t>(sizeof(float)),
            },
            1,
            spectrum.data(),
            framed.data(),
            1.0F / static_cast<float>(config_.n_fft),
            config_.threads);
        // 3) windowed overlap-add 到滚动累加器
        for (int64_t frame = 0; frame < frames; ++frame) {
            const int64_t abs_start = (inc_frames_emitted_ + frame) * config_.hop_length;
            const float * src = framed.data() + static_cast<size_t>(frame * config_.n_fft);
            for (int64_t i = 0; i < config_.n_fft; ++i) {
                const int64_t pos = abs_start + i;
                if (pos >= static_cast<int64_t>(inc_folded_.size())) {
                    continue;
                }
                const float w = window[static_cast<size_t>(i)];
                inc_folded_[static_cast<size_t>(pos)] += src[i] * w;
            }
        }
        inc_frames_emitted_ += frames;
    }
};

class CudaLogMagnitudePhaseISTFT::Impl {
public:
    explicit Impl(const CudaLogMagnitudePhaseISTFTConfig & config)
        : config_(config) {
        require(config_.frames > 0, "CUDA ISTFT requires positive frames");
        require(config_.n_fft > 0, "CUDA ISTFT requires positive n_fft");
        require(config_.hop_length > 0, "CUDA ISTFT requires positive hop length");
        require(config_.out_dim == (config_.n_fft / 2 + 1) * 2, "CUDA ISTFT expects log-magnitude and phase halves");
#ifdef ENGINE_HAS_CUDA_ISTFT
        runtime_ = std::make_unique<detail::CudaIstftRuntime>(
            detail::CudaIstftRuntimeConfig{
                config_.frames,
                config_.n_fft,
                config_.hop_length,
                config_.out_dim,
                config_.device,
            });
#else
        throw std::runtime_error("CUDA ISTFT runtime was not built");
#endif
    }

    CudaLogMagnitudePhaseISTFTResult compute(
        const std::vector<float> & log_magnitude_phase,
        const std::vector<float> & window) {
        require(
            static_cast<int64_t>(log_magnitude_phase.size()) == config_.frames * config_.out_dim,
            "CUDA ISTFT input size mismatch");
        require(
            static_cast<int64_t>(window.size()) == config_.n_fft,
            "CUDA ISTFT window size mismatch");

#ifdef ENGINE_HAS_CUDA_ISTFT
        auto runtime_result = runtime_->compute(log_magnitude_phase, window);
        CudaLogMagnitudePhaseISTFTResult result;
        result.audio = std::move(runtime_result.audio);
        result.timing.input_upload_ms = runtime_result.timing.input_upload_ms;
        result.timing.spectrum_kernel_ms = runtime_result.timing.spectrum_kernel_ms;
        result.timing.fft_inverse_ms = runtime_result.timing.fft_inverse_ms;
        result.timing.overlap_add_ms = runtime_result.timing.overlap_add_ms;
        result.timing.normalize_ms = runtime_result.timing.normalize_ms;
        result.timing.audio_read_ms = runtime_result.timing.audio_read_ms;
        result.timing.total_ms = runtime_result.timing.total_ms;
        return result;
#else
        throw std::runtime_error("CUDA ISTFT runtime was not built");
#endif
    }

private:
    CudaLogMagnitudePhaseISTFTConfig config_;
#ifdef ENGINE_HAS_CUDA_ISTFT
    std::unique_ptr<detail::CudaIstftRuntime> runtime_;
#endif
};

HostLogMagnitudePhaseISTFT::HostLogMagnitudePhaseISTFT(const HostLogMagnitudePhaseISTFTConfig & config)
    : impl_(std::make_unique<Impl>(config)) {}

HostLogMagnitudePhaseISTFT::~HostLogMagnitudePhaseISTFT() = default;

HostLogMagnitudePhaseISTFT::HostLogMagnitudePhaseISTFT(HostLogMagnitudePhaseISTFT &&) noexcept = default;

HostLogMagnitudePhaseISTFT & HostLogMagnitudePhaseISTFT::operator=(
    HostLogMagnitudePhaseISTFT &&) noexcept = default;

HostLogMagnitudePhaseISTFTResult HostLogMagnitudePhaseISTFT::compute(
    const std::vector<float> & log_magnitude_phase,
    const std::vector<float> & window) {
    return impl_->compute(log_magnitude_phase, window);
}

std::vector<float> HostLogMagnitudePhaseISTFT::append_incremental(
    const std::vector<float> & log_magnitude_phase,
    int64_t frames,
    const std::vector<float> & window) {
    return impl_->append_incremental(log_magnitude_phase, frames, window);
}

std::vector<float> HostLogMagnitudePhaseISTFT::finish_incremental() {
    return impl_->finish_incremental();
}

CudaLogMagnitudePhaseISTFT::CudaLogMagnitudePhaseISTFT(
    const CudaLogMagnitudePhaseISTFTConfig & config)
    : impl_(std::make_unique<Impl>(config)) {}

CudaLogMagnitudePhaseISTFT::~CudaLogMagnitudePhaseISTFT() = default;

CudaLogMagnitudePhaseISTFT::CudaLogMagnitudePhaseISTFT(CudaLogMagnitudePhaseISTFT &&) noexcept = default;

CudaLogMagnitudePhaseISTFT & CudaLogMagnitudePhaseISTFT::operator=(
    CudaLogMagnitudePhaseISTFT &&) noexcept = default;

CudaLogMagnitudePhaseISTFTResult CudaLogMagnitudePhaseISTFT::compute(
    const std::vector<float> & log_magnitude_phase,
    const std::vector<float> & window) {
    return impl_->compute(log_magnitude_phase, window);
}

}  // namespace engine::audio
