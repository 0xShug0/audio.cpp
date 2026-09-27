#include "istft_cuda_runtime.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::audio::detail {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void check_cuda(cudaError_t status, const char * label) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
    }
}

void check_cufft(cufftResult status, const char * label) {
    if (status != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string(label) + ": cufft status " + std::to_string(static_cast<int>(status)));
    }
}

struct CudaTimer {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    CudaTimer() {
        check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    }

    ~CudaTimer() {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
    }

    void begin() {
        check_cuda(cudaEventRecord(start, 0), "cudaEventRecord start");
    }

    double end_ms(const char * label) {
        check_cuda(cudaEventRecord(stop, 0), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop), label);
        float ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        return static_cast<double>(ms);
    }
};

__global__ void build_spectrum_kernel(
    const float * head,
    cufftComplex * spectrum,
    int frames,
    int out_dim,
    int freq_bins) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = frames * freq_bins;
    if (index >= total) {
        return;
    }
    const int frame = index / freq_bins;
    const int freq = index - frame * freq_bins;
    const float * row = head + frame * out_dim;
    const float mag = fminf(expf(row[freq]), 100.0F);
    const float phase = row[freq_bins + freq];
    float sin_value = 0.0F;
    float cos_value = 0.0F;
    sincosf(phase, &sin_value, &cos_value);
    spectrum[index] = make_cuFloatComplex(mag * cos_value, mag * sin_value);
}

__global__ void overlap_add_kernel(
    const float * framed,
    const float * window,
    float * folded,
    float * envelope,
    int frames,
    int n_fft,
    int hop_length) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = frames * n_fft;
    if (index >= total) {
        return;
    }
    const int frame = index / n_fft;
    const int i = index - frame * n_fft;
    const int dst = frame * hop_length + i;
    const float w = window[i];
    atomicAdd(folded + dst, framed[index] * w / static_cast<float>(n_fft));
    atomicAdd(envelope + dst, w * w);
}

__global__ void normalize_kernel(
    const float * folded,
    const float * envelope,
    float * audio,
    int samples,
    int pad) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= samples) {
        return;
    }
    const int src = index + pad;
    audio[index] = folded[src] / envelope[src];
}

// One thread owns each output sample, so scheduling cannot reorder its sum.
__global__ void deterministic_overlap_add_kernel(
    const float * framed,
    const float * window,
    float * folded,
    float * envelope,
    int samples,
    int frames,
    int n_fft,
    int hop_length) {
    const int dst = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst >= samples) return;
    const int first = dst < n_fft ? 0 : (dst - n_fft) / hop_length + 1;
    const int last = min(dst / hop_length, frames - 1);
    float value = 0.0F;
    float weight = 0.0F;
    for (int frame = first; frame <= last; ++frame) {
        const int i = dst - frame * hop_length;
        const float w = window[i];
        // Keep the original product/division rounding, without fused adds.
        value = __fadd_rn(value, __fdiv_rn(__fmul_rn(framed[frame * n_fft + i], w), static_cast<float>(n_fft)));
        weight = __fadd_rn(weight, __fmul_rn(w, w));
    }
    folded[dst] = value;
    envelope[dst] = weight;
}

int blocks_for(int count) {
    constexpr int kThreads = 256;
    return (count + kThreads - 1) / kThreads;
}

}  // namespace

class CudaIstftRuntime::Impl {
public:
    explicit Impl(const CudaIstftRuntimeConfig & config)
        : config_(config),
          freq_bins_(config.n_fft / 2 + 1),
          pad_((config.n_fft - config.hop_length) / 2),
          output_size_((config.frames - 1) * config.hop_length + config.n_fft),
          samples_(output_size_ - 2 * pad_) {
        if (config_.frames <= 0 || config_.n_fft <= 0 || config_.hop_length <= 0) {
            throw std::runtime_error("CUDA ISTFT requires positive shape parameters");
        }
        if (config_.out_dim != freq_bins_ * 2) {
            throw std::runtime_error("CUDA ISTFT expects log-magnitude and phase halves");
        }
        if (samples_ <= 0) {
            throw std::runtime_error("CUDA ISTFT produced non-positive sample count");
        }

        check_cuda(cudaSetDevice(config_.device), "cudaSetDevice");
        check_cuda(cudaMalloc(&head_, bytes(config_.frames * config_.out_dim)), "cudaMalloc head");
        check_cuda(cudaMalloc(&window_, bytes(config_.n_fft)), "cudaMalloc window");
        check_cuda(cudaMalloc(&spectrum_, sizeof(cufftComplex) * static_cast<size_t>(config_.frames * freq_bins_)), "cudaMalloc spectrum");
        check_cuda(cudaMalloc(&framed_, bytes(config_.frames * config_.n_fft)), "cudaMalloc framed");
        check_cuda(cudaMalloc(&folded_, bytes(output_size_)), "cudaMalloc folded");
        check_cuda(cudaMalloc(&envelope_, bytes(output_size_)), "cudaMalloc envelope");
        check_cuda(cudaMalloc(&audio_, bytes(samples_)), "cudaMalloc audio");

        int n[] = {static_cast<int>(config_.n_fft)};
        int inembed[] = {static_cast<int>(freq_bins_)};
        int onembed[] = {static_cast<int>(config_.n_fft)};
        check_cufft(
            cufftPlanMany(
                &plan_,
                1,
                n,
                inembed,
                1,
                static_cast<int>(freq_bins_),
                onembed,
                1,
                static_cast<int>(config_.n_fft),
                CUFFT_C2R,
                static_cast<int>(config_.frames)),
            "cufftPlanMany");
    }

    ~Impl() {
        if (plan_ != 0) {
            cufftDestroy(plan_);
        }
        cudaFree(audio_);
        cudaFree(envelope_);
        cudaFree(folded_);
        cudaFree(framed_);
        cudaFree(spectrum_);
        cudaFree(window_);
        cudaFree(head_);
    }

    CudaIstftRuntimeResult compute(
        const std::vector<float> & log_magnitude_phase,
        const std::vector<float> & window) {
        if (static_cast<int64_t>(log_magnitude_phase.size()) != config_.frames * config_.out_dim) {
            throw std::runtime_error("CUDA ISTFT input size mismatch");
        }
        if (static_cast<int64_t>(window.size()) != config_.n_fft) {
            throw std::runtime_error("CUDA ISTFT window size mismatch");
        }

        CudaIstftRuntimeResult result;
        auto & timing = result.timing;
        const auto total_start = Clock::now();
        check_cuda(cudaSetDevice(config_.device), "cudaSetDevice");

        auto timing_start = Clock::now();
        if (window_cache_ != window) {
            check_cuda(cudaMemcpy(window_, window.data(), bytes(config_.n_fft), cudaMemcpyHostToDevice), "cudaMemcpy window");
            window_cache_ = window;
        }
        check_cuda(
            cudaMemcpy(head_, log_magnitude_phase.data(), bytes(config_.frames * config_.out_dim), cudaMemcpyHostToDevice),
            "cudaMemcpy head");
        timing.input_upload_ms = elapsed_ms(timing_start, Clock::now());

        CudaTimer timer;
        timer.begin();
        build_spectrum_kernel<<<blocks_for(static_cast<int>(config_.frames * freq_bins_)), 256>>>(
            head_,
            spectrum_,
            static_cast<int>(config_.frames),
            static_cast<int>(config_.out_dim),
            static_cast<int>(freq_bins_));
        check_cuda(cudaGetLastError(), "build_spectrum_kernel");
        timing.spectrum_kernel_ms = timer.end_ms("build_spectrum_kernel synchronize");

        timer.begin();
        check_cufft(cufftExecC2R(plan_, spectrum_, framed_), "cufftExecC2R");
        timing.fft_inverse_ms = timer.end_ms("cufftExecC2R synchronize");

        timer.begin();
        if (config_.deterministic_overlap_add) {
            deterministic_overlap_add_kernel<<<blocks_for(static_cast<int>(output_size_)), 256>>>(
                framed_, window_, folded_, envelope_, static_cast<int>(output_size_),
                static_cast<int>(config_.frames), static_cast<int>(config_.n_fft),
                static_cast<int>(config_.hop_length));
            check_cuda(cudaGetLastError(), "deterministic_overlap_add_kernel");
        } else {
            check_cuda(cudaMemset(folded_, 0, bytes(output_size_)), "cudaMemset folded");
            check_cuda(cudaMemset(envelope_, 0, bytes(output_size_)), "cudaMemset envelope");
            overlap_add_kernel<<<blocks_for(static_cast<int>(config_.frames * config_.n_fft)), 256>>>(
                framed_,
                window_,
                folded_,
                envelope_,
                static_cast<int>(config_.frames),
                static_cast<int>(config_.n_fft),
                static_cast<int>(config_.hop_length));
            check_cuda(cudaGetLastError(), "overlap_add_kernel");
        }
        timing.overlap_add_ms = timer.end_ms("overlap_add_kernel synchronize");

        timer.begin();
        normalize_kernel<<<blocks_for(static_cast<int>(samples_)), 256>>>(
            folded_,
            envelope_,
            audio_,
            static_cast<int>(samples_),
            static_cast<int>(pad_));
        check_cuda(cudaGetLastError(), "normalize_kernel");
        timing.normalize_ms = timer.end_ms("normalize_kernel synchronize");

        timing_start = Clock::now();
        result.audio.resize(static_cast<size_t>(samples_));
        check_cuda(cudaMemcpy(result.audio.data(), audio_, bytes(samples_), cudaMemcpyDeviceToHost), "cudaMemcpy audio");
        timing.audio_read_ms = elapsed_ms(timing_start, Clock::now());
        timing.total_ms = elapsed_ms(total_start, Clock::now());
        return result;
    }

private:
    static size_t bytes(int64_t elements) {
        return sizeof(float) * static_cast<size_t>(elements);
    }

    CudaIstftRuntimeConfig config_;
    int64_t freq_bins_ = 0;
    int64_t pad_ = 0;
    int64_t output_size_ = 0;
    int64_t samples_ = 0;
    float * head_ = nullptr;
    float * window_ = nullptr;
    cufftComplex * spectrum_ = nullptr;
    float * framed_ = nullptr;
    float * folded_ = nullptr;
    float * envelope_ = nullptr;
    float * audio_ = nullptr;
    cufftHandle plan_ = 0;
    std::vector<float> window_cache_;
};

CudaIstftRuntime::CudaIstftRuntime(const CudaIstftRuntimeConfig & config)
    : impl_(std::make_unique<Impl>(config)) {}

CudaIstftRuntime::~CudaIstftRuntime() = default;

CudaIstftRuntime::CudaIstftRuntime(CudaIstftRuntime &&) noexcept = default;

CudaIstftRuntime & CudaIstftRuntime::operator=(CudaIstftRuntime &&) noexcept = default;

CudaIstftRuntimeResult CudaIstftRuntime::compute(
    const std::vector<float> & log_magnitude_phase,
    const std::vector<float> & window) {
    return impl_->compute(log_magnitude_phase, window);
}

}  // namespace engine::audio::detail
