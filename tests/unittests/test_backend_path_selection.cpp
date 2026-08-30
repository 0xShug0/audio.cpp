// Backend path selection for the capabilities that used to be gated on CUDA
// specifically rather than on "is this a GPU".
//
// Everything here is pure logic: no weights, no device, no inference. That is
// the point — the selection rules are what regressed, and they are testable on
// a machine with no GPU at all.

#include "engine/community_models/f5_tts/runtime.h"
#include "engine/framework/audio/fft.h"
#include "engine/framework/audio/istft_graph.h"
#include "engine/framework/core/module.h"

#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace audio = engine::audio;
namespace core = engine::core;
namespace f5 = engine::models::f5_tts;

// ---------------------------------------------------------------------------
// iSTFT: which implementation each backend gets
// ---------------------------------------------------------------------------

void test_istft_path_never_cuda_without_the_cuda_runtime() {
    // The CUDA iSTFT lives in istft_cuda_runtime.cu, compiled only under
    // ENGINE_ENABLE_CUDA AND NOT ENGINE_ENABLE_HIP. If it is absent, no
    // backend may select it — the constructor would throw.
    if (audio::cuda_log_magnitude_phase_istft_available()) {
        return;
    }
    const core::BackendType all[] = {
        core::BackendType::Cpu,
        core::BackendType::Cuda,
        core::BackendType::Hip,
        core::BackendType::Vulkan,
        core::BackendType::Metal,
        core::BackendType::BestAvailable,
    };
    for (const core::BackendType backend : all) {
        engine::test::require(
            audio::select_log_magnitude_phase_istft_path(backend) ==
                audio::LogMagnitudePhaseISTFTPath::Host,
            "iSTFT must select the host path when the CUDA runtime is not built");
    }
}

void test_istft_path_metal_and_cpu_take_the_host_path() {
    // True in every build configuration: there is no Metal or Vulkan iSTFT
    // kernel, so widening the gate to "any GPU" would only produce a throw.
    engine::test::require(
        audio::select_log_magnitude_phase_istft_path(core::BackendType::Metal) ==
            audio::LogMagnitudePhaseISTFTPath::Host,
        "Metal must select the host iSTFT");
    engine::test::require(
        audio::select_log_magnitude_phase_istft_path(core::BackendType::Vulkan) ==
            audio::LogMagnitudePhaseISTFTPath::Host,
        "Vulkan must select the host iSTFT");
    engine::test::require(
        audio::select_log_magnitude_phase_istft_path(core::BackendType::Cpu) ==
            audio::LogMagnitudePhaseISTFTPath::Host,
        "CPU must select the host iSTFT");
    engine::test::require(
        audio::select_log_magnitude_phase_istft_path(core::BackendType::Hip) ==
            audio::LogMagnitudePhaseISTFTPath::Host,
        "HIP must select the host iSTFT: the .cu is not compiled for HIP");
}

void test_istft_path_cuda_follows_the_build() {
    const auto expected = audio::cuda_log_magnitude_phase_istft_available()
        ? audio::LogMagnitudePhaseISTFTPath::Cuda
        : audio::LogMagnitudePhaseISTFTPath::Host;
    engine::test::require(
        audio::select_log_magnitude_phase_istft_path(core::BackendType::Cuda) == expected,
        "a CUDA backend must select the CUDA iSTFT exactly when it is built");
}

// ---------------------------------------------------------------------------
// F5-TTS: DiT and Vocos device-vs-host selection
// ---------------------------------------------------------------------------

// The predicates are constexpr, so the contract is also checked at compile
// time. A Metal session must reach the device path; only Cpu must not.
static_assert(
    f5::f5_uses_device_backend(f5::F5ComputeDevice{core::BackendType::Metal}),
    "F5-TTS must run the DiT and the vocoder on Metal, not on the CPU");
static_assert(
    f5::f5_uses_device_backend(f5::F5ComputeDevice{core::BackendType::Cuda}),
    "F5-TTS must run the DiT and the vocoder on CUDA");
static_assert(
    f5::f5_uses_device_backend(f5::F5ComputeDevice{core::BackendType::Vulkan}),
    "F5-TTS must run the DiT and the vocoder on Vulkan");
static_assert(
    !f5::f5_uses_device_backend(f5::F5ComputeDevice{core::BackendType::Cpu}),
    "F5-TTS must run on the host path when there is no GPU");

void test_f5_backend_selection() {
    f5::F5ComputeDevice metal;
    metal.backend = core::BackendType::Metal;
    engine::test::require(
        f5::f5_requested_backend(metal) == core::BackendType::Metal,
        "F5 must ask for the Metal backend it was given");
    engine::test::require(
        f5::f5_uses_device_backend(metal),
        "a Metal F5 session must take the device path");

    f5::F5ComputeDevice host;
    engine::test::require(
        f5::f5_requested_backend(host) == core::BackendType::Cpu,
        "a default F5 device must be CPU");
    engine::test::require(
        !f5::f5_uses_device_backend(host),
        "a CPU-only F5 session must take the host path");

    // The pre-`backend` spelling still works: the parity harnesses under
    // tests/f5_tts set only use_cuda.
    f5::F5ComputeDevice legacy;
    legacy.use_cuda = true;
    engine::test::require(
        f5::f5_requested_backend(legacy) == core::BackendType::Cuda,
        "the legacy use_cuda flag must still resolve to the CUDA backend");
    engine::test::require(
        f5::f5_uses_device_backend(legacy),
        "the legacy use_cuda flag must still take the device path");

    // An explicit backend wins over the legacy flag.
    f5::F5ComputeDevice both;
    both.backend = core::BackendType::Metal;
    both.use_cuda = true;
    engine::test::require(
        f5::f5_requested_backend(both) == core::BackendType::Metal,
        "an explicit backend must win over the legacy use_cuda flag");
}

// ---------------------------------------------------------------------------
// The host iSTFT overlap-add rewrite is bit-identical to the loop it replaced
// ---------------------------------------------------------------------------

// The serial fold was a scatter over frames, which cannot be an OpenMP loop
// without a race; it is now a gather over output samples. The gather visits
// each output sample's contributing frames in increasing frame order, which is
// the order the scatter reached them in, so the float accumulation order is
// unchanged and the result must match to the last bit — not merely to a
// tolerance. Assert exactly that: max |difference| == 0.
void test_host_istft_matches_the_scatter_reference_bit_for_bit() {
    // Large enough that the fold and normalise loops clear their OpenMP
    // `if` thresholds, so the parallel path is what is being compared.
    constexpr int64_t kFrames = 500;
    constexpr int64_t kNfft = 64;
    constexpr int64_t kHop = 16;
    constexpr int64_t kFreqBins = kNfft / 2 + 1;
    constexpr int64_t kOutDim = kFreqBins * 2;

    // Periodic Hann: sum of squares at hop N/4 is a strictly positive envelope.
    std::vector<float> window(static_cast<size_t>(kNfft));
    for (int64_t i = 0; i < kNfft; ++i) {
        window[static_cast<size_t>(i)] = 0.5F -
            0.5F * std::cos(6.283185307179586F * static_cast<float>(i) / static_cast<float>(kNfft));
    }

    // Deterministic, reproducible log-magnitude/phase field with enough
    // dynamic range that a reordered accumulation would show up.
    std::vector<float> log_magnitude_phase(static_cast<size_t>(kFrames * kOutDim));
    for (int64_t frame = 0; frame < kFrames; ++frame) {
        for (int64_t bin = 0; bin < kFreqBins; ++bin) {
            const auto t = static_cast<float>(frame);
            const auto f = static_cast<float>(bin);
            const size_t row = static_cast<size_t>(frame * kOutDim);
            log_magnitude_phase[row + static_cast<size_t>(bin)] =
                -2.0F + 3.0F * std::sin(0.31F * t + 0.17F * f);
            log_magnitude_phase[row + static_cast<size_t>(kFreqBins + bin)] =
                std::cos(0.11F * t - 0.23F * f) * 3.0F;
        }
    }

    audio::HostLogMagnitudePhaseISTFTConfig config;
    config.frames = kFrames;
    config.n_fft = kNfft;
    config.hop_length = kHop;
    config.out_dim = kOutDim;
    config.threads = 4;
    audio::HostLogMagnitudePhaseISTFT istft(config);
    const auto actual = istft.compute(log_magnitude_phase, window);

    // ---- independent reference: the pre-change scatter loop ----
    std::vector<std::complex<float>> spectrum(static_cast<size_t>(kFrames * kFreqBins));
    for (int64_t frame = 0; frame < kFrames; ++frame) {
        const size_t row = static_cast<size_t>(frame * kOutDim);
        for (int64_t bin = 0; bin < kFreqBins; ++bin) {
            const float magnitude =
                std::min(std::exp(log_magnitude_phase[row + static_cast<size_t>(bin)]), 100.0F);
            const float phase = log_magnitude_phase[row + static_cast<size_t>(kFreqBins + bin)];
            spectrum[static_cast<size_t>(frame * kFreqBins + bin)] = {
                magnitude * std::cos(phase),
                magnitude * std::sin(phase),
            };
        }
    }
    std::vector<float> framed(static_cast<size_t>(kFrames * kNfft), 0.0F);
    audio::real_fft_inverse(
        {static_cast<size_t>(kFrames), static_cast<size_t>(kNfft)},
        {
            static_cast<std::ptrdiff_t>(kFreqBins * static_cast<int64_t>(sizeof(std::complex<float>))),
            static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)),
        },
        {
            static_cast<std::ptrdiff_t>(kNfft * static_cast<int64_t>(sizeof(float))),
            static_cast<std::ptrdiff_t>(sizeof(float)),
        },
        1,
        spectrum.data(),
        framed.data(),
        1.0F / static_cast<float>(kNfft),
        4);  // same thread count the runtime used, so the FFT input is identical

    const int64_t output_size = (kFrames - 1) * kHop + kNfft;
    std::vector<float> folded(static_cast<size_t>(output_size), 0.0F);
    std::vector<float> envelope(static_cast<size_t>(output_size), 0.0F);
    for (int64_t frame = 0; frame < kFrames; ++frame) {
        const int64_t start = frame * kHop;
        const float * src = framed.data() + static_cast<size_t>(frame * kNfft);
        for (int64_t i = 0; i < kNfft; ++i) {
            const float w = window[static_cast<size_t>(i)];
            folded[static_cast<size_t>(start + i)] += src[i] * w;
            envelope[static_cast<size_t>(start + i)] += w * w;
        }
    }
    const int64_t pad = (kNfft - kHop) / 2;
    const int64_t samples = output_size - 2 * pad;
    std::vector<float> expected(static_cast<size_t>(samples));
    for (int64_t i = 0; i < samples; ++i) {
        const int64_t src = i + pad;
        expected[static_cast<size_t>(i)] =
            folded[static_cast<size_t>(src)] / envelope[static_cast<size_t>(src)];
    }

    engine::test::require_eq(actual.audio.size(), expected.size(), "host iSTFT sample count");
    float max_abs_difference = 0.0F;
    float peak = 0.0F;
    for (size_t i = 0; i < expected.size(); ++i) {
        max_abs_difference = std::max(
            max_abs_difference, std::fabs(actual.audio[i] - expected[i]));
        peak = std::max(peak, std::fabs(expected[i]));
    }
    engine::test::require(peak > 1.0e-3F, "host iSTFT reference signal must not be silent");
    engine::test::require_close(
        max_abs_difference, 0.0F, 0.0F,
        "host iSTFT overlap-add must be bit-identical to the scatter reference");
}

void test_host_istft_rejects_a_degenerate_window() {
    // A window whose squared envelope underflows must still be rejected, and
    // the check must fire even though the normalise loop no longer branches.
    constexpr int64_t kFrames = 8;
    constexpr int64_t kNfft = 32;
    constexpr int64_t kHop = 8;
    constexpr int64_t kFreqBins = kNfft / 2 + 1;
    constexpr int64_t kOutDim = kFreqBins * 2;

    audio::HostLogMagnitudePhaseISTFTConfig config;
    config.frames = kFrames;
    config.n_fft = kNfft;
    config.hop_length = kHop;
    config.out_dim = kOutDim;
    config.threads = 1;
    audio::HostLogMagnitudePhaseISTFT istft(config);

    const std::vector<float> zero_window(static_cast<size_t>(kNfft), 0.0F);
    const std::vector<float> input(static_cast<size_t>(kFrames * kOutDim), 0.0F);
    bool threw = false;
    try {
        (void) istft.compute(input, zero_window);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    engine::test::require(threw, "a zero window must be rejected as an envelope underflow");
}

}  // namespace

int main() {
    try {
        test_istft_path_never_cuda_without_the_cuda_runtime();
        test_istft_path_metal_and_cpu_take_the_host_path();
        test_istft_path_cuda_follows_the_build();
        test_f5_backend_selection();
        test_host_istft_matches_the_scatter_reference_bit_for_bit();
        test_host_istft_rejects_a_degenerate_window();
        std::cout << "backend_path_selection_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "backend_path_selection_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
