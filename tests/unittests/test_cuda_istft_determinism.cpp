#include "engine/framework/audio/istft_graph.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require_close(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) throw std::runtime_error("ISTFT output length differs");
    double squared = 0.0;
    float maximum = 0.0F;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("nonfinite ISTFT output");
        const float error = a[i] - b[i];
        maximum = std::max(maximum, std::abs(error));
        squared += static_cast<double>(error) * error;
    }
    if (maximum > 1.0e-5F || std::sqrt(squared / a.size()) > 1.0e-6)
        throw std::runtime_error("deterministic ISTFT differs from reference");
}

void run(int frames, int n_fft, int hop) {
    using namespace engine::audio;
    const int out_dim = (n_fft / 2 + 1) * 2;
    std::vector<float> head(static_cast<size_t>(frames * out_dim));
    for (int f = 0; f < frames; ++f) {
        for (int i = 0; i < out_dim / 2; ++i) {
            head[f * out_dim + i] = -2.0F + 0.3F * std::sin(float(f * 31 + i));
            head[f * out_dim + out_dim / 2 + i] = 0.7F * std::cos(float(f * 19 + i));
        }
    }
    std::vector<float> window(n_fft);
    for (int i = 0; i < n_fft; ++i) window[i] = 0.6F - 0.4F * std::cos(6.283185307179586F * i / n_fft);
    HostLogMagnitudePhaseISTFT host({frames, n_fft, hop, out_dim, 2});
    CudaLogMagnitudePhaseISTFT first({frames, n_fft, hop, out_dim, 0, true});
    CudaLogMagnitudePhaseISTFT second({frames, n_fft, hop, out_dim, 0, true});
    CudaLogMagnitudePhaseISTFT original({frames, n_fft, hop, out_dim, 0});
    const auto expected = first.compute(head, window).audio;
    require_close(expected, host.compute(head, window).audio);
    require_close(expected, original.compute(head, window).audio);
    auto repeat = [&](CudaLogMagnitudePhaseISTFT & runtime) {
        for (int i = 0; i < 32; ++i) {
            const auto value = runtime.compute(head, window).audio;
            if (value.size() != expected.size() ||
                std::memcmp(value.data(), expected.data(), value.size() * sizeof(float)) != 0)
                throw std::runtime_error("ISTFT changes across repeat or independent session");
        }
    };
    auto a = std::async(std::launch::async, [&] { repeat(first); });
    auto b = std::async(std::launch::async, [&] { repeat(second); });
    a.get(); b.get();
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    try {
        run(1, 392, 98);
        run(37, 392, 98);
        run(41, 1920, 480);
        run(19, 258, 73);
        run(17, 256, 256);
        std::cout << "cuda_istft_determinism_test: ok\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
