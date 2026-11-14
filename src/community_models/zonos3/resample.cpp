#include "engine/community_models/zonos3/resample.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace engine::models::zonos3 {

std::vector<float> zonos3_resample_44k_to_16k(const std::vector<float> & kernel, const std::vector<float> & x) {
    constexpr int kOrig = 441;
    constexpr int kRate = 160;
    constexpr int kWidth = 17;
    constexpr int kKernel = 475;
    if (kernel.size() != static_cast<size_t>(kRate) * kKernel) {
        throw std::runtime_error("zonos3: the 44.1 -> 16 kHz resampler kernel must be [160][475]");
    }
    // Each phase only reads its kernel's non-zero support [begin, end).
    std::array<int, kRate> begin{};
    std::array<int, kRate> end{};
    for (int phase = 0; phase < kRate; ++phase) {
        const float * row = kernel.data() + static_cast<size_t>(phase) * kKernel;
        int b = 0;
        int e = kKernel;
        while (b < e && row[b] == 0.0f) ++b;
        while (e > b && row[e - 1] == 0.0f) --e;
        begin[static_cast<size_t>(phase)] = b;
        end[static_cast<size_t>(phase)] = e;
    }
    const auto n = static_cast<int64_t>(x.size());
    const int count = static_cast<int>(std::ceil(static_cast<float>(static_cast<double>(kRate) * static_cast<double>(n) / kOrig)));
    std::vector<float> out(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        const int phase = i % kRate;
        const int64_t start = static_cast<int64_t>(i / kRate) * kOrig - kWidth;
        const float * row = kernel.data() + static_cast<size_t>(phase) * kKernel;
        float sum = 0.0f;
        const int lo = static_cast<int>(std::max<int64_t>(begin[static_cast<size_t>(phase)], -start));
        const int hi = static_cast<int>(std::min<int64_t>(end[static_cast<size_t>(phase)], n - start));
        for (int j = lo; j < hi; ++j) sum += x[static_cast<size_t>(start + j)] * row[j];
        out[static_cast<size_t>(i)] = sum;
    }
    return out;
}

}  // namespace engine::models::zonos3
