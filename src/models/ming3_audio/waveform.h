#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace engine::models::ming3_audio::detail {

// Trim trailing low-energy 100 ms windows while retaining a 300 ms tail.
// Include a final partial window so samples after the last full window are
// measured and never discarded just because they do not fill another window.
inline void trim_trailing_silence(std::vector<float> & samples, int64_t sample_rate) {
    if (samples.empty()) {
        return;
    }
    if (sample_rate <= 0) {
        throw std::invalid_argument("Ming3 Audio trim requires a positive sample rate");
    }

    const size_t window = static_cast<size_t>(std::max<int64_t>(1, sample_rate / 10));
    const size_t tail = static_cast<size_t>(sample_rate * 3 / 10);
    const size_t window_count = 1 + (samples.size() - 1) / window;
    size_t last = window_count;
    while (last > 0) {
        const size_t begin = (last - 1) * window;
        const size_t end = std::min(begin + window, samples.size());
        double energy = 0.0;
        for (size_t i = begin; i < end; ++i) {
            energy += std::abs(samples[i]);
        }
        if (energy / static_cast<double>(end - begin) > 1e-3) {
            break;
        }
        --last;
    }

    if (last == window_count) {
        return;
    }
    const size_t tail_start = std::min(last * window, samples.size());
    const size_t keep = tail > samples.size() - tail_start ? samples.size() : tail_start + tail;
    samples.resize(keep);
}

}  // namespace engine::models::ming3_audio::detail
