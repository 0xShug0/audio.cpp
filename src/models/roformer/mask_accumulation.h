#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine::models::roformer::detail {

inline void accumulate_band_masks(
    const std::vector<float> & raw_masks,
    const std::vector<int64_t> & merged_freq_indices,
    int64_t frames,
    int64_t total_band_input_dim,
    std::vector<float> & averaged_masks) {
    const int64_t merged = static_cast<int64_t>(merged_freq_indices.size());
#ifdef _OPENMP
    #pragma omp parallel for if(merged * frames >= 4096)
#endif
    // Bands overlap in frequency. Each frame has one writer; the serial band
    // order is preserved without atomic additions or an inference lock.
    for (int64_t t = 0; t < frames; ++t) {
        for (int64_t m = 0; m < merged; ++m) {
            const int64_t merged_index = merged_freq_indices[static_cast<size_t>(m)];
            const size_t src = static_cast<size_t>(t * total_band_input_dim + m * 2);
            const size_t dst = static_cast<size_t>(((merged_index * frames) + t) * 2);
            averaged_masks[dst] += raw_masks[src];
            averaged_masks[dst + 1] += raw_masks[src + 1];
        }
    }
}

} // namespace engine::models::roformer::detail
