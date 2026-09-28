#include "mask_accumulation.h"

#include <algorithm>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

void exercise(int threads) {
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
#else
    (void)threads;
#endif
    constexpr int64_t frames = 257;
    constexpr int64_t overlapping = 192;
    constexpr int64_t bands = overlapping + 2;
    std::vector<int64_t> indices(bands, 0);
    indices[bands - 2] = indices[bands - 1] = 1;
    std::vector<float> raw(frames * bands * 2);
    for (int64_t t = 0; t < frames; ++t) {
        for (int64_t m = 0; m < bands; ++m) {
            const float value = m < overlapping
                ? (m % 3 == 0 ? 1.0e8F : m % 3 == 1 ? -1.0e8F : 1.0F)
                : static_cast<float>(t + 1);
            raw[(t * bands + m) * 2] = value;
            raw[(t * bands + m) * 2 + 1] = -value;
        }
    }
    for (int run = 0; run < 40; ++run) {
        std::vector<float> output(3 * frames * 2, 0.0F);
        engine::models::roformer::detail::accumulate_band_masks(raw, indices, frames, bands * 2, output);
        for (int64_t t = 0; t < frames; ++t) {
            // Each triplet loses the previous unit at 1e8 then ends at +/-1.
            // Different orders or concurrent writers violate this result.
            if (output[t * 2] != 1.0F || output[t * 2 + 1] != -1.0F ||
                output[(frames + t) * 2] != 2.0F * (t + 1) ||
                output[(frames + t) * 2 + 1] != -2.0F * (t + 1) ||
                output[(2 * frames + t) * 2] != 0.0F || output[(2 * frames + t) * 2 + 1] != 0.0F)
                throw std::runtime_error("overlapping band masks lost updates or changed accumulation order");
        }
    }
}

} // namespace

int main() {
    try {
        for (int threads : {1, 2, 8}) exercise(threads);
        auto a = std::async(std::launch::async, [] { exercise(4); });
        auto b = std::async(std::launch::async, [] { exercise(4); });
        a.get(); b.get();
        std::cout << "roformer_mask_accumulation_test: ok\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
