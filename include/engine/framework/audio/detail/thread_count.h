#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace engine::audio::detail {

// Keep OpenMP stages and the audio FFT on the same explicit request budget.
// A zero request follows the runtime's OpenMP setting when available and the
// hardware concurrency fallback used by the audio FFT otherwise.
inline size_t resolve_thread_count(size_t requested) noexcept {
    if (requested != 0) {
        return requested;
    }
#ifdef _OPENMP
    const int openmp_threads = omp_get_max_threads();
    return static_cast<size_t>(std::max(1, openmp_threads));
#else
    return static_cast<size_t>(std::max(1u, std::thread::hardware_concurrency()));
#endif
}

inline int resolve_openmp_thread_count(size_t requested) noexcept {
    const size_t bounded = std::min(
        resolve_thread_count(requested),
        static_cast<size_t>(std::numeric_limits<int>::max()));
    return static_cast<int>(bounded);
}

inline int recommended_openmp_thread_count(size_t requested, size_t work_items) noexcept {
    const int requested_threads = resolve_openmp_thread_count(requested);
    // Avoid waking a full OpenMP team for work that has too few independent
    // items to amortize scheduling and barrier costs.
    const size_t useful_threads = std::max<size_t>(1, work_items / 256);
    return static_cast<int>(std::min<size_t>(
        static_cast<size_t>(requested_threads), useful_threads));
}

}  // namespace engine::audio::detail
