#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::yue2::detail {

// Splits the codec frames into NAR chunks. Every chunk attends to the whole
// prefix plus its own frames, so attention cost grows with the square of the
// chunk length. A nonzero window caps the chunk length, and consecutive chunks
// share overlap frames. With window == 0 the chunks are the largest the context
// allows and do not overlap.
inline std::vector<std::pair<int64_t, int64_t>> nar_chunk_ranges(
    int64_t frames,
    int64_t prefix_tokens,
    int64_t context,
    int64_t window,
    int64_t overlap) {
    int64_t chunk = std::min((context - prefix_tokens - 3) / 2, context);
    if (window > 0) {
        chunk = std::min(chunk, window);
    } else {
        overlap = 0;
    }
    if (frames < 1 || chunk < 1 || overlap < 0 || overlap >= chunk) {
        throw std::runtime_error("Yue2 NAR chunk leaves no acoustic context");
    }
    std::vector<std::pair<int64_t, int64_t>> out;
    for (int64_t begin = 0; begin < frames; begin += chunk - overlap) {
        const int64_t end = std::min(begin + chunk, frames);
        out.push_back({begin, end});
        if (end == frames) {
            break;
        }
    }
    return out;
}

// Writes one solved chunk into the full latent. Frames already rendered by the
// previous chunk are crossfaded linearly; both chunks solved the same slice of
// the global noise there. No gain or limiter is applied.
inline void blend_nar_chunk(
    std::vector<float> & out,
    const std::vector<float> & chunk,
    int64_t begin,
    int64_t end,
    int64_t rendered_end,
    int64_t latent_dim) {
    const int64_t shared = std::max<int64_t>(0, rendered_end - begin);
    for (int64_t t = begin; t < end; ++t) {
        const float mix = t < rendered_end
            ? static_cast<float>(t - begin + 1) / static_cast<float>(shared + 1)
            : 1.0F;
        for (int64_t c = 0; c < latent_dim; ++c) {
            const size_t dst = static_cast<size_t>(t * latent_dim + c);
            const size_t src = static_cast<size_t>((t - begin) * latent_dim + c);
            out[dst] = out[dst] * (1.0F - mix) + chunk[src] * mix;
        }
    }
}

} // namespace engine::models::yue2::detail
