#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace engine::video {

struct RgbVideoBuffer {
    int64_t width = 0, height = 0, frames = 0;
    double fps = 0.0;
    std::vector<std::byte> rgb24;
};

// Contiguous planar [C,T,H,W] storage used by spatial video operations.
struct FloatVideo {
    int64_t channels = 3;
    int64_t frames = 0;
    int64_t height = 0;
    int64_t width = 0;
    std::vector<float> values;

    float & at(int64_t c, int64_t t, int64_t y, int64_t x) {
        return values[static_cast<size_t>(((c * frames + t) * height + y) * width + x)];
    }
    const float & at(int64_t c, int64_t t, int64_t y, int64_t x) const {
        return values[static_cast<size_t>(((c * frames + t) * height + y) * width + x)];
    }
};

struct SpatialTiles {
    std::vector<int64_t> starts;
    // One overlap between each pair of adjacent tiles; empty for one tile.
    std::vector<int64_t> overlaps;
};

SpatialTiles plan_spatial_tiles(int64_t input_length, int64_t tile_size,
                                int64_t minimum_overlap, int64_t spatial_ratio);
void blend_height_from_previous(const FloatVideo & previous, FloatVideo & current, int64_t extent);
void blend_width_from_previous(const FloatVideo & previous, FloatVideo & current, int64_t extent);

// Borrowed contiguous planar RGB. The caller retains the input for the call.
struct PlanarRgbVideoView {
    const float * data;
    int64_t frames, height, width;
};

// Only layout/slicing is shared. The callable supplies the exact scalar mapping
// (including channel normalization, operation order and rounding) to a byte.
// It must be safe for concurrent calls when threads > 1.
template <typename Convert>
std::vector<std::byte> pack_rgb24(PlanarRgbVideoView input,
                                 int64_t first_frame, int64_t frame_count,
                                 Convert && convert, int threads = 1) {
    if (!input.data || input.frames <= 0 || input.height <= 0 || input.width <= 0 ||
        first_frame < 0 || frame_count <= 0 || first_frame > input.frames ||
        frame_count > input.frames - first_frame) {
        throw std::invalid_argument("Invalid planar RGB video slice");
    }
    const auto pixels = static_cast<size_t>(input.height * input.width);
    std::vector<std::byte> output(static_cast<size_t>(frame_count) * pixels * 3);
#if defined(_OPENMP)
#pragma omp parallel for num_threads(threads) if(threads > 1) schedule(static)
#else
    (void) threads;
#endif
    for (int64_t t = 0; t < frame_count; ++t) {
        for (size_t p = 0; p < pixels; ++p) {
            for (int64_t c = 0; c < 3; ++c) {
                const auto index = (static_cast<size_t>(c * input.frames + first_frame + t) * pixels) + p;
                output[(static_cast<size_t>(t) * pixels + p) * 3 + static_cast<size_t>(c)] =
                    static_cast<std::byte>(convert(input.data[index], c));
            }
        }
    }
    return output;
}

} // namespace engine::video
