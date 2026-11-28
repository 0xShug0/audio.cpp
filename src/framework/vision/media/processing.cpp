#include "engine/framework/vision/media/processing.h"

#include <algorithm>

namespace engine::video {

SpatialTiles plan_spatial_tiles(int64_t input_length, int64_t tile_size,
                                int64_t minimum_overlap, int64_t spatial_ratio) {
    if (input_length <= 0 || tile_size <= 0 || minimum_overlap < 0 ||
        spatial_ratio <= 0 || tile_size <= minimum_overlap) {
        throw std::invalid_argument("Invalid video spatial tile configuration");
    }
    if (tile_size >= input_length) return {{0}, {}};
    int64_t count = (input_length + tile_size - minimum_overlap - 1) / (tile_size - minimum_overlap);
    while (tile_size * count - minimum_overlap * (count - 1) < input_length) ++count;
    SpatialTiles result{{0}, std::vector<int64_t>(static_cast<size_t>(count - 1), minimum_overlap)};
    const int64_t remaining = tile_size * count - minimum_overlap * (count - 1) - input_length;
    for (int64_t i = 0; i < remaining / spatial_ratio; ++i) {
        result.overlaps[static_cast<size_t>(i % (count - 1))] += spatial_ratio;
    }
    for (int64_t i = 0; i + 1 < count; ++i) {
        result.starts.push_back(result.starts.back() + tile_size - result.overlaps[static_cast<size_t>(i)]);
    }
    return result;
}

void blend_height_from_previous(const FloatVideo & previous, FloatVideo & current, int64_t extent) {
    extent = std::min<int64_t>({extent, previous.height, current.height});
    for (int64_t y = 0; y < extent; ++y) {
        const float weight = static_cast<float>(y) / static_cast<float>(extent);
        const int64_t previous_y = previous.height - extent + y;
        for (int64_t c = 0; c < current.channels; ++c)
            for (int64_t t = 0; t < current.frames; ++t)
                for (int64_t x = 0; x < current.width; ++x)
                    current.at(c, t, y, x) =
                        previous.at(c, t, previous_y, x) * (1.0F - weight) + current.at(c, t, y, x) * weight;
    }
}

void blend_width_from_previous(const FloatVideo & previous, FloatVideo & current, int64_t extent) {
    extent = std::min<int64_t>({extent, previous.width, current.width});
    for (int64_t x = 0; x < extent; ++x) {
        const float weight = static_cast<float>(x) / static_cast<float>(extent);
        const int64_t previous_x = previous.width - extent + x;
        for (int64_t c = 0; c < current.channels; ++c)
            for (int64_t t = 0; t < current.frames; ++t)
                for (int64_t y = 0; y < current.height; ++y)
                    current.at(c, t, y, x) =
                        previous.at(c, t, y, previous_x) * (1.0F - weight) + current.at(c, t, y, x) * weight;
    }
}

} // namespace engine::video
