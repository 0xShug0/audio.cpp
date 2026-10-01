#pragma once

#include <array>
#include <optional>
#include <vector>

namespace engine::models::latentsync {

using LatentSyncFacePoints = std::array<std::array<float, 2>, 3>;
using LatentSyncAffine = std::array<float, 6>;

LatentSyncFacePoints latentsync_alignment_points(const std::vector<float> & landmarks);

// Planar image layout; the matrix maps source pixels to destination pixels.
std::vector<float> latentsync_warp_affine(const std::vector<float> & image,
    int channels, int height, int width, const LatentSyncAffine & matrix,
    int output_height, int output_width, float fill, int threads);

std::vector<float> latentsync_resize_lanczos4(const std::vector<float> & image,
    int channels, int height, int width, int output_height, int output_width, int threads);

class LatentSyncFaceAlignment {
public:
    explicit LatentSyncFaceAlignment(int resolution = 512);
    LatentSyncAffine transform(const LatentSyncFacePoints & points, bool smooth = true);
    void reset();
    int crop_width() const { return width_; }
    int crop_height() const { return height_; }

private:
    LatentSyncFacePoints target_;
    int width_, height_;
    std::optional<std::array<float, 2>> previous_bias_;
};

}  // namespace engine::models::latentsync
