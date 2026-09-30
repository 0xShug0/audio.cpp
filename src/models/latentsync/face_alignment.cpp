#include "engine/models/latentsync/face_alignment.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace engine::models::latentsync {

LatentSyncFacePoints latentsync_alignment_points(const std::vector<float> & landmarks) {
    if (landmarks.size() != 212) throw std::runtime_error("LatentSync alignment requires 106 landmarks");
    constexpr int left[] = {43, 48, 49, 51, 50};
    constexpr int right[] = {101, 102, 103, 104, 105};
    constexpr int nose[] = {74, 77, 83, 86};
    LatentSyncFacePoints points{};
    for (int d = 0; d < 2; ++d) {
        for (const int i : left) points[0][d] += std::nearbyint(landmarks[i * 2 + d]);
        for (const int i : right) points[1][d] += std::nearbyint(landmarks[i * 2 + d]);
        for (const int i : nose) points[2][d] += std::nearbyint(landmarks[i * 2 + d]);
        points[0][d] = std::nearbyint(points[0][d] / 5);
        points[1][d] = std::nearbyint(points[1][d] / 5);
        points[2][d] = std::nearbyint(points[2][d] / 4);
    }
    return points;
}

LatentSyncFaceAlignment::LatentSyncFaceAlignment(int resolution) {
    if (resolution <= 0) throw std::runtime_error("LatentSync face resolution must be positive");
    const double ratio = resolution / 256.0 * 2.8;
    target_ = {{{float(17 * ratio), float(20 * ratio)},
                {float(58 * ratio), float(20 * ratio)},
                {float(37.5 * ratio), float(40 * ratio)}}};
    width_ = int(75 * ratio);
    height_ = int(100 * ratio);
}

LatentSyncAffine LatentSyncFaceAlignment::transform(const LatentSyncFacePoints & points, bool smooth) {
    std::array<float, 2> center{}, target_center{};
    for (int i = 0; i < 3; ++i) {
        for (int d = 0; d < 2; ++d) {
            center[d] += points[i][d] / 3;
            target_center[d] += target_[i][d] / 3;
        }
    }
    auto source = points;
    auto target = target_;
    float source_variance = 0, target_variance = 0;
    for (int i = 0; i < 3; ++i) {
        for (int d = 0; d < 2; ++d) {
            source[i][d] -= center[d];
            target[i][d] -= target_center[d];
            source_variance += source[i][d] * source[i][d];
            target_variance += target[i][d] * target[i][d];
        }
    }
    const float s1 = std::sqrt(source_variance / 5);
    const float s2 = std::sqrt(target_variance / 5);
    if (!(s1 > 0)) throw std::runtime_error("Degenerate LatentSync face alignment points");
    float dot = 0, cross = 0;
    for (int i = 0; i < 3; ++i) {
        for (int d = 0; d < 2; ++d) {
            source[i][d] /= s1;
            target[i][d] /= s2;
        }
        dot += source[i][0] * target[i][0] + source[i][1] * target[i][1];
        cross += source[i][0] * target[i][1] - source[i][1] * target[i][0];
    }
    // Proper 2D Procrustes rotation, equivalent to the reference's determinant-corrected SVD.
    const float norm = std::hypot(dot, cross);
    if (!(norm > 0)) throw std::runtime_error("Indeterminate LatentSync face rotation");
    const float a = s2 / s1 * dot / norm;
    const float b = s2 / s1 * cross / norm;
    LatentSyncAffine matrix{a, -b, target_center[0] - a * center[0] + b * center[1],
                           b, a, target_center[1] - b * center[0] - a * center[1]};
    if (smooth) {
        std::array<float, 2> bias;
        for (int d = 0; d < 2; ++d) {
            bias[d] = target[2][d] - source[2][d];
            if (previous_bias_) bias[d] = (*previous_bias_)[d] * 0.2F + bias[d] * 0.8F;
            matrix[d * 3 + 2] += bias[d];
        }
        previous_bias_ = bias;
    }
    return matrix;
}

void LatentSyncFaceAlignment::reset() { previous_bias_.reset(); }

std::vector<float> latentsync_warp_affine(const std::vector<float> & image,
    int channels, int height, int width, const LatentSyncAffine & matrix,
    int output_height, int output_width, float fill) {
    if (channels <= 0 || height <= 0 || width <= 0 || output_height <= 0 || output_width <= 0 ||
        image.size() != static_cast<size_t>(channels) * height * width) {
        throw std::runtime_error("Invalid LatentSync affine image dimensions");
    }
    const double determinant = double(matrix[0]) * matrix[4] - double(matrix[1]) * matrix[3];
    if (!std::isfinite(determinant) || determinant == 0) throw std::runtime_error("Singular LatentSync affine transform");
    const double a = matrix[4] / determinant;
    const double b = -matrix[1] / determinant;
    const double c = -matrix[3] / determinant;
    const double d = matrix[0] / determinant;
    std::vector<float> output(static_cast<size_t>(channels) * output_height * output_width);
    for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            const double dx = x - matrix[2], dy = y - matrix[5];
            const double sx = a * dx + b * dy, sy = c * dx + d * dy;
            const int x0 = static_cast<int>(std::floor(sx)), y0 = static_cast<int>(std::floor(sy));
            const float wx = float(sx - x0), wy = float(sy - y0);
            for (int channel = 0; channel < channels; ++channel) {
                float value = 0;
                for (int yy = 0; yy < 2; ++yy) {
                    for (int xx = 0; xx < 2; ++xx) {
                        const int ix = x0 + xx, iy = y0 + yy;
                        const float pixel = ix >= 0 && ix < width && iy >= 0 && iy < height
                            ? image[(static_cast<size_t>(channel) * height + iy) * width + ix] : fill;
                        value += pixel * (xx ? wx : 1 - wx) * (yy ? wy : 1 - wy);
                    }
                }
                output[(static_cast<size_t>(channel) * output_height + y) * output_width + x] = value;
            }
        }
    }
    return output;
}

std::vector<float> latentsync_resize_lanczos4(const std::vector<float> & image,
    int channels, int height, int width, int output_height, int output_width) {
    if (channels <= 0 || height <= 0 || width <= 0 || output_height <= 0 || output_width <= 0 ||
        image.size() != static_cast<size_t>(channels) * height * width) {
        throw std::runtime_error("Invalid LatentSync resize dimensions");
    }
    struct Taps { int start; std::array<float, 8> weights; };
    std::array<std::vector<Taps>, 2> axes;
    constexpr double pi = 3.14159265358979323846;
    for (int axis = 0; axis < 2; ++axis) {
        const int source_size = axis ? height : width;
        const int target_size = axis ? output_height : output_width;
        axes[axis].resize(target_size);
        for (int i = 0; i < target_size; ++i) {
            const double position = (i + 0.5) * source_size / target_size - 0.5;
            auto & taps = axes[axis][i];
            taps.start = int(std::floor(position)) - 3;
            double sum = 0;
            for (int j = 0; j < 8; ++j) {
                const double distance = position - (taps.start + j);
                const double weight = std::abs(distance) < 1e-12 ? 1 :
                    std::sin(pi * distance) * std::sin(pi * distance / 4) / (pi * pi * distance * distance / 4);
                taps.weights[j] = float(weight);
                sum += weight;
            }
            for (auto & weight : taps.weights) weight /= float(sum);
        }
    }
    std::vector<float> horizontal(static_cast<size_t>(channels) * height * output_width);
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                float value = 0;
                const auto & taps = axes[0][x];
                for (int j = 0; j < 8; ++j) {
                    value += image[(static_cast<size_t>(c) * height + y) * width + std::clamp(taps.start + j, 0, width - 1)] * taps.weights[j];
                }
                horizontal[(static_cast<size_t>(c) * height + y) * output_width + x] = value;
            }
        }
    }
    std::vector<float> output(static_cast<size_t>(channels) * output_height * output_width);
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < output_height; ++y) {
            const auto & taps = axes[1][y];
            for (int x = 0; x < output_width; ++x) {
                float value = 0;
                for (int j = 0; j < 8; ++j) {
                    value += horizontal[(static_cast<size_t>(c) * height + std::clamp(taps.start + j, 0, height - 1)) * output_width + x] * taps.weights[j];
                }
                output[(static_cast<size_t>(c) * output_height + y) * output_width + x] = value;
            }
        }
    }
    return output;
}

}  // namespace engine::models::latentsync
