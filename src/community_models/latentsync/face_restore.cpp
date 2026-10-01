#include "engine/community_models/latentsync/face_restore.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace engine::models::latentsync {
namespace {

std::vector<float> erode_square(const std::vector<float> & image, int height, int width, int size,
    [[maybe_unused]] int threads) {
    if (size <= 0) throw std::runtime_error("LatentSync restoration mask is too small");
    auto input = image;
    std::vector<float> output(image.size());
    const int anchor = size / 2;
    for (int axis = 0; axis < 2; ++axis) {
        const int lines = axis ? width : height, length = axis ? height : width;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) if(threads > 1) schedule(static)
#endif
        for (int line = 0; line < lines; ++line) {
            std::deque<std::pair<int, float>> queue;
            for (int position = -anchor; position < length + size - 1 - anchor; ++position) {
                const float value = position >= 0 && position < length
                    ? input[axis ? position * width + line : line * width + position]
                    : std::numeric_limits<float>::infinity();
                while (!queue.empty() && queue.back().second >= value) queue.pop_back();
                queue.emplace_back(position, value);
                while (queue.front().first <= position - size) queue.pop_front();
                const int target = position - (size - 1 - anchor);
                if (target >= 0 && target < length) output[axis ? target * width + line : line * width + target] = queue.front().second;
            }
        }
        input.swap(output);
    }
    return input;
}

std::vector<float> gaussian_mask(const std::vector<float> & image, int height, int width, int radius,
    [[maybe_unused]] int threads) {
    const float sigma = 0.3F * (radius - 1) + 0.8F;
    std::vector<float> kernel(radius * 2 + 1);
    float total = 0;
    for (int i = -radius; i <= radius; ++i) {
        kernel[i + radius] = std::exp(-float(i * i) / (2 * sigma * sigma));
        total += kernel[i + radius];
    }
    for (auto & value : kernel) value /= total;
    auto input = image;
    std::vector<float> output(image.size());
    for (int axis = 0; axis < 2; ++axis) {
        const int length = axis ? height : width;
        if (radius >= length) throw std::runtime_error("LatentSync restoration blur exceeds frame dimensions");
        if (axis == 0) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) if(threads > 1) schedule(static)
#endif
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    float value = 0;
                    for (int i = -radius; i <= radius; ++i) {
                        int source = x + i;
                        if (source < 0) source = -source;
                        if (source >= width) source = 2 * width - 2 - source;
                        value += input[y * width + source] * kernel[i + radius];
                    }
                    output[y * width + x] = value;
                }
            }
        } else {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) if(threads > 1) schedule(static)
#endif
            for (int y = 0; y < height; ++y) {
                auto * destination = output.data() + static_cast<size_t>(y) * width;
                std::fill_n(destination, width, 0.0F);
                for (int i = -radius; i <= radius; ++i) {
                    int source = y + i;
                    if (source < 0) source = -source;
                    if (source >= height) source = 2 * height - 2 - source;
                    const auto * row = input.data() + static_cast<size_t>(source) * width;
                    const float weight = kernel[i + radius];
                    for (int x = 0; x < width; ++x) destination[x] += row[x] * weight;
                }
            }
        }
        input.swap(output);
    }
    return input;
}

}  // namespace

std::vector<float> latentsync_resize_bicubic(const std::vector<float> & image,
    int channels, int height, int width, int output_height, int output_width) {
    if (channels <= 0 || height <= 0 || width <= 0 || output_height <= 0 || output_width <= 0 ||
        image.size() != static_cast<size_t>(channels) * height * width) throw std::runtime_error("Invalid LatentSync bicubic resize dimensions");
    auto input = image;
    for (int axis = 0; axis < 2; ++axis) {
        const int source_size = axis ? height : width, target_size = axis ? output_height : output_width;
        const int other = axis ? output_width : height;
        const double scale = double(source_size) / target_size;
        const double filter_scale = std::max(1.0, scale);
        const double support = 2 * filter_scale;
        std::vector<float> output(static_cast<size_t>(channels) * other * target_size);
        for (int target = 0; target < target_size; ++target) {
            const double center = (target + 0.5) * scale;
            const int begin = std::max(0, int(center - support + 0.5));
            const int end = std::min(source_size, int(center + support + 0.5));
            std::vector<float> weights(end - begin);
            double sum = 0;
            for (int source = begin; source < end; ++source) {
                const double x = std::abs((source + 0.5 - center) / filter_scale);
                const double value = x < 1 ? ((1.5 * x - 2.5) * x * x + 1) :
                    (x < 2 ? ((-0.5 * x + 2.5) * x - 4) * x + 2 : 0);
                weights[source - begin] = float(value);
                sum += value;
            }
            for (auto & weight : weights) weight /= float(sum);
            for (int c = 0; c < channels; ++c) for (int j = 0; j < other; ++j) {
                float value = 0;
                for (int source = begin; source < end; ++source) {
                    const size_t index = axis ? (static_cast<size_t>(c) * height + source) * output_width + j
                                              : (static_cast<size_t>(c) * height + j) * width + source;
                    value += input[index] * weights[source - begin];
                }
                const size_t index = axis ? (static_cast<size_t>(c) * target_size + target) * other + j
                                          : (static_cast<size_t>(c) * other + j) * target_size + target;
                output[index] = value;
            }
        }
        input = std::move(output);
    }
    return input;
}

std::vector<float> latentsync_restore_face(const std::vector<float> & original,
    int height, int width, const std::vector<float> & face, const LatentSyncAffine & transform, int threads) {
    if (height <= 0 || width <= 0 || original.size() != static_cast<size_t>(height) * width * 3) {
        throw std::runtime_error("Invalid LatentSync restoration frame");
    }
    const auto resized = latentsync_resize_bicubic(face, 3, 512, 512, 560, 420);
    const float det = transform[0] * transform[4] - transform[1] * transform[3];
    if (!std::isfinite(det) || det == 0) throw std::runtime_error("Singular LatentSync restoration transform");
    LatentSyncAffine inverse{transform[4] / det, -transform[1] / det, 0, -transform[3] / det, transform[0] / det, 0};
    inverse[2] = -inverse[0] * transform[2] - inverse[1] * transform[5];
    inverse[5] = -inverse[3] * transform[2] - inverse[4] * transform[5];
    auto warped = latentsync_warp_affine(resized, 3, 560, 420, inverse, height, width, 127, threads);
    auto mask = latentsync_warp_affine(std::vector<float>(560 * 420, 1), 1, 560, 420, inverse, height, width, 0, threads);
    mask = erode_square(mask, height, width, 2, threads);
    const double area = std::accumulate(mask.begin(), mask.end(), 0.0);
    const int edge = int(std::sqrt(area)) / 20;
    const auto center = erode_square(mask, height, width, edge * 2, threads);
    const auto soft = gaussian_mask(center, height, width, edge, threads);
    const size_t pixels = static_cast<size_t>(height) * width;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) if(threads > 1) schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(warped.size()); ++i) {
        const size_t pixel = i % pixels;
        const float value = std::clamp(warped[i] / 2 + 0.5F, 0.0F, 1.0F) * 255;
        warped[i] = std::floor(std::clamp(soft[pixel] * mask[pixel] * value + (1 - soft[pixel]) * original[i], 0.0F, 255.0F));
    }
    return warped;
}

}  // namespace engine::models::latentsync
