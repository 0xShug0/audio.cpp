#include "engine/framework/vision/media/image.h"

#include <algorithm>
#include <cmath>
#include <memory>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "../../../../external/ggml/examples/stb_image.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace engine::video {
namespace {
void validate_region(ImageRegion region, int64_t width, int64_t height) {
    if (width <= 0 || height <= 0 || region.x < 0 || region.y < 0 ||
        region.width <= 0 || region.height <= 0 || region.x > width || region.y > height ||
        region.width > width - region.x || region.height > height - region.y)
        throw std::invalid_argument("Invalid RGB image region");
}
}

RgbImage load_rgb_image(const std::filesystem::path & path) {
    int width = 0, height = 0, channels = 0;
    std::unique_ptr<unsigned char, decltype(&stbi_image_free)> data(
        stbi_load(path.string().c_str(), &width, &height, &channels, 3), stbi_image_free);
    if (!data || width <= 0 || height <= 0)
        throw std::runtime_error("Cannot load RGB image: " + path.string());
    return {width, height, {data.get(), data.get() + static_cast<size_t>(width) * height * 3}};
}

std::vector<BilinearRgbResizePlan::Sample> BilinearRgbResizePlan::axis(int64_t begin, int64_t length, int64_t output) {
    std::vector<Sample> samples;
    samples.reserve(static_cast<size_t>(output));
    for (int64_t i = 0; i < output; ++i) {
        const double position = (static_cast<double>(i) + 0.5) * static_cast<double>(length) /
            static_cast<double>(output) - 0.5;
        const int64_t first = static_cast<int64_t>(std::floor(position));
        samples.push_back({begin + std::clamp<int64_t>(first, 0, length - 1),
                           begin + std::clamp<int64_t>(first + 1, 0, length - 1),
                           position - static_cast<double>(first)});
    }
    return samples;
}

BilinearRgbResizePlan::BilinearRgbResizePlan(int64_t source_width, int64_t source_height, ImageRegion crop,
                                             int64_t output_width, int64_t output_height)
    : source_width_(source_width), source_height_(source_height) {
    validate_region(crop, source_width, source_height);
    if (output_width <= 0 || output_height <= 0) throw std::invalid_argument("Invalid RGB resize output size");
    horizontal_ = axis(crop.x, crop.width, output_width);
    vertical_ = axis(crop.y, crop.height, output_height);
}

std::vector<PillowBilinearRgbResizePlan::Taps> PillowBilinearRgbResizePlan::axis(int64_t input, int64_t output) {
    constexpr int precision = 1 << 22;
    const double scale = static_cast<double>(input) / output;
    const double support = std::max(1., scale);
    std::vector<Taps> result;
    result.reserve(static_cast<size_t>(output));
    for (int64_t i = 0; i < output; ++i) {
        const double center = (i + .5) * scale;
        const int64_t begin = std::max<int64_t>(0, static_cast<int64_t>(center - support + .5));
        const int64_t end = std::min<int64_t>(input, static_cast<int64_t>(center + support + .5));
        std::vector<double> values;
        double sum = 0;
        for (int64_t j = begin; j < end; ++j) {
            const double value = std::max(0., 1 - std::abs((j + .5 - center) / support));
            values.push_back(value);
            sum += value;
        }
        Taps taps{begin, {}};
        for (double value : values) taps.weights.push_back(static_cast<int32_t>(value / sum * precision + .5));
        result.push_back(std::move(taps));
    }
    return result;
}

PillowBilinearRgbResizePlan::PillowBilinearRgbResizePlan(int64_t source_width, int64_t source_height,
                                                       int64_t output_width, int64_t output_height)
    : source_width_(source_width), source_height_(source_height),
      output_width_(output_width), output_height_(output_height) {
    if (source_width <= 0 || source_height <= 0 || output_width <= 0 || output_height <= 0)
        throw std::invalid_argument("Invalid Pillow RGB resize dimensions");
    horizontal_ = axis(source_width, output_width);
    vertical_ = axis(source_height, output_height);
}

std::vector<float> PillowBilinearRgbResizePlan::planar_rgb(RgbImageView source, ImageRegion crop) const {
    validate_region(crop, output_width_, output_height_);
    if (!source.data || source.width != source_width_ || source.height != source_height_ || source.stride < source.width * 3)
        throw std::invalid_argument("Pillow RGB resize input does not match its plan");
    std::vector<uint8_t> intermediate(static_cast<size_t>(source_height_ * output_width_ * 3));
    for (int64_t y = 0; y < source_height_; ++y)
        for (int64_t x = 0; x < output_width_; ++x)
            for (int64_t c = 0; c < 3; ++c) {
                const auto & taps = horizontal_[static_cast<size_t>(x)];
                int64_t sum = 1 << 21;
                for (size_t tap = 0; tap < taps.weights.size(); ++tap)
                    sum += static_cast<int64_t>(source.data[y * source.stride + (taps.begin + tap) * 3 + c]) * taps.weights[tap];
                intermediate[static_cast<size_t>((y * output_width_ + x) * 3 + c)] =
                    static_cast<uint8_t>(std::clamp<int64_t>(sum >> 22, 0, 255));
            }
    std::vector<float> output(static_cast<size_t>(crop.width * crop.height * 3));
    for (int64_t y = 0; y < crop.height; ++y)
        for (int64_t x = 0; x < crop.width; ++x)
            for (int64_t c = 0; c < 3; ++c) {
                const auto & taps = vertical_[static_cast<size_t>(y + crop.y)];
                int64_t sum = 1 << 21;
                for (size_t tap = 0; tap < taps.weights.size(); ++tap)
                    sum += static_cast<int64_t>(intermediate[static_cast<size_t>(((taps.begin + tap) * output_width_ + x + crop.x) * 3 + c)]) * taps.weights[tap];
                output[static_cast<size_t>((c * crop.height + y) * crop.width + x)] =
                    static_cast<float>(std::clamp<int64_t>(sum >> 22, 0, 255));
            }
    return output;
}
} // namespace engine::video
