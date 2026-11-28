#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace engine::video {

struct RgbImageView {
    const uint8_t * data;
    int64_t width, height, stride;
};

struct RgbImage {
    int64_t width = 0, height = 0;
    std::vector<uint8_t> pixels;
    RgbImageView view() const { return {pixels.data(), width, height, width * 3}; }
};

struct ImageRegion {
    int64_t x, y, width, height;
};

RgbImage load_rgb_image(const std::filesystem::path & path);

// Half-pixel bilinear sampling with clamped borders and double arithmetic.
// The caller's conversion runs before the final float store, preserving its
// normalization operation order. Cropping precedes resize.
class BilinearRgbResizePlan {
public:
    BilinearRgbResizePlan(int64_t source_width, int64_t source_height, ImageRegion crop,
                          int64_t output_width, int64_t output_height);

    template <typename Convert>
    std::vector<float> planar_rgb(RgbImageView source, Convert && convert) const {
        if (!source.data || source.width != source_width_ || source.height != source_height_ ||
            source.stride < source.width * 3)
            throw std::invalid_argument("RGB resize input does not match its plan");
        const auto width = horizontal_.size(), height = vertical_.size();
        std::vector<float> output(3 * width * height);
        for (size_t y = 0; y < height; ++y) {
            const auto & row = vertical_[y];
            for (size_t x = 0; x < width; ++x) {
                const auto & col = horizontal_[x];
                for (int64_t c = 0; c < 3; ++c) {
                    const auto at = [&](int64_t yy, int64_t xx) -> double {
                        return static_cast<double>(source.data[yy * source.stride + xx * 3 + c]);
                    };
                    const double top = at(row.first, col.first) * (1.0 - col.weight) + at(row.first, col.second) * col.weight;
                    const double bottom = at(row.second, col.first) * (1.0 - col.weight) + at(row.second, col.second) * col.weight;
                    const double value = top * (1.0 - row.weight) + bottom * row.weight;
                    output[(static_cast<size_t>(c) * height + y) * width + x] = static_cast<float>(convert(value));
                }
            }
        }
        return output;
    }
private:
    struct Sample { int64_t first, second; double weight; };
    static std::vector<Sample> axis(int64_t begin, int64_t length, int64_t output);
    int64_t source_width_, source_height_;
    std::vector<Sample> horizontal_, vertical_;
};

// Pillow RGB bilinear: widened downsampling support, 22-bit coefficients and
// byte rounding after each separable pass. The output crop is applied after
// resize. Rounded vertical values are written directly to planar F32 [3,H,W]
// in [0,255], without materializing another interleaved RGB image.
class PillowBilinearRgbResizePlan {
public:
    PillowBilinearRgbResizePlan(int64_t source_width, int64_t source_height,
                                int64_t output_width, int64_t output_height);
    std::vector<float> planar_rgb(RgbImageView source, ImageRegion output_crop) const;
private:
    struct Taps { int64_t begin; std::vector<int32_t> weights; };
    static std::vector<Taps> axis(int64_t input, int64_t output);
    int64_t source_width_, source_height_, output_width_, output_height_;
    std::vector<Taps> horizontal_, vertical_;
};

} // namespace engine::video
