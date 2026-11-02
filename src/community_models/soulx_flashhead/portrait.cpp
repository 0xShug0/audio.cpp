#include "engine/community_models/soulx_flashhead/portrait.h"
#include "engine/framework/core/backend.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "../../../external/ggml/examples/stb_image.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace engine::community_models::soulx_flashhead {
namespace {
struct Taps {
    int begin;
    std::vector<int32_t> weights;
};

// Match Pillow's RGB bilinear resize: widened downsampling support and
// 22-bit coefficients, with byte rounding after each separable pass.
// https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c
std::vector<Taps> filter(int input, int output) {
    constexpr int precision = 1 << 22;
    const double scale = static_cast<double>(input) / output;
    const double support = std::max(1., scale);
    std::vector<Taps> result;
    result.reserve(static_cast<size_t>(output));
    for (int i = 0; i < output; ++i) {
        const double center = (i + .5) * scale;
        const int begin = std::max(0, static_cast<int>(center - support + .5));
        const int end = std::min(input, static_cast<int>(center + support + .5));
        std::vector<double> values;
        double sum = 0;
        for (int j = begin; j < end; ++j) {
            const double value = std::max(0., 1 - std::abs((j + .5 - center) / support));
            values.push_back(value);
            sum += value;
        }
        Taps taps{begin, {}};
        for (double value : values)
            taps.weights.push_back(static_cast<int32_t>(value / sum * precision + .5));
        result.push_back(std::move(taps));
    }
    return result;
}
} // namespace

std::vector<float> load_portrait(const std::filesystem::path & path, int64_t height, int64_t width) {
    int sw = 0, sh = 0, channels = 0;
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(
        stbi_load(path.string().c_str(), &sw, &sh, &channels, 3), stbi_image_free);
    if (!pixels)
        throw std::runtime_error("Cannot read FlashHead portrait: " + path.string());
    const double scale = std::max(static_cast<double>(height) / sh, static_cast<double>(width) / sw);
    const int rh = static_cast<int>(std::ceil(sh * scale)), rw = static_cast<int>(std::ceil(sw * scale));
    const int64_t top = static_cast<int64_t>(std::nearbyint((rh - height) / 2.));
    const int64_t left = static_cast<int64_t>(std::nearbyint((rw - width) / 2.));
    const auto horizontal = filter(sw, rw), vertical = filter(sh, rh);
    std::vector<uint8_t> intermediate(static_cast<size_t>(sh) * rw * 3);
    for (int y = 0; y < sh; ++y)
        for (int x = 0; x < rw; ++x)
            for (int c = 0; c < 3; ++c) {
                const auto & taps = horizontal[static_cast<size_t>(x)];
                int64_t sum = 1 << 21;
                for (size_t tap = 0; tap < taps.weights.size(); ++tap)
                    sum +=
                        static_cast<int64_t>(pixels.get()[(static_cast<size_t>(y) * sw + taps.begin + tap) * 3 + c]) *
                        taps.weights[tap];
                intermediate[(static_cast<size_t>(y) * rw + x) * 3 + c] =
                    static_cast<uint8_t>(std::clamp<int64_t>(sum >> 22, 0, 255));
            }
    std::vector<float> result(static_cast<size_t>(3 * height * width));
    for (int64_t y = 0; y < height; ++y)
        for (int64_t x = 0; x < width; ++x)
            for (int c = 0; c < 3; ++c) {
                const auto & taps = vertical[static_cast<size_t>(y + top)];
                int64_t sum = 1 << 21;
                for (size_t tap = 0; tap < taps.weights.size(); ++tap)
                    sum += static_cast<int64_t>(intermediate[((taps.begin + tap) * rw + x + left) * 3 + c]) *
                           taps.weights[tap];
                result[static_cast<size_t>((c * height + y) * width + x)] =
                    static_cast<float>(std::clamp<int64_t>(sum >> 22, 0, 255));
            }
    core::round_f32_to_bf16_in_place(result);
    for (auto & value : result)
        value /= 255;
    core::round_f32_to_bf16_in_place(result);
    for (auto & value : result)
        value -= .5F;
    core::round_f32_to_bf16_in_place(result);
    for (auto & value : result)
        value *= 2;
    core::round_f32_to_bf16_in_place(result);
    return result;
}
} // namespace engine::community_models::soulx_flashhead
