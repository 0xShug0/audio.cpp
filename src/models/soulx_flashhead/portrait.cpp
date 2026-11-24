#include "engine/models/soulx_flashhead/portrait.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/vision/media/image.h"
#include <algorithm>
#include <cmath>

namespace engine::models::soulx_flashhead {
std::vector<float> load_portrait(const std::filesystem::path & path, int64_t height, int64_t width) {
    const auto source = video::load_rgb_image(path);
    const double scale = std::max(static_cast<double>(height) / source.height, static_cast<double>(width) / source.width);
    const int rh = static_cast<int>(std::ceil(source.height * scale)), rw = static_cast<int>(std::ceil(source.width * scale));
    const int64_t top = static_cast<int64_t>(std::nearbyint((rh - height) / 2.));
    const int64_t left = static_cast<int64_t>(std::nearbyint((rw - width) / 2.));
    auto result = video::PillowBilinearRgbResizePlan(source.width, source.height, rw, rh)
        .planar_rgb(source.view(), {left, top, width, height});
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
} // namespace engine::models::soulx_flashhead
