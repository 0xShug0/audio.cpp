#include "engine/framework/vision/media/processing.h"
#include "engine/framework/vision/media/image.h"
#include "engine/framework/debug/profiler.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}
}

int main(int, char **) {
    engine::debug::configure_logging({true});
    const auto started = std::chrono::steady_clock::now();
    using namespace engine::video;
    const auto one = plan_spatial_tiles(128, 192, 64, 8);
    require(one.starts == std::vector<int64_t>{0} && one.overlaps.empty(), "single tile");
    const auto tiles = plan_spatial_tiles(512, 192, 64, 8);
    require(tiles.starts == std::vector<int64_t>({0, 104, 208, 320}), "tile starts");
    require(tiles.overlaps == std::vector<int64_t>({88, 88, 80}), "tile overlaps");

    FloatVideo previous{1, 1, 4, 4, std::vector<float>(16, 8.0F)};
    FloatVideo current{1, 1, 4, 4, std::vector<float>(16, 0.0F)};
    blend_height_from_previous(previous, current, 4);
    for (int64_t y = 0; y < 4; ++y)
        for (int64_t x = 0; x < 4; ++x)
            require(current.at(0, 0, y, x) == 8.0F - 2.0F * static_cast<float>(y), "height ramp");
    current.values.assign(16, 0.0F);
    blend_width_from_previous(previous, current, 4);
    for (int64_t y = 0; y < 4; ++y)
        for (int64_t x = 0; x < 4; ++x)
            require(current.at(0, 0, y, x) == 8.0F - 2.0F * static_cast<float>(x), "width ramp");

    const std::vector<float> planar{0,1,2,3,4,5, 10,11,12,13,14,15, 20,21,22,23,24,25};
    const std::vector<std::byte> expected{std::byte{2},std::byte{12},std::byte{22},
        std::byte{3},std::byte{13},std::byte{23},std::byte{4},std::byte{14},std::byte{24},
        std::byte{5},std::byte{15},std::byte{25}};
    for (int threads : {1, 8}) {
        const auto packed = pack_rgb24({planar.data(), 3, 1, 2}, 1, 2,
            [](float value, int64_t) { return static_cast<uint8_t>(value); }, threads);
        require(packed == expected, "channel ordering and frame slicing");
    }
    // Analytic two-dimensional ramp, including padded RGB rows and border
    // clamping. Its center is the mean of the four corners in every channel.
    const std::vector<uint8_t> ramp{0,10,20, 20,30,40, 255,255,
                                   40,50,60, 60,70,80, 255,255};
    const auto resized = BilinearRgbResizePlan(2, 2, {0,0,2,2}, 3, 3)
        .planar_rgb({ramp.data(),2,2,8}, [](double value) { return value; });
    for (size_t c = 0; c < 3; ++c) {
        require(resized[c * 9] == 10.0F * c, "bilinear first clamped corner");
        require(resized[c * 9 + 4] == 30.0F + 10.0F * c, "bilinear center mean");
        require(resized[c * 9 + 8] == 60.0F + 10.0F * c, "bilinear last clamped corner");
    }
    // Independent Pillow 12.1.1 RGB BILINEAR reference for arange(36)*7,
    // reshaped to 3x4x3, then transposed to CHW F32. These fixed values do not
    // depend on another C++ implementation or a moving branch baseline.
    RgbImage image{4,3,{}};
    for (int i = 0; i < 36; ++i) image.pixels.push_back(static_cast<uint8_t>(i * 7));
    const auto down = PillowBilinearRgbResizePlan(4,3,2,2).planar_rgb(image.view(), {0,0,2,2});
    require(down == std::vector<float>({47,80,152,185, 54,87,159,192, 61,94,166,199}), "Pillow downsampling reference");
    const auto crop = PillowBilinearRgbResizePlan(4,3,7,5).planar_rgb(image.view(), {2,1,3,3});
    require(crop == std::vector<float>({54,66,77,104,116,127,154,166,177,
        61,73,84,111,123,134,161,173,184, 68,80,91,118,130,141,168,180,191}), "Pillow resize/crop reference");
    std::cout << "video_processing_test passed\n";
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
}
