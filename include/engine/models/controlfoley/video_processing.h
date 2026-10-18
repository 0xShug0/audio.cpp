#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace engine::models::controlfoley::video {

struct DecodedFrame {
    double time = 0.0;
    std::vector<uint8_t> rgb;
};

struct SourceVideo {
    int64_t width = 0;
    int64_t height = 0;
    std::vector<DecodedFrame> frames;
};

SourceVideo load_video_frames(
    const std::filesystem::path & path,
    double max_duration_sec = 0.0);

class ResizedFrameCache {
public:
    ResizedFrameCache(
        const SourceVideo & video,
        int64_t size,
        bool center_crop);

    std::vector<float> extract(
        float duration_sec,
        int64_t fps,
        const std::array<float, 3> & mean,
        const std::array<float, 3> & std,
        std::string_view trace_name = {});

private:
    const SourceVideo * video_;
    int64_t size_;
    bool center_crop_;
    std::unordered_map<int64_t, std::vector<uint8_t>> cache_;
};

std::vector<float> extract_video_frames(
    const SourceVideo & video,
    float duration_sec,
    int64_t fps,
    int64_t size,
    bool center_crop,
    const std::array<float, 3> & mean,
    const std::array<float, 3> & std);

}  // namespace engine::models::controlfoley::video
