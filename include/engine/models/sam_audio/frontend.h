#pragma once

#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <vector>

namespace engine::models::sam_audio {

struct DecodedFrame {
    double time = 0.0;
    std::vector<uint8_t> rgb;
};

struct SourceVideo {
    int64_t width = 0;
    int64_t height = 0;
    std::vector<DecodedFrame> frames;
};

SourceVideo load_video_frames(const std::filesystem::path & path);
std::vector<size_t> select_video_frames(const SourceVideo & video, int64_t frames,
                                       int64_t hop, int sample_rate);

// Matches torchvision's antialiased bicubic RGB resize and [-1,1] normalization.
std::vector<float> prepare_video_frame(const std::vector<uint8_t> & rgb, int width, int height);
std::vector<float> load_reference_image(const std::filesystem::path & path);

}  // namespace engine::models::sam_audio
