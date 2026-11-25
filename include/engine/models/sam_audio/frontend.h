#pragma once

#include "engine/framework/vision/media/video.h"

#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <vector>

namespace engine::models::sam_audio {

using SourceVideo = engine::io::RgbVideo;

std::vector<size_t> select_video_frames(const SourceVideo & video, int64_t frames,
                                       int64_t hop, int sample_rate);

// Matches torchvision's antialiased bicubic RGB resize and [-1,1] normalization.
std::vector<float> prepare_video_frame(const uint8_t * rgb, int width, int height, int64_t stride);
std::vector<float> load_reference_image(const std::filesystem::path & path);

}  // namespace engine::models::sam_audio
