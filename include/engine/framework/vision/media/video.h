#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

namespace engine::io {
struct RgbVideoFrame {
    double time = 0.0;
    int64_t stride = 0;
    // Owns padded RGB24 storage. Only width * 3 bytes per row are pixels.
    std::shared_ptr<const uint8_t> rgb;
};

struct RgbVideo {
    int64_t width = 0, height = 0;
    std::vector<RgbVideoFrame> frames;
};

// Called in presentation order on the calling thread. A frame can be retained by
// copying its shared owner; returning false stops decoding. Exceptions propagate
// only after any decoder worker has joined. Zero prefetch runs synchronously;
// positive prefetch bounds the queued frames (plus one producer/consumer frame).
using RgbVideoConsumer = std::function<bool(RgbVideoFrame, int64_t width, int64_t height)>;
void read_video_rgb(const std::filesystem::path & path, const RgbVideoConsumer & consume,
                    double max_duration_sec = 0.0, size_t prefetch_frames = 0);

// Decode native RGB24 frames and their presentation timestamps in-process.
// Requires FFmpeg 6+ shared libraries; never launches an executable. A positive
// duration limit retains frames through that timestamp; zero decodes all frames.
RgbVideo load_video_rgb(const std::filesystem::path & path, double max_duration_sec = 0.0);
} // namespace engine::io
