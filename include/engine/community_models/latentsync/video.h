#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace engine::models::latentsync {
struct LatentSyncVideo {
    int height = 0, width = 0;
    std::vector<std::vector<uint8_t>> frames;
};
LatentSyncVideo load_latentsync_video(const std::filesystem::path & path, double fps);
}  // namespace engine::models::latentsync
