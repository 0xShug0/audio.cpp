#pragma once

#include <filesystem>
#include <vector>

namespace engine::audio {

void write_pcm16_wav(
    const std::filesystem::path & path,
    int sample_rate,
    int channel_count,
    const std::vector<float> & audio);

}  // namespace engine::audio
