#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace engine::audio {

struct AudioBuffer {
    int sample_rate = 0;
    int channel_count = 1;
    std::vector<float> samples;
};

class IAudioSink {
public:
    virtual ~IAudioSink() = default;
    virtual std::string family() const = 0;
    virtual void write(const std::filesystem::path & path, const AudioBuffer & audio) const = 0;
};

class WavPcm16Sink final : public IAudioSink {
public:
    std::string family() const override;
    void write(const std::filesystem::path & path, const AudioBuffer & audio) const override;
};

}  // namespace engine::audio
