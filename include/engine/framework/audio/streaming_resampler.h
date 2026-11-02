#pragma once

#include <memory>
#include <vector>

namespace engine::audio {

// Stateful mono F32 resampling. Flush once at end; never resample chunks independently.
class StreamingSoxrResampler {
public:
    StreamingSoxrResampler(int source_rate, int target_rate);
    ~StreamingSoxrResampler();
    std::vector<float> process(const std::vector<float> & samples, bool final = false);
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::audio
