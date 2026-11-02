#pragma once

#include <memory>
#include <vector>

namespace minitts::app {

// Pitch-preserving mono speech tempo. State spans chunks; flush once at stream end.
// Speed 1 is an exact passthrough and does not load libsonic.
class StreamingSpeechTempo {
public:
    StreamingSpeechTempo(int sample_rate, float speed);
    ~StreamingSpeechTempo();
    std::vector<float> process(const std::vector<float> & samples, bool final = false);
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace minitts::app
