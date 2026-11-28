#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::mulacover {

struct ChordFrontendOutput {
    // Row-major [frames,252], abs(hybrid_cqt)[18:270].
    std::vector<float> cqt;
    int64_t frames = 0;
    double tuning = 0;  // Fractions of a 36-bin octave bin.
    double bpm = 0;  // Explicit BPM or source-rate beat_track estimate.
};

// Session-owned, serial frontend. Retains FFT plans, sparse filter banks for
// the discrete automatic-tuning estimates, octave responses, tempo mel filters,
// autocorrelation/beat-tracker buffers, and scratch buffers.
class ChordFrontend {
public:
    ChordFrontend();
    ~ChordFrontend();
    ChordFrontend(const ChordFrontend &) = delete;
    ChordFrontend & operator=(const ChordFrontend &) = delete;

    // Decoded interleaved F32 PCM. Downmix before soxr_hq resampling to 22050 Hz.
    // bpm=0 estimates with librosa beat_track defaults at the SOURCE sample rate.
    // A positive explicit BPM bypasses estimation and is preserved unchanged.
    // The returned reference remains valid until the next forward call/destruction.
    const ChordFrontendOutput & forward(const std::vector<float> & samples,
                                        int sample_rate, int channels, double bpm = 0);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::mulacover
