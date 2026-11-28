#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::sanotts {

/** AArch64 NEON replacement for the host log-magnitude/phase iSTFT used by the
 *  sanoTTS runtime (n_fft 1024, hop 256 only).
 *
 *  Input is the decoder output, frames x (n_fft + 2): n_fft/2 + 1 log
 *  magnitudes followed by n_fft/2 + 1 phases per frame. Bins 0 and n_fft/2 are
 *  treated as zero, as the runtime already does. Returns the (frames - 1) * hop
 *  samples that the runtime keeps after trimming (n_fft / 2 per side), i.e. the
 *  same samples as HostLogMagnitudePhaseISTFT followed by the runtime's trim,
 *  before the DC blocker. Results match to float rounding, not bit-exactly:
 *  exp / sin / cos are polynomial approximations (about 1 ulp) and the inverse
 *  FFT is a radix-8x8x8 transform that handles four frames at once.
 *
 *  Compared with the framework path it avoids allocating and zeroing about
 *  6 MB of work buffers per call (the buffer is kept between calls).
 */
class SanoTtsNeonIstft {
public:
    static bool available(int64_t n_fft, int64_t hop_length);

    SanoTtsNeonIstft();
    ~SanoTtsNeonIstft();

    std::vector<float> compute(
        const std::vector<float> & spectrum,
        int64_t frames,
        const std::vector<float> & window,
        int threads);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::sanotts
