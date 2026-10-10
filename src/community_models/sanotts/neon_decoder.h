#pragma once

#include "engine/community_models/sanotts/assets.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::sanotts {

/** Hand-written ARM NEON version of the nano-lineage decoder (frame-stage
 *  acoustic blocks -> mel projection -> noise-fed ConvNeXt-1D -> spectrum
 *  head), for CPU sessions that ask for it with sanotts.cpu_decoder=neon.
 *
 *  Same inputs and output as the GGML decoder graph in runtime.cpp, so the
 *  rest of the pipeline (durations, token stage, host iSTFT, DC block) is
 *  unchanged. Outputs match the GGML path to float rounding, not bit-exact:
 *  acoustic.output is folded into decoder.embed, GELU uses an erf
 *  approximation with |error| < 1.5e-7, and sums run in a different order.
 *
 *  Only built on AArch64 with NEON; elsewhere available() is false and the
 *  constructor throws. */
class SanoTtsNeonDecoder {
public:
    /** Returns the host float32 data of one model tensor, in PyTorch order. */
    using WeightReader = std::function<std::vector<float>(const std::string & name)>;

    static bool available();

    /** threads: used once here to fold acoustic.output into decoder.embed. */
    SanoTtsNeonDecoder(const SanoTtsConfig & config, const WeightReader & read_weight, int threads);
    ~SanoTtsNeonDecoder();

    /** context: acoustic_hidden x frames, feats: 3 x frames, noise:
     *  noise_channels x frames (all channel-major, as the GGML graph's
     *  inputs). Returns frames x (n_fft + 2) rows of [log-magnitude | phase],
     *  as the GGML graph's spectrum output. Not thread-safe: one call at a
     *  time per instance (the runtime serialises synthesize()). */
    std::vector<float> decode(
        const std::vector<float> & context,
        const std::vector<float> & feats,
        const std::vector<float> & noise,
        int64_t frames,
        int threads);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::sanotts
