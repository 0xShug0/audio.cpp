#pragma once

#include "engine/community_models/sanotts/config.h"
#ifdef ENGINE_HAVE_NEON
#include "engine/community_models/sanotts/neon_weights.h"
#endif

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::sanotts {

/** Complete nano-lineage decoder (frame-stage
 *  acoustic blocks -> mel projection -> noise-fed ConvNeXt-1D -> spectrum
 *  head), plus optional NEON front-end (duration + token-stage acoustic) and
 *  fused head->iSTFT->DC audio output, for CPU sessions selecting neon
 *  through sanotts.exec_plan.
 *
 *  decode() takes the same inputs and gives the same output as the GGML
 *  decoder graph in runtime.cpp. Outputs match the GGML path to float
 *  rounding, not bit-exact: acoustic.output is folded into decoder.embed,
 *  GELU uses an erf approximation with |error| < 1.5e-7, and sums run in a
 *  different order. The front end and decode_audio() exist only without
 *  ENGINE_HAVE_NEON.
 */
class SanoTtsNeonDecoder {
public:
    static bool available();

#ifdef ENGINE_HAVE_NEON
    SanoTtsNeonDecoder(const SanoTtsConfig &config, const SanoTtsNeonWeights &weights);
#else
    using WeightReader = std::function<std::vector<float>(const std::string &)>;
    SanoTtsNeonDecoder(const SanoTtsConfig &config, const WeightReader &read_weight, int threads);

    void init_frontend(const SanoTtsConfig &config, const WeightReader &read_weight);
    bool has_frontend() const;
    std::vector<float> predict_duration(const std::vector<int32_t> &tokens,
                                        const std::vector<float> &feats, int threads);
    std::vector<float> predict_acoustic(const std::vector<int32_t> &tokens,
                                        const std::vector<float> &feats, int threads);
    bool can_decode_audio() const;
    std::vector<float> decode_audio(const std::vector<float> &context,
                                    const std::vector<float> &feats,
                                    const std::vector<float> &noise,
                                    int64_t frames, int threads);
#endif
    ~SanoTtsNeonDecoder();

    /** context: acoustic_hidden x frames, feats: 3 x frames, noise:
     *  noise_channels x frames (all channel-major, as the GGML graph's
     *  inputs). Returns frames x (n_fft + 2) rows of [log-magnitude | phase],
     *  as the GGML graph's spectrum output. Not thread-safe: one call at a
     *  time per instance (the runtime serialises synthesize()). */
    std::vector<float> decode(const std::vector<float> &context, const std::vector<float> &feats,
                              const std::vector<float> &noise, int64_t frames, int threads);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::sanotts
