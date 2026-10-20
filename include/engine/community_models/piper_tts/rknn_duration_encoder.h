#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::piper_tts {

// Rockchip NPU (RKNN) backend for the VITS duration encoder (text tokens to prior
// stats and flow conditioning).
//
// The graph is the voice's own encoder + duration discriminator sliced at their
// projections (see tools/prepare_piper_tts_rknn_vocoder.py --encoder): tokens and
// their length in, the prior stats (2*inter_channels rows) and the flow conditioning
// (inter_channels rows) out. The export carries the frame mask inside the encoder,
// so a right-padded fixed-width run reproduces the exact-length run on the real
// frames (measured 117-130 dB fp32); each compiled engine covers one token width,
// so the sidecar keeps a ladder of bins and an utterance longer than the widest bin
// runs the ggml encoder for that step.
class PiperTtsRknnDurationEncoder {
public:
    // `model_paths` is a comma-separated list of compiled engines (the exec_plan
    // artifact list); each is validated against the voice at load.
    PiperTtsRknnDurationEncoder(
        const std::string & model_paths,
        int64_t inter_channels);
    ~PiperTtsRknnDurationEncoder();

    PiperTtsRknnDurationEncoder(const PiperTtsRknnDurationEncoder &) = delete;
    PiperTtsRknnDurationEncoder & operator=(const PiperTtsRknnDurationEncoder &) = delete;

    // Widest token count any loaded engine covers.
    int64_t max_tokens() const;

    // `token_ids` (length n <= max_tokens()) in; stats [2*inter_channels][n] and
    // conditioning [inter_channels][n], channel-major, in the layout the flow
    // graphs and prior expansion consume.
    void encode(
        const std::vector<int32_t> & token_ids,
        std::vector<float> & stats,
        std::vector<float> & conditioning);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::piper_tts
