#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::piper_tts {

// Rockchip NPU (RKNN) backend for the VITS decoder (latent to PCM).
//
// The graph is the voice's own decoder sliced at the latent (see
// tools/prepare_piper_tts_rknn_vocoder.py): one call voices a fixed window of latent
// slices and returns window * samples_per_slice audio samples. The engine has no
// dynamic shapes, so a whole utterance is decoded as overlapping windows: 45 emitted
// slices with real context on both sides, the last window right-aligned in the graph
// input so the end of the utterance sits at the same tensor boundary it has in a
// full-length decode, and a depop search plus a short linear crossfade at every seam.
//
// The engine is compiled from one voice, so it is only valid for the model package it
// was built from; the latent channel count and the samples-per-slice product are
// checked against the voice config at load.
class PiperTtsRknnDecoder {
public:
    PiperTtsRknnDecoder(
        const std::string & model_path,
        int64_t channels,
        int64_t samples_per_slice);
    ~PiperTtsRknnDecoder();

    PiperTtsRknnDecoder(const PiperTtsRknnDecoder &) = delete;
    PiperTtsRknnDecoder & operator=(const PiperTtsRknnDecoder &) = delete;

    // Latent slices per NPU step (the compiled window) and per emitted step.
    int64_t window() const;
    int64_t chunk() const;

    // `latent` is row-major [channels][frames]; the result is frames *
    // samples_per_slice float samples in [-1, 1].
    std::vector<float> decode(const float * latent, int64_t channels, int64_t frames);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::piper_tts
