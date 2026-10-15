#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::inflect_v2 {

// Rockchip NPU (RKNN) backend for the Inflect v2 decoder (latent to PCM).
//
// The graph is the model's own decoder half sliced at the latent (see
// tools/rknn/prepare_inflect_v2_rknn_decoder.py): the inverse flow plus `dec`, one call per fixed
// window of latent slices, returning window * samples_per_slice audio samples. The engine has no
// dynamic shapes, so a whole utterance is decoded as overlapping windows: 192 emitted slices with
// real context on both sides, the last window extending its left context to fill the compiled
// shape, and a depop search plus a short linear crossfade at every seam.
//
// The engine is compiled from one model, so it is only valid for the model package it was built
// from; the latent channel count and the samples-per-slice product are checked against the model
// config at load.
class InflectV2RknnDecoder {
public:
    InflectV2RknnDecoder(
        const std::string & model_path,
        int64_t channels,
        int64_t samples_per_slice);
    ~InflectV2RknnDecoder();

    InflectV2RknnDecoder(const InflectV2RknnDecoder &) = delete;
    InflectV2RknnDecoder & operator=(const InflectV2RknnDecoder &) = delete;

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

}  // namespace engine::models::inflect_v2
