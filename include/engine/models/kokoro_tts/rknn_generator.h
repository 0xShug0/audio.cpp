#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::kokoro_tts {

// Rockchip NPU (RKNN) backend for the iSTFTNet generator (latent to spectrogram).
//
// The graph is the generator the GGUF weights describe, re-traced at one fixed window
// (see tools/prepare_kokoro_tts_rknn_generator.py): `x [1,512,W]`, the harmonic
// conditioning `har [1,22,300W+1]`, the style vector, and the frame mask at the two
// rates the graph normalizes at. One call voices a window and returns its
// spectrogram columns; the engine has no dynamic shapes, so a longer utterance walks
// the window over the latent: `W - 2*padding` frames emitted per step, `padding` real
// frames of context on each side, zero-padded and mask-zeroed where the latent has run
// out, and the window that reaches the end right-aligned in the graph input.
//
// The seams are stitched in the spectrogram domain: each window's emitted columns are
// written into one full-length spectrogram and the iSTFT head runs once over it, so
// there is no waveform crossfade and no depop search.
//
// The generator normalizes per window (AdaIN), so a multi-window decode is a different
// synthesis than a full-length one, not an approximation of it: pick a window wide
// enough that a whole text chunk fits one window, which is exact.
//
// Two knobs soften the per-window-AdaIN pumping of the shipping walk, each off by
// default (overlap = 0, reference = null keeps the walk byte-identical):
//  - `overlap` (constructor): advance the walk by `emitted - overlap` latent frames
//  instead of `emitted`, so adjacent windows share real frames, and blend the shared
//  spectrogram columns with a raised-cosine ramp instead of hard-writing them.
//  - `reference` (run): a full-length CPU-decode spectrogram in the same layout as the
//  result; each window's log-magnitude rows get a per-window additive gain aligning
//  their loudness (the per-column sum of linear magnitudes, the waveform's energy
//  proxy) with the reference's over the columns the reference marks as speech (the
//  silence floor is not comparable between the two normalizations and swamps a plain
//  log-domain mean).
class KokoroTtsRknnGenerator {
public:
    // `latent_channels`, `feature_rows` and `samples_per_frame` (audio samples per latent
    // frame, the upsampling product times the iSTFT hop) come from the loaded weights;
    // the engine's own shapes are checked against them at load.
    KokoroTtsRknnGenerator(
        const std::string & model_path,
        int64_t latent_channels,
        int64_t feature_rows,
        int64_t samples_per_frame,
        int64_t overlap = 0);
    ~KokoroTtsRknnGenerator();

    KokoroTtsRknnGenerator(const KokoroTtsRknnGenerator &) = delete;
    KokoroTtsRknnGenerator & operator=(const KokoroTtsRknnGenerator &) = delete;

    // Latent frames per NPU step (the compiled window) and per emitted step.
    int64_t window() const;
    int64_t emitted() const;
    // Spectrogram columns the generator emits per latent frame (the upsampling product).
    int64_t columns_per_frame() const;

    // `x` is `latent_channels` rows of `x_stride` floats, `har` is `feature_rows` rows
    // of `har_stride` floats; only the first `frames` columns of `x` and the first
    // `frames * columns_per_frame() + 1` columns of `har` are real. The result is
    // `feature_rows` rows of `frames * columns_per_frame() + 1` contiguous floats.
    // `reference` (optional) is a full-length spectrogram in this call's result
    // layout; each window's log-magnitude rows are gain-corrected toward its loudness.
    std::vector<float> run(
        const float * x,
        int64_t x_stride,
        const float * har,
        int64_t har_stride,
        const float * style,
        int64_t frames,
        const float * reference = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::kokoro_tts
