#pragma once

// LFM2.5-Audio audio detokenizer (detokenizer.py, LFM2AudioDetokenizer): each
// frame's codes become the mean of their embeddings, repeated `upsample`
// times; a small LFM2 hybrid with causal sliding-window attention and a
// linear head give log-magnitude and phase, and an ISTFT gives the waveform.
//
// The model is causal and sees only the last few dozen steps, so long audio
// runs in chunks that overlap by that receptive field and give the same
// output as one pass, without a quadratic attention mask.

#include "engine/community_models/lfm2_audio/assets.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::lfm2_audio {

class Lfm2DetokenizerRuntime {
public:
    // Frames each chunk adds past its context; about 10 s of audio by default.
    static constexpr int64_t kDefaultChunkFrames = 128;

    Lfm2DetokenizerRuntime(
        std::shared_ptr<const assets::TensorSource> detokenizer,
        std::shared_ptr<const assets::TensorSource> vocoder,
        const Lfm2DetokenizerConfig & config,
        core::ExecutionContext & execution,
        int64_t chunk_frames = kDefaultChunkFrames);
    ~Lfm2DetokenizerRuntime();

    Lfm2DetokenizerRuntime(const Lfm2DetokenizerRuntime &) = delete;
    Lfm2DetokenizerRuntime & operator=(const Lfm2DetokenizerRuntime &) = delete;

    // The head output, row-major [frames * upsample][output_size]: the
    // log-magnitudes of the n_fft / 2 + 1 bins, then their phases. `frames`
    // holds one code per codebook for each frame, each below codebook_size.
    std::vector<float> spectrum(const std::vector<std::vector<int32_t>> & frames);

    // Mono audio at config.sample_rate, hop_length samples per spectrum row.
    std::vector<float> decode(const std::vector<std::vector<int32_t>> & frames);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The detokenizer's ISTFT (ISTFT with padding "same", after Vocos): irfft of
// exp(log-magnitude) * e^(i phase), windowed overlap-add, (n_fft - hop) / 2
// samples trimmed at each end and division by the window envelope. Unlike the
// framework's Vocos ISTFT, the magnitude is not clamped, as in the reference.
std::vector<float> lfm2_audio_istft(
    const std::vector<float> & spectrum, int64_t rows, const std::vector<float> & window, int64_t hop_length);

}  // namespace engine::community_models::lfm2_audio
