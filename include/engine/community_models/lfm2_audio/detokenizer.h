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

    // The head output for frames[first_frame:], row-major
    // [frames * upsample][output_size]: the log-magnitudes of the n_fft / 2 + 1
    // bins, then their phases. `frames` holds one code per codebook for each
    // frame, each below codebook_size; the frames before first_frame are the
    // context, so a stream can decode each new frame as it comes.
    std::vector<float> spectrum(const std::vector<std::vector<int32_t>> & frames, int64_t first_frame = 0);

    // Mono audio at config.sample_rate, hop_length samples per spectrum row.
    std::vector<float> decode(const std::vector<std::vector<int32_t>> & frames);

    [[nodiscard]] const std::vector<float> & window() const;
    [[nodiscard]] const Lfm2DetokenizerConfig & config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The detokenizer's ISTFT (ISTFT with padding "same", after Vocos): irfft of
// exp(log-magnitude) * e^(i phase), windowed overlap-add, (n_fft - hop) / 2
// samples trimmed at each end and division by the window envelope. Unlike the
// framework's Vocos ISTFT, the magnitude is not clamped, as in the reference.
//
// Rows can come in pieces: a sample is final once every window over it has
// been added, and pieces give the same samples as one call.
class Lfm2StreamingIstft {
public:
    Lfm2StreamingIstft(std::vector<float> window, int64_t hop_length);

    // Adds `rows` spectrum rows and returns the samples they complete.
    std::vector<float> push(const std::vector<float> & spectrum, int64_t rows);

    // The remaining samples, once no rows follow: hop_length per row in all.
    std::vector<float> finish();

private:
    std::vector<float> emit(int64_t until);

    std::vector<float> window_;
    int64_t n_fft_ = 0;
    int64_t hop_ = 0;
    int64_t pad_ = 0;
    int64_t rows_ = 0;
    int64_t emitted_ = 0;
    int64_t base_ = 0;  // overlap-add index of folded_[0]
    std::vector<float> folded_;
    std::vector<float> envelope_;
};

std::vector<float> lfm2_audio_istft(
    const std::vector<float> & spectrum, int64_t rows, const std::vector<float> & window, int64_t hop_length);

}  // namespace engine::community_models::lfm2_audio
