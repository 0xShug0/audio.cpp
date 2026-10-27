#pragma once

// LFM3-Audio codec decoder (LiquidCodec): FSQ latents of a frame's 12 codes,
// projected and upsampled x6, an LFM3 stack with sliding-window attention, a
// linear head to log-magnitude and phase, and a streaming ISTFT to 24 kHz.

#include "engine/community_models/lfm3_audio/assets.h"
#include "engine/community_models/lfm3_audio/weights.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::lfm3_audio {

// d_j = ((c / basis_j) % levels_j - levels_j / 2) / (levels_j / 2) per code; codes in [0, prod(levels)).
std::vector<float> lfm3_fsq_latents(const std::vector<std::vector<int32_t>> & frames, const std::vector<int64_t> & levels);  // [F][48] codebook-major

// Intermediate stages of spectrum(), row-major (tests and dumps).
struct Lfm3CodecTaps {
    std::vector<float> project_out;     // [F][quantizer_dim]
    std::vector<float> from_quantizer;  // [F][hidden]
    std::vector<float> decoder;         // [6F][hidden], the stack's final-normed output
};

class Lfm3CodecDecoder {
public:
    static constexpr int64_t kChunkFrames = 128;

    Lfm3CodecDecoder(std::shared_ptr<const assets::TensorSource> codec, const Lfm3CodecConfig & config,
                     core::ExecutionContext & execution, const Lfm3WeightOptions & options);
    ~Lfm3CodecDecoder();
    // Frames must not contain EOS. Attention K/V live in a ring of kChunkFrames * 6 + window rows (rounded up
    // to 256), so a stream has no length limit. Graphs are keyed by chunk frame count; longer pushes run in
    // kChunkFrames pieces.
    void start_stream();
    std::vector<float> push(const std::vector<std::vector<int32_t>> & frames);   // samples completed so far (Lfm2StreamingIstft::push)
    std::vector<float> finish();                                                 // remaining samples; ends the stream
    std::vector<float> decode(const std::vector<std::vector<int32_t>> & frames);  // start + push chunks + finish
    // Spectrum rows [6F][1282] of `frames` (advances the stream state but not its ISTFT; dumps/tests).
    std::vector<float> spectrum(const std::vector<std::vector<int32_t>> & frames, Lfm3CodecTaps * taps = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::lfm3_audio
