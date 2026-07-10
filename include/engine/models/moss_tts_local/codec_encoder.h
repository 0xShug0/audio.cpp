#pragma once

#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace engine::models::moss_tts_local {

// MOSS-Audio-Tokenizer-v2 encoder: turns a reference waveform into RLFQ codes
// for zero-shot voice cloning. It is the structural mirror of MossCodecDecoder --
// stereo is interleaved into one stream, patched down and run through a stack of
// causal Transformer blocks (interleaved RoPE, LayerScale, GELU MLP), then the
// RLFQ quantizer selects the nearest codes. Produces the same [num_quantizers,
// frames] code matrix the generator consumes.
class MossCodecEncoder {
public:
    MossCodecEncoder(
        const std::filesystem::path & codec_dir,
        core::ExecutionContext & execution_context,
        int64_t num_quantizers,
        size_t weight_context_bytes,
        size_t graph_arena_bytes);
    ~MossCodecEncoder();

    MossCodecEncoder(const MossCodecEncoder &) = delete;
    MossCodecEncoder & operator=(const MossCodecEncoder &) = delete;

    // Encodes a waveform given as {left, right} channels (each with the same
    // per-channel sample count, 48 kHz) into [num_quantizers][frames] codes.
    std::vector<std::vector<int32_t>> encode(const std::vector<std::vector<float>> & channels) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::moss_tts_local
