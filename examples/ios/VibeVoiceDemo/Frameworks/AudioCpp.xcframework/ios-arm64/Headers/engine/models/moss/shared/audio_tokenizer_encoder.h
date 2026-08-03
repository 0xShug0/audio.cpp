#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/moss/shared/audio_tokenizer_config.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::assets {
class TensorSource;
}

namespace engine::models::moss {

// MOSS-Audio-Tokenizer encoder: turns a reference waveform into RLFQ codes
// for zero-shot voice cloning. It is the structural mirror of MossAudioTokenizerDecoder --
// stereo is interleaved into one stream, patched down and run through a stack of
// causal Transformer blocks (interleaved RoPE, LayerScale, GELU MLP), then the
// RLFQ quantizer selects the nearest codes. Produces the same [num_quantizers,
// frames] code matrix the generator consumes.
class MossAudioTokenizerEncoder {
public:
    MossAudioTokenizerEncoder(
        const assets::TensorSource & source,
        core::ExecutionContext & execution_context,
        int64_t num_quantizers,
        size_t weight_context_bytes,
        size_t graph_arena_bytes,
        AudioTokenizerConfig config = moss_audio_tokenizer_v2_config());
    ~MossAudioTokenizerEncoder();

    MossAudioTokenizerEncoder(const MossAudioTokenizerEncoder &) = delete;
    MossAudioTokenizerEncoder & operator=(const MossAudioTokenizerEncoder &) = delete;

    // Encodes a waveform given as {left, right} channels (each with the same
    // per-channel sample count, 48 kHz) into [num_quantizers][frames] codes.
    std::vector<std::vector<int32_t>> encode(const std::vector<std::vector<float>> & channels) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::moss
