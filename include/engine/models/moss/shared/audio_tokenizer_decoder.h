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

// MOSS-Audio-Tokenizer-v2 decoder: turns generated RVQ codes into a 48 kHz
// stereo waveform. The codec is "CNN-free" -- the decoder is a stack of causal
// Transformer blocks (interleaved RoPE, LayerScale, GELU MLP) separated by
// reshape-based patch upsamples, ending in a channel de-interleave that splits
// the jointly-processed stream back into left/right. The RLFQ dequantizer
// (codes -> latent) is provided by MossAudioTokenizerQuantizer.
class MossAudioTokenizerDecoder {
public:
    MossAudioTokenizerDecoder(
        const assets::TensorSource & source,
        core::ExecutionContext & execution_context,
        int64_t num_quantizers,
        size_t weight_context_bytes,
        size_t graph_arena_bytes,
        AudioTokenizerConfig config = moss_audio_tokenizer_v2_config());
    ~MossAudioTokenizerDecoder();

    MossAudioTokenizerDecoder(const MossAudioTokenizerDecoder &) = delete;
    MossAudioTokenizerDecoder & operator=(const MossAudioTokenizerDecoder &) = delete;

    int64_t sampling_rate() const noexcept;

    // Decodes [num_quantizers][steps] codes into a stereo waveform returned as
    // {left, right}, each with steps * 3840 samples at 48 kHz.
    std::vector<std::vector<float>> decode(const std::vector<std::vector<int32_t>> & codes) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::moss
