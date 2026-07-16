#pragma once

#include "engine/models/moss/shared/audio_tokenizer_config.h"

#include <cstdint>
#include <vector>

namespace engine::assets {
class TensorSource;
}

namespace engine::models::moss {

// Dequantizes MOSS-Audio-Tokenizer-v2 codes (RLFQ) into the codec's continuous
// latent, i.e. the input to the codec decoder stack. Codes are the
// [num_quantizers, steps] matrix produced by generation; the returned latent is
// [code_dim, steps] row-major (channel-major), matching the Python
// quantizer.decode_codes output [1, code_dim, steps]. This is the plain-linear
// dequant path (per-codebook embedding lookup -> weight-normalized 1x1 conv ->
// residual sum -> output projection); the transformer decoder is a later phase.
class MossAudioTokenizerQuantizer {
public:
    MossAudioTokenizerQuantizer(
        const assets::TensorSource & source,
        int64_t num_quantizers,
        AudioTokenizerQuantizerConfig config = moss_audio_tokenizer_v2_config().quantizer);

    int64_t code_dim() const noexcept { return code_dim_; }
    int64_t num_quantizers() const noexcept { return num_quantizers_; }

    std::vector<float> decode(const std::vector<std::vector<int32_t>> & codes) const;

    // Quantizes the encoder latent into codes: the inverse of decode(). `hidden`
    // is [frames, code_dim] feature-last (row-major: frame * code_dim + channel),
    // matching the codec encoder's output. Mirrors the RLFQ forward pass
    // (input_proj -> per-quantizer in_proj -> L2-normalized nearest code ->
    // residual subtraction) and returns the [num_quantizers][frames] code matrix.
    std::vector<std::vector<int32_t>> encode(const std::vector<float> & hidden, int64_t frames) const;

private:
    struct Codebook {
        std::vector<float> table;             // [codebook_size, codebook_dim] row-major
        std::vector<float> table_normalized;  // [codebook_size, codebook_dim], L2-normalized rows (encode)
        std::vector<float> out_weight;        // [rvq_dim, codebook_dim] row-major
        std::vector<float> out_bias;          // [rvq_dim]
        std::vector<float> latent_table;      // [codebook_size, code_dim] row-major (decode)
        std::vector<float> in_weight;         // [codebook_dim, rvq_dim] row-major (encode)
        std::vector<float> in_bias;           // [codebook_dim] (encode)
    };

    int64_t codebook_size_ = 0;
    int64_t codebook_dim_ = 0;
    int64_t rvq_dim_ = 0;
    int64_t code_dim_ = 0;
    int64_t num_quantizers_ = 0;
    std::vector<Codebook> codebooks_;
    std::vector<float> output_weight_;  // [code_dim, rvq_dim] row-major
    std::vector<float> output_bias_;    // [code_dim]
    std::vector<float> input_weight_;   // [rvq_dim, code_dim] row-major (encode)
    std::vector<float> input_bias_;     // [rvq_dim] (encode)
};

}  // namespace engine::models::moss
