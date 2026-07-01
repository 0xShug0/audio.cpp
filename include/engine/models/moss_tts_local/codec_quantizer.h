#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace engine::models::moss_tts_local {

// Dequantizes MOSS-Audio-Tokenizer-v2 codes (RLFQ) into the codec's continuous
// latent, i.e. the input to the codec decoder stack. Codes are the
// [num_quantizers, steps] matrix produced by generation; the returned latent is
// [code_dim, steps] row-major (channel-major), matching the Python
// quantizer.decode_codes output [1, code_dim, steps]. This is the plain-linear
// dequant path (per-codebook embedding lookup -> weight-normalized 1x1 conv ->
// residual sum -> output projection); the transformer decoder is a later phase.
class MossCodecDequantizer {
public:
    MossCodecDequantizer(const std::filesystem::path & codec_dir, int64_t num_quantizers);

    int64_t code_dim() const noexcept { return code_dim_; }
    int64_t num_quantizers() const noexcept { return num_quantizers_; }

    std::vector<float> decode(const std::vector<std::vector<int32_t>> & codes) const;

private:
    struct Codebook {
        std::vector<float> table;       // [codebook_size, codebook_dim] row-major
        std::vector<float> out_weight;  // [rvq_dim, codebook_dim] row-major
        std::vector<float> out_bias;    // [rvq_dim]
    };

    int64_t codebook_size_ = 0;
    int64_t codebook_dim_ = 0;
    int64_t rvq_dim_ = 0;
    int64_t code_dim_ = 0;
    int64_t num_quantizers_ = 0;
    std::vector<Codebook> codebooks_;
    std::vector<float> output_weight_;  // [code_dim, rvq_dim] row-major
    std::vector<float> output_bias_;    // [code_dim]
};

}  // namespace engine::models::moss_tts_local
