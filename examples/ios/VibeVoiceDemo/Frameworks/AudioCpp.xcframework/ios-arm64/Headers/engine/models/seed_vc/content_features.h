#pragma once

#include "engine/framework/modules/speech_encoders/hubert_encoder.h"
#include "engine/models/seed_vc/astral_quantizer.h"

#include <cstdint>
#include <vector>

namespace engine::models::seed_vc {

enum class SeedVcContentFeatureKind {
    Wide,
    Narrow,
};

std::vector<float> seed_vc_wav2vec2_normalize_16k(const std::vector<float> & waveform);

struct SeedVcContentFeatureOutput {
    std::vector<int32_t> indices;
    int64_t batch = 0;
    int64_t tokens = 0;
    int64_t codebook_size = 0;
};

class SeedVcContentFeatureExtractor {
public:
    SeedVcContentFeatureExtractor() = default;
    SeedVcContentFeatureExtractor(
        const engine::modules::HubertEncoderComponent * hubert,
        const SeedVcAstralQuantizer * wide_quantizer,
        const SeedVcAstralQuantizer * narrow_quantizer);

    SeedVcContentFeatureOutput extract_16k_mono(
        const std::vector<float> & waveform,
        SeedVcContentFeatureKind kind) const;

private:
    SeedVcContentFeatureOutput extract_chunk_16k_mono(
        const std::vector<float> & waveform,
        SeedVcContentFeatureKind kind) const;

    const engine::modules::HubertEncoderComponent * hubert_ = nullptr;
    const SeedVcAstralQuantizer * wide_quantizer_ = nullptr;
    const SeedVcAstralQuantizer * narrow_quantizer_ = nullptr;
};

}  // namespace engine::models::seed_vc
