#pragma once

// LFM3-Audio depth decoder: from a backbone hidden state, a bidirectional
// transformer over the 12 codebook slots predicts one audio frame in MaskGIT
// passes.

#include "engine/community_models/lfm3_audio/assets.h"
#include "engine/community_models/lfm3_audio/weights.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::lfm3_audio {

using Lfm3Codes = std::array<int32_t, 12>;

class Lfm3AudioSampler;  // sampling.h

class Lfm3DepthDecoderRuntime {
public:
    Lfm3DepthDecoderRuntime(std::shared_ptr<const assets::TensorSource> model, const Lfm3DepthConfig & config,
                            core::ExecutionContext & execution, const Lfm3WeightOptions & options);
    ~Lfm3DepthDecoderRuntime();
    // One forward: x = depth_linear(h).view(12,1024) + (masked ? mask_embedding : E_i[code_i]);
    // 3 bidirectional layers + trunk embedding_norm; per-codebook RMSNorm; logits_i = E_i · o_i.
    // Returns row-major [12][2017]. hidden is [2048].
    std::vector<float> logits(const std::vector<float> & hidden, const Lfm3Codes & codes, const std::array<bool, 12> & masked);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct Lfm3FrameTaps {  // for dumps
    std::vector<std::vector<float>> pass_logits;
};

// MaskGIT: passes over `order` split by `schedule`; samples each picked codebook in order; stops after
// the first pass that yields the EOS code (output-identical shortcut); EOS -> all 2016.
Lfm3Codes lfm3_maskgit_frame(Lfm3DepthDecoderRuntime & depth, const std::vector<float> & hidden, const Lfm3Codes & order,
                             const std::vector<int64_t> & schedule, class Lfm3AudioSampler & sampler, int32_t end_of_audio,
                             Lfm3FrameTaps * taps = nullptr);

}  // namespace engine::community_models::lfm3_audio
