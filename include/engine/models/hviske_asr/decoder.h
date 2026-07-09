#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/hviske_asr/assets.h"
#include "engine/models/hviske_asr/encoder.h"
#include "engine/models/hviske_asr/weights.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::hviske_asr {

struct HviskeDecoderResult {
    std::vector<int32_t> token_ids;
};

struct HviskeDecodingOptions {
    int64_t max_new_tokens = 256;
    int64_t num_beams = 1;
    float length_penalty = 1.0f;
    bool do_sample = false;
    float temperature = 1.0f;
    int64_t top_k = 50;
    float top_p = 1.0f;
    uint32_t seed = 0;
};

class HviskeDecoderRuntime {
public:
    HviskeDecoderRuntime(
        std::shared_ptr<const HviskeAssets> assets,
        std::shared_ptr<const HviskeWeights> weights,
        engine::core::ExecutionContext & execution_context,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes);
    ~HviskeDecoderRuntime();

    HviskeDecoderRuntime(const HviskeDecoderRuntime &) = delete;
    HviskeDecoderRuntime & operator=(const HviskeDecoderRuntime &) = delete;
    HviskeDecoderRuntime(HviskeDecoderRuntime &&) noexcept;
    HviskeDecoderRuntime & operator=(HviskeDecoderRuntime &&) noexcept;

    HviskeDecoderResult generate(
        const std::vector<int32_t> & prompt_ids,
        const HviskeEncodedAudio & encoded,
        const HviskeDecodingOptions & options);

private:
    struct PrefillGraph;
    struct DecodeGraph;
    struct BeamDecodeGraph;

    std::shared_ptr<const HviskeAssets> assets_;
    std::shared_ptr<const HviskeWeights> weights_;
    engine::core::ExecutionContext * execution_context_ = nullptr;
    size_t prefill_graph_arena_bytes_ = 0;
    size_t decode_graph_arena_bytes_ = 0;
    std::unique_ptr<PrefillGraph> prefill_graph_;
    std::unique_ptr<DecodeGraph> decode_graph_;
    std::unique_ptr<BeamDecodeGraph> beam_decode_graph_;
};

}  // namespace engine::models::hviske_asr
