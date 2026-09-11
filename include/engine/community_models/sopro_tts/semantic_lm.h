#pragma once

#include "engine/community_models/sopro_tts/assets.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace engine::core {
class ExecutionContext;
}
namespace engine::assets {
enum class TensorStorageType;
}

namespace engine::community_models::sopro_tts {

struct SoproSemanticLMOptions {
    int64_t max_steps = 1;
    int64_t min_steps = 1;
    float temperature = 0.8F;
    float top_p = 0.9F;
    int64_t top_k = 25;
};

// sopro/nn/ar.py + SoproModel.stream_semantic_tokens. The prompt is
// [style prefix | text | carried semantic tokens | BOS] and the model
// autoregresses semantic ids until EOS or the step budget.
//
// The stack is a standard pre-norm transformer with QK RMS-norm, SwiGLU and
// half-rotation RoPE, so it maps onto the shared Qwen decoder runtime once the
// per-branch LayerScale vectors are folded into the output projections.
class SoproSemanticLMRuntime final {
public:
    SoproSemanticLMRuntime(
        const SoproTTSAssets & assets,
        engine::core::ExecutionContext & execution_context,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType weight_storage_type);
    ~SoproSemanticLMRuntime();

    SoproSemanticLMRuntime(const SoproSemanticLMRuntime &) = delete;
    SoproSemanticLMRuntime & operator=(const SoproSemanticLMRuntime &) = delete;

    std::vector<int32_t> generate(
        const std::vector<int32_t> & text_ids,
        const std::vector<int32_t> & style_tokens,
        const std::vector<int32_t> & prompt_tokens,
        const SoproSemanticLMOptions & options,
        std::mt19937_64 & rng) const;

    void release_runtime_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// sopro/sampling.py sample_next_token, exposed for testing. `logits` is
// modified in place.
int32_t sample_next_token(
    std::vector<float> & logits,
    float temperature,
    float top_p,
    int64_t top_k,
    int32_t bos_id,
    int32_t eos_id,
    bool allow_eos,
    std::mt19937_64 & rng);

}  // namespace engine::community_models::sopro_tts
