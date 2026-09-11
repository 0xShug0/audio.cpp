#pragma once

// VibeASR language model: the Qwen2 causal decoder whose projections are stored
// as ternary GGML_TYPE_I2_S. Speech features from the VAE encoder replace the
// prompt's <|speech_pad|> placeholders before prefill.
//
// Ported from https://github.com/microsoft/VibeASR.cpp (src/lm.cpp).

#include "engine/community_models/vibeasr/assets.h"
#include "engine/framework/core/execution_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::vibeasr {

struct VibeASRLmPrompt {
    std::vector<int32_t> input_ids;
    // Positions in input_ids occupied by <|speech_pad|>, in order.
    std::vector<int32_t> speech_positions;
};

// Summed acoustic + semantic connector output, row-major [tokens][hidden_size].
struct VibeASRSpeechEmbeddings {
    int64_t tokens = 0;
    int64_t hidden_size = 0;
    std::vector<float> values;
};

struct VibeASRGenerationOptions {
    int64_t max_new_tokens = 1024;
    std::vector<int32_t> eos_token_ids;
};

class VibeASRLmRuntime {
public:
    VibeASRLmRuntime(
        std::shared_ptr<const assets::TensorSource> weights_source,
        const VibeASRLmConfig & config,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes);
    ~VibeASRLmRuntime();

    VibeASRLmRuntime(const VibeASRLmRuntime &) = delete;
    VibeASRLmRuntime & operator=(const VibeASRLmRuntime &) = delete;

    // Greedy decode. Stops at any eos id or after max_new_tokens.
    std::vector<int32_t> generate(
        const VibeASRLmPrompt & prompt,
        const VibeASRSpeechEmbeddings & speech,
        const VibeASRGenerationOptions & options);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::vibeasr
