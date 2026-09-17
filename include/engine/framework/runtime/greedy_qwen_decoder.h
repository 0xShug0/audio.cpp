#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/transformers/qwen_causal_decoder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::runtime {

// Specification for a greedy Qwen-family causal decoder: how to find its
// tensors in a weight source and how the shared QwenCausalDecoder stack is
// configured. Covers Qwen2-style decoders (attention biases, no Q/K norms,
// as in Audio8-ASR) and Qwen3-style decoders (Q/K norms, no attention
// biases, as in the qwen3_asr thinker), with separate or packed QKV
// projections and tied or separate LM heads.
struct GreedyQwenDecoderSpec {
    modules::QwenCausalDecoderConfig decoder;
    int64_t vocab_size = 0;
    int64_t max_position_embeddings = 0;
    bool tie_word_embeddings = false;
    bool attention_bias = false;
    bool packed_qkv = false;
    std::string token_embedding_tensor;
    std::string lm_head_tensor;  // used when !tie_word_embeddings
    std::string final_norm_tensor;
    std::string layer_prefix;  // e.g. "language_model.model.layers"
    std::vector<int64_t> eos_token_ids;
};

// Greedy autoregressive decoding over a Qwen-style decoder stack: prefill
// with optional audio-embedding injection (ggml_set_rows at prompt
// positions) and static-cache step decode, hiding the graph lifetime and
// K/V state handoff that model families otherwise duplicate.
class GreedyQwenDecoderRuntime {
public:
    struct Injection {
        std::vector<float> values;        // tokens * hidden, token-major
        int64_t tokens = 0;
        std::vector<int32_t> positions;   // prompt positions to replace
    };

    struct Prompt {
        std::vector<int32_t> input_ids;
        Injection injection;  // optional
    };

    GreedyQwenDecoderRuntime(
        std::shared_ptr<const assets::TensorSource> weights_source,
        const GreedyQwenDecoderSpec & spec,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type);
    ~GreedyQwenDecoderRuntime();

    GreedyQwenDecoderRuntime(const GreedyQwenDecoderRuntime &) = delete;
    GreedyQwenDecoderRuntime & operator=(const GreedyQwenDecoderRuntime &) = delete;

    std::vector<int32_t> generate(const Prompt & prompt, int64_t max_new_tokens);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::runtime
