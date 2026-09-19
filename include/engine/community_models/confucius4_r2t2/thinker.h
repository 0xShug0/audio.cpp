#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/community_models/confucius4_r2t2/assets.h"
#include "engine/community_models/confucius4_r2t2/types.h"

#include <cstddef>
#include <memory>

namespace engine::community_models::confucius4_r2t2 {

/// Greedy decoding of the Confucius4-R2T2 thinker. The Qwen3-style decoder
/// stack, the audio-embedding injection, the static KV cache, and the graph
/// lifetimes all come from the shared framework runtime
/// (runtime::GreedyQwenDecoderRuntime), so this class only maps the family's
/// config and tensor layout onto it.
class R2T2ASRThinkerRuntime {
public:
    struct Impl;

    R2T2ASRThinkerRuntime(
        std::shared_ptr<const R2T2ASRAssets> assets,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type);
    ~R2T2ASRThinkerRuntime();

    R2T2ASRGeneratedTokens generate(
        const R2T2ASRPrompt & prompt,
        const R2T2ASRAudioEmbeddings & audio_embeddings,
        const R2T2ASRGenerationOptions & options);

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::confucius4_r2t2
