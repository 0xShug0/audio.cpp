#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/community_models/audio8_asr/types.h"

#include <cstddef>
#include <memory>

namespace engine::community_models::audio8_asr {

// Greedy causal decoder for the Audio8 8-layer Qwen2-style LM. Audio
// embeddings are injected into the token embedding sequence at the prompt's
// audio placeholder positions before prefill.
class Audio8ThinkerRuntime {
public:
    Audio8ThinkerRuntime(
        std::shared_ptr<const assets::TensorSource> weights_source,
        const Audio8ASRDecoderConfig & config,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type);
    ~Audio8ThinkerRuntime();

    Audio8ThinkerRuntime(const Audio8ThinkerRuntime &) = delete;
    Audio8ThinkerRuntime & operator=(const Audio8ThinkerRuntime &) = delete;

    Audio8ASRGeneratedTokens generate(
        const Audio8ASRPrompt & prompt,
        const Audio8ASRAudioEmbeddings & audio_embeddings,
        const Audio8ASRGenerationOptions & options);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::audio8_asr
