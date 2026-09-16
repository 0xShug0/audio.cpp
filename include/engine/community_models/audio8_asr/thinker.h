#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/greedy_qwen_decoder.h"
#include "engine/community_models/audio8_asr/types.h"

#include <cstddef>
#include <memory>

namespace engine::community_models::audio8_asr {

// The Audio8 8-layer Qwen2-style decoder, expressed through the framework's
// shared greedy Qwen decoder runtime (prefill with audio-embedding injection
// plus static-cache step decode). Owns only the family-specific spec.
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
    runtime::GreedyQwenDecoderRuntime runtime_;
    std::shared_ptr<const Audio8ASRDecoderConfig> config_;
};

}  // namespace engine::community_models::audio8_asr
