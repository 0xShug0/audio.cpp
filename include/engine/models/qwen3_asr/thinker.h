#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/attention_fallback.h"
#include "engine/models/qwen3_asr/assets.h"
#include "engine/models/qwen3_asr/types.h"

#include <cstddef>
#include <memory>
#include <optional>

namespace engine::models::qwen3_asr {

struct Qwen3ASRPerfSettings {
    std::optional<core::AttentionPreference> attention;
    bool packed_projections = false;
    ggml_type kv_cache_type = GGML_TYPE_F32;
    int64_t kv_cache_alignment = 1;
    int64_t offline_batch_size = 1;
};

class Qwen3ASRThinkerRuntime {
public:
    struct Impl;

    Qwen3ASRThinkerRuntime(
        std::shared_ptr<const Qwen3ASRAssets> assets,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type,
        const Qwen3ASRPerfSettings & perf = {});
    ~Qwen3ASRThinkerRuntime();

    Qwen3ASRGeneratedTokens generate(
        const Qwen3ASRPrompt & prompt,
        const Qwen3ASRAudioEmbeddings & audio_embeddings,
        const Qwen3ASRGenerationOptions & options);

    std::vector<Qwen3ASRGeneratedTokens> generate_batch(
        const std::vector<Qwen3ASRPrompt> & prompts,
        const std::vector<Qwen3ASRAudioEmbeddings> & embeddings,
        const std::vector<Qwen3ASRGenerationOptions> & options);

    std::vector<int32_t> classify_prompt(
        const Qwen3ASRPrompt & prompt,
        const Qwen3ASRAudioEmbeddings & audio_embeddings);

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::qwen3_asr
