#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/models/muscriptor/assets.h"
#include "engine/models/muscriptor/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace engine::models::muscriptor {

class MuScriptorDecodeCacheStorage;

struct MuScriptorLayerWeights {
    modules::LinearWeights in_proj;
    modules::LinearWeights out_proj;
    modules::NormWeights norm1;
    modules::NormWeights norm2;
    modules::FeedForwardWeights mlp;
};

struct MuScriptorWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::LinearWeights mel_projection;
    core::TensorValue instrument_embedding;
    core::TensorValue dataset_embedding;
    core::TensorValue token_embedding;
    std::vector<MuScriptorLayerWeights> layers;
    modules::NormWeights output_norm;
    modules::LinearWeights output_head;
};

enum class MuScriptorPerfMode {
    Exact,
    FlashAttention,
};

struct MuScriptorDecoderOptions {
    assets::TensorStorageType weight_type = assets::TensorStorageType::Native;
    MuScriptorPerfMode perf_mode = MuScriptorPerfMode::FlashAttention;
    size_t weight_context_bytes = 512ull * 1024ull * 1024ull;
    size_t condition_graph_arena_bytes = 128ull * 1024ull * 1024ull;
    size_t prefill_graph_arena_bytes = 768ull * 1024ull * 1024ull;
    size_t decode_graph_arena_bytes = 512ull * 1024ull * 1024ull;
};

class MuScriptorDecoderRuntime {
public:
    MuScriptorDecoderRuntime(
        std::shared_ptr<const MuScriptorAssets> assets,
        core::ExecutionContext & execution,
        MuScriptorDecoderOptions options);
    ~MuScriptorDecoderRuntime();

    MuScriptorConditioning condition(
        const std::vector<float> & log_mel,
        const std::vector<int32_t> & mel_mask,
        const std::vector<int32_t> & instrument_ids,
        int32_t dataset_id);

    MuScriptorConditioning condition_batch(
        const std::vector<float> & log_mel,
        const std::vector<int32_t> & mel_mask,
        const std::vector<int32_t> & instrument_ids,
        const std::vector<int32_t> & dataset_ids,
        int64_t batch);

    MuScriptorGeneratedChunk generate_greedy(
        const MuScriptorConditioning & conditioning,
        const std::vector<int32_t> & prompt,
        int64_t max_tokens,
        int32_t eos_id,
        const std::vector<int32_t> & forbidden_tokens);

    MuScriptorGeneratedChunk generate(
        const MuScriptorConditioning & conditioning,
        const MuScriptorConditioning * null_conditioning,
        const std::vector<int32_t> & prompt,
        int64_t max_tokens,
        int32_t eos_id,
        const std::vector<int32_t> & forbidden_tokens,
        const MuScriptorGenerationOptions & options);

    std::vector<MuScriptorGeneratedChunk> generate_batch(
        const MuScriptorConditioning & conditioning,
        const std::vector<std::vector<int32_t>> & prompts,
        int64_t max_tokens,
        int32_t eos_id,
        const std::vector<int32_t> & forbidden_tokens,
        const MuScriptorGenerationOptions & options);

private:
    class ConditionGraph;
    class PrefillGraph;
    class DecodeGraph;

    std::shared_ptr<const MuScriptorAssets> assets_;
    core::ExecutionContext & execution_;
    std::shared_ptr<MuScriptorWeights> weights_;
    MuScriptorDecoderOptions options_;
    std::unique_ptr<ConditionGraph> condition_graph_;
    std::unique_ptr<PrefillGraph> prefill_graph_;
    std::unique_ptr<MuScriptorDecodeCacheStorage> decode_cache_;
    std::unique_ptr<DecodeGraph> decode_graph_;
};

}  // namespace engine::models::muscriptor
