#pragma once

#include "engine/community_models/minimax_music3/assets.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/transformers/qwen_decoder.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::minimax_music3 {

struct MiniMaxMusic3DepthCodes {
    std::vector<int32_t> codes;
    std::vector<float> hidden;
};

struct MiniMaxMusic3DepthWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue audio_embeddings;
    modules::LinearWeights projection;
    core::TensorValue position_embedding;
    modules::QwenDecoderStackWeights stack;
    modules::NormWeights norm;
    std::vector<modules::LinearWeights> audio_heads;
};

class MiniMaxMusic3DepthDecoderRuntime {
public:
    MiniMaxMusic3DepthDecoderRuntime(
        std::shared_ptr<const MiniMaxMusic3Assets> assets,
        core::TensorValue global_token_embedding,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type,
        bool evict_cuda_graph_cache_on_release = false);
    ~MiniMaxMusic3DepthDecoderRuntime();

    MiniMaxMusic3DepthCodes generate(
        const std::vector<float> & last_hidden_cond,
        const std::vector<float> & last_hidden_uncond,
        int32_t semantic_code,
        float guidance_scale,
        int64_t top_k,
        uint64_t seed,
        uint64_t & sample_call_index,
        uint64_t & rng_offset_blocks);

    // Ensemble variant: decodes the depth chain for `songs` independent takes
    // in one batched pass (rows are [cond_0, uncond_0, cond_1, ...]). Each
    // take samples with its own seed/counters, so results are identical to
    // running the single-take path per song.
    std::vector<MiniMaxMusic3DepthCodes> generate_batch(
        const std::vector<float> & interleaved_hiddens,
        int64_t songs,
        const std::vector<int32_t> & semantic_codes,
        float guidance_scale,
        int64_t top_k,
        const std::vector<uint64_t> & seeds,
        std::vector<uint64_t> & sample_call_indices,
        std::vector<uint64_t> & rng_offset_blocks);

    std::vector<float> feedback_embedding(const std::vector<int32_t> & codes) const;
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::minimax_music3
