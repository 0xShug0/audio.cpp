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
        assets::TensorStorageType storage_type);
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

    std::vector<float> feedback_embedding(const std::vector<int32_t> & codes) const;
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::minimax_music3
