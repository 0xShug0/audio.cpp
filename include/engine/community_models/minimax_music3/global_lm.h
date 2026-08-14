#pragma once

#include "engine/community_models/minimax_music3/assets.h"
#include "engine/community_models/minimax_music3/prompt.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/transformers/qwen_causal_decode_runtime.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::minimax_music3 {

struct MiniMaxMusic3GlobalLMWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    modules::QwenCausalDecodeRuntimeWeights qwen;
};

MiniMaxMusic3GlobalLMWeights load_minimax_music3_global_lm_weights(
    const MiniMaxMusic3Assets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type);

modules::QwenCausalDecodeRuntimeConfig make_minimax_music3_global_lm_runtime_config(
    const MiniMaxMusic3Config & config,
    core::BackendType backend_type,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes);

}  // namespace engine::models::minimax_music3
