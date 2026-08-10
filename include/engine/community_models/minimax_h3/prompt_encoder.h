#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/community_models/minimax_h3/assets.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace engine::models::minimax_h3 {

class PromptEncoderWeightStore {
public:
    PromptEncoderWeightStore(
        engine::core::ExecutionContext & execution_context,
        std::shared_ptr<const engine::assets::TensorSource> tensor_source,
        size_t weight_context_bytes);
    PromptEncoderWeightStore(
        engine::core::ExecutionContext & execution_context,
        std::shared_ptr<const engine::assets::TensorSource> tensor_source,
        size_t weight_context_bytes,
        const std::vector<std::string> & required_names,
        const std::vector<std::string> & prefix_filters);

    const engine::core::TensorValue & require(std::string_view name) const;
    engine::core::ExecutionContext & execution;

private:
    std::shared_ptr<const engine::assets::TensorSource> source_;
    engine::core::BackendWeightStore store_;
    std::unordered_map<std::string, engine::core::TensorValue> weights_;
};

std::vector<float> run_prompt_graph(PromptEncoderWeightStore & weights, const MiniMaxH3Config & cfg, const std::vector<int32_t> & ids);
std::vector<float> run_prompt_graph_layerwise(
    engine::core::ExecutionContext & execution,
    std::shared_ptr<const engine::assets::TensorSource> tensor_source,
    const MiniMaxH3Config & cfg,
    const std::vector<int32_t> & ids,
    size_t weight_context_bytes,
    int64_t layer_batch);

}  // namespace engine::models::minimax_h3
