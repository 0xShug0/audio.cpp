#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/models/open_wake_word/weights.h"

#include <optional>

namespace engine::core {
class ExecutionContext;
}

namespace engine::models::open_wake_word {

struct GraphWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    std::vector<std::optional<modules::Conv2dWeights>> embedding;
    std::vector<std::optional<modules::LinearWeights>> classifier;
};

GraphWeights bind_graph_weights(const Weights &weights, const core::ExecutionContext &execution, bool embedding = true);

} // namespace engine::models::open_wake_word
