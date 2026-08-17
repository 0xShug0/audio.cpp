#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/community_models/minimax_h3/assets.h"
#include "engine/community_models/minimax_h3/types.h"

#include <cstddef>
#include <memory>

namespace engine::models::minimax_h3 {

class MiniMaxH3PipelineRuntime final {
public:
    MiniMaxH3PipelineRuntime(
        engine::core::ExecutionContext & execution,
        std::shared_ptr<const MiniMaxH3Assets> assets,
        size_t weight_context_bytes,
        bool mem_saver);
    ~MiniMaxH3PipelineRuntime();

    MiniMaxH3GenerateResult generate(const MiniMaxH3GenerateRequest & request);

private:
    struct Impl;

    engine::core::ExecutionContext & execution_;
    std::shared_ptr<const MiniMaxH3Assets> assets_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::minimax_h3
