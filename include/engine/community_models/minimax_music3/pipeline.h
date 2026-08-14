#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/community_models/minimax_music3/assets.h"
#include "engine/community_models/minimax_music3/types.h"

#include <cstddef>
#include <memory>

namespace engine::models::minimax_music3 {

class MiniMaxMusic3PipelineRuntime final {
public:
    MiniMaxMusic3PipelineRuntime(
        engine::core::ExecutionContext & execution,
        std::shared_ptr<const MiniMaxMusic3Assets> assets,
        size_t weight_context_bytes,
        bool mem_saver);
    ~MiniMaxMusic3PipelineRuntime();

    MiniMaxMusic3GenerateResult generate(const MiniMaxMusic3GenerateRequest & request);

private:
    struct Impl;

    engine::core::ExecutionContext & execution_;
    std::shared_ptr<const MiniMaxMusic3Assets> assets_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::minimax_music3
