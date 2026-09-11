#pragma once

#include "engine/community_models/mira_tts/assets.h"
#include "engine/framework/core/execution_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::mira_tts {

class MiraGenerator final {
public:
    MiraGenerator(
        const MiraTTSAssets & assets,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type);
    ~MiraGenerator();

    std::vector<int32_t> generate(
        const std::vector<int32_t> & prompt_ids,
        const MiraGenerationOptions & options);
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::mira_tts
