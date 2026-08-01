#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/miotts/assets.h"
#include "engine/models/miotts/types.h"

#include <cstddef>
#include <memory>

namespace engine::models::miotts {

class MioTTSCausalLMRuntime {
public:
    struct Impl;

    MioTTSCausalLMRuntime(
        std::shared_ptr<const MioTTSAssets> assets,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type);
    ~MioTTSCausalLMRuntime();

    MioTTSGeneratedTokens generate(
        const MioTTSPrompt & prompt,
        const MioTTSGenerationOptions & options);

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::miotts
