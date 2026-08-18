#pragma once

#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/magpie_tts/assets.h"
#include "engine/models/magpie_tts/types.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::magpie_tts {

struct MagpieTTSRuntimeOptions {
    size_t graph_arena_bytes = 1024ull * 1024ull * 1024ull;
    size_t weight_context_bytes = 2048ull * 1024ull * 1024ull;
    assets::TensorStorageType matmul_weight_storage_type = assets::TensorStorageType::Native;
    assets::TensorStorageType conv_weight_storage_type = assets::TensorStorageType::Native;
};

class MagpieTTSRuntime {
public:
    MagpieTTSRuntime(
        std::shared_ptr<const MagpieTTSAssets> assets,
        core::ExecutionContext & execution,
        MagpieTTSRuntimeOptions options);
    ~MagpieTTSRuntime();

    MagpieTTSRuntime(const MagpieTTSRuntime &) = delete;
    MagpieTTSRuntime & operator=(const MagpieTTSRuntime &) = delete;

    runtime::AudioBuffer synthesize(
        const MagpieTokenizationResult & tokenized,
        const MagpieTTSGenerationOptions & options);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::magpie_tts
