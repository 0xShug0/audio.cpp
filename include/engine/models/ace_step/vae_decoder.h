#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/ace_step/assets.h"
#include "engine/models/ace_step/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace engine::models::ace_step {

struct AceStepVAEDecodeConfig {
    std::optional<int64_t> tile_frames;
    std::optional<int64_t> tile_overlap_frames;
};

class AceStepVAEDecoderRuntime {
public:
    AceStepVAEDecoderRuntime(
        std::shared_ptr<const AceStepAssets> assets,
        core::ExecutionContext & execution,
        assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native,
        size_t graph_arena_bytes = 256ull * 1024ull * 1024ull,
        size_t weight_context_bytes = 64ull * 1024ull * 1024ull,
        AceStepVAEDecodeConfig decode_config = {});
    ~AceStepVAEDecoderRuntime();

    runtime::AudioBuffer decode(const AceStepLatents & latents) const;
    void release_runtime_graphs() const;

private:
    class Impl;

    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::ace_step
