#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/ace_step/assets.h"
#include "engine/models/ace_step/types.h"

#include <cstddef>
#include <memory>
#include <string>

namespace engine::models::ace_step {

class AceStepVAEEncoderRuntime {
public:
    AceStepVAEEncoderRuntime(
        std::shared_ptr<const AceStepAssets> assets,
        core::ExecutionContext & execution,
        assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native,
        size_t graph_arena_bytes = 256ull * 1024ull * 1024ull,
        size_t weight_context_bytes = 64ull * 1024ull * 1024ull);
    ~AceStepVAEEncoderRuntime();

    AceStepLatents encode(const runtime::AudioBuffer & audio, uint32_t seed, const std::string & noise_file = {});
    void release_runtime_graphs() const;

private:
    class Impl;

    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::ace_step
