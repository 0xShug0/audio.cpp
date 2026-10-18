#pragma once

#include "engine/community_models/mira_tts/assets.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::mira_tts {

class MiraSnakeConvDecoder final {
public:
    MiraSnakeConvDecoder(
        const MiraTTSAssets & assets,
        core::ExecutionContext & execution,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        assets::TensorStorageType storage_type);
    ~MiraSnakeConvDecoder();

    runtime::AudioBuffer decode(
        const std::vector<float> & latents,
        int64_t frames);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::mira_tts
