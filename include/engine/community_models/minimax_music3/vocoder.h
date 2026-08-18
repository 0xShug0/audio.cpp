#pragma once

#include "engine/community_models/minimax_music3/assets.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/model.h"

#include <memory>
#include <vector>

namespace engine::models::minimax_music3 {

class MiniMaxMusic3VocoderRuntime {
public:
    MiniMaxMusic3VocoderRuntime(
        std::shared_ptr<const MiniMaxMusic3Assets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type);
    ~MiniMaxMusic3VocoderRuntime();

    runtime::AudioBuffer decode(const std::vector<float> & latents, int64_t latent_frames);
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::minimax_music3

