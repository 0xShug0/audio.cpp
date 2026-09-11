#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/community_models/audio8_tts/assets.h"
#include "engine/community_models/audio8_tts/types.h"

#include <memory>

namespace engine::models::audio8_tts {

class Audio8TtsARRuntime {
public:
    Audio8TtsARRuntime(
        std::shared_ptr<const Audio8TtsAssets> assets,
        core::BackendConfig backend,
        int threads,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type);
    ~Audio8TtsARRuntime();

    Audio8TtsCodes generate(const Audio8TtsPrompt & prompt, const Audio8TtsGenerationOptions & options);
    void release_runtime_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::audio8_tts
