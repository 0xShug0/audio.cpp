#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/audio_flamingo_next/assets.h"
#include "engine/models/audio_flamingo_next/types.h"

#include <cstddef>
#include <memory>

namespace engine::models::audio_flamingo_next {

class AFNextAudioProjectorGraph;
struct AFNextAudioProjectorWeights;

class AFNextAudioProjectorRuntime {
public:
    AFNextAudioProjectorRuntime(
        std::shared_ptr<const AFNextAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        assets::TensorStorageType weight_storage_type);
    ~AFNextAudioProjectorRuntime();

    AFNextAudioProjectorOutput project(
        const AFNextAudioEncoderOutput & encoded,
        const AFNextAudioFeatures & features,
        const AFNextPrompt & prompt);

private:
    std::shared_ptr<const AFNextAssets> assets_;
    std::shared_ptr<const AFNextAudioProjectorWeights> weights_;
    core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::unique_ptr<AFNextAudioProjectorGraph> graph_;
};

}  // namespace engine::models::audio_flamingo_next
