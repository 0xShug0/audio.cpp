#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/audio_flamingo/assets.h"
#include "engine/models/audio_flamingo/types.h"

#include <cstddef>
#include <memory>

namespace engine::models::audio_flamingo {

class AudioFlamingoAudioProjectorGraph;
struct AudioFlamingoAudioProjectorWeights;

class AudioFlamingoAudioProjectorRuntime {
public:
    AudioFlamingoAudioProjectorRuntime(
        std::shared_ptr<const AudioFlamingoAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        assets::TensorStorageType weight_storage_type);
    ~AudioFlamingoAudioProjectorRuntime();

    AudioFlamingoAudioProjectorOutput project(
        const AudioFlamingoAudioEncoderOutput & encoded,
        const AudioFlamingoAudioFeatures & features,
        const AudioFlamingoPrompt & prompt);

private:
    std::shared_ptr<const AudioFlamingoAssets> assets_;
    std::shared_ptr<const AudioFlamingoAudioProjectorWeights> weights_;
    core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::unique_ptr<AudioFlamingoAudioProjectorGraph> graph_;
};

}  // namespace engine::models::audio_flamingo
