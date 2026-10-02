#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/audio_flamingo/assets.h"
#include "engine/models/audio_flamingo/types.h"

#include <cstddef>
#include <memory>

namespace engine::models::audio_flamingo {

class AudioFlamingoAudioEncoderGraph;
struct AudioFlamingoAudioEncoderWeights;

class AudioFlamingoAudioEncoderRuntime {
public:
    AudioFlamingoAudioEncoderRuntime(
        std::shared_ptr<const AudioFlamingoAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        assets::TensorStorageType weight_storage_type);
    ~AudioFlamingoAudioEncoderRuntime();

    AudioFlamingoAudioEncoderOutput encode(const AudioFlamingoAudioFeatures & features);

private:
    std::shared_ptr<const AudioFlamingoAssets> assets_;
    std::shared_ptr<const AudioFlamingoAudioEncoderWeights> weights_;
    core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::unique_ptr<AudioFlamingoAudioEncoderGraph> graph_;
};

}  // namespace engine::models::audio_flamingo
