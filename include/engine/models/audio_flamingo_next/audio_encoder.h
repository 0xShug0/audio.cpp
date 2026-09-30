#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/audio_flamingo_next/assets.h"
#include "engine/models/audio_flamingo_next/types.h"

#include <cstddef>
#include <memory>

namespace engine::models::audio_flamingo_next {

class AFNextAudioEncoderGraph;
struct AFNextAudioEncoderWeights;

class AFNextAudioEncoderRuntime {
public:
    AFNextAudioEncoderRuntime(
        std::shared_ptr<const AFNextAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        assets::TensorStorageType weight_storage_type);
    ~AFNextAudioEncoderRuntime();

    AFNextAudioEncoderOutput encode(const AFNextAudioFeatures & features);

private:
    std::shared_ptr<const AFNextAssets> assets_;
    std::shared_ptr<const AFNextAudioEncoderWeights> weights_;
    core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::unique_ptr<AFNextAudioEncoderGraph> graph_;
};

}  // namespace engine::models::audio_flamingo_next
