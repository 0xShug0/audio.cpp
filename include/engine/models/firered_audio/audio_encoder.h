#pragma once

#include "engine/framework/audio/dsp.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/model.h"
#include "engine/models/firered_audio/assets.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::models::firered_audio {

struct FireRedAudioUnderstandFeatures {
    std::vector<float> embeddings;
    int64_t tokens = 0;
};

class FireRedAudioAudioEncoderRuntime {
public:
    FireRedAudioAudioEncoderRuntime(
        std::shared_ptr<const FireRedAudioAssets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type);
    ~FireRedAudioAudioEncoderRuntime();

    FireRedAudioAudioEncoderRuntime(const FireRedAudioAudioEncoderRuntime &) = delete;
    FireRedAudioAudioEncoderRuntime & operator=(const FireRedAudioAudioEncoderRuntime &) = delete;

    FireRedAudioUnderstandFeatures encode(const engine::runtime::AudioBuffer & audio);
    void release_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::firered_audio

