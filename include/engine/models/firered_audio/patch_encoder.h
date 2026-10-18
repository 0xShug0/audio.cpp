#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/firered_audio/assets.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::models::firered_audio {

class FireRedAudioPatchEncoderRuntime {
public:
    FireRedAudioPatchEncoderRuntime(
        std::shared_ptr<const FireRedAudioAssets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type);
    ~FireRedAudioPatchEncoderRuntime();

    FireRedAudioPatchEncoderRuntime(const FireRedAudioPatchEncoderRuntime &) = delete;
    FireRedAudioPatchEncoderRuntime & operator=(const FireRedAudioPatchEncoderRuntime &) = delete;

    std::vector<float> encode(const std::vector<float> & latents);
    void release_graph();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::firered_audio
