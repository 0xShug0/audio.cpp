#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/models/fish_audio/assets.h"
#include "engine/models/fish_audio/types.h"

#include <memory>

namespace engine::models::fish_audio {

class FishAudioCodecRuntime {
public:
    FishAudioCodecRuntime(
        std::shared_ptr<const FishAudioAssets> assets,
        core::BackendConfig backend,
        int threads,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType matmul_weight_storage_type,
        assets::TensorStorageType conv_weight_storage_type);
    ~FishAudioCodecRuntime();

    FishAudioCodes encode_reference(const runtime::AudioBuffer & audio);
    runtime::AudioBuffer decode(const FishAudioCodes & codes);
    void release_encode_graph();
    void release_runtime_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::fish_audio
