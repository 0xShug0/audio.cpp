#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/community_models/audio8_tts/assets.h"
#include "engine/community_models/audio8_tts/types.h"

#include <memory>

namespace engine::models::audio8_tts {

class Audio8TtsCodecRuntime {
public:
    Audio8TtsCodecRuntime(
        std::shared_ptr<const Audio8TtsAssets> assets,
        core::BackendConfig backend,
        int threads,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType matmul_weight_storage_type,
        assets::TensorStorageType conv_weight_storage_type);
    ~Audio8TtsCodecRuntime();

    Audio8TtsCodes encode_reference(const runtime::AudioBuffer & audio);
    runtime::AudioBuffer decode(const Audio8TtsCodes & codes);
    void release_encode_graph();
    void release_runtime_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::audio8_tts
