#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/fireredtts3/assets.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::models::fireredtts3 {

std::vector<float> prepare_firered_prompt_audio_24k(
    const engine::runtime::AudioBuffer & audio,
    const FireRedTTS3RedAeConfig & config,
    int64_t patch_size);

class FireRedRedAeRuntime {
public:
    FireRedRedAeRuntime(
        std::shared_ptr<const FireRedTTS3Assets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type);
    ~FireRedRedAeRuntime();

    FireRedRedAeRuntime(const FireRedRedAeRuntime &) = delete;
    FireRedRedAeRuntime & operator=(const FireRedRedAeRuntime &) = delete;

    std::vector<float> encode(const std::vector<float> & audio_24k);
    engine::runtime::AudioBuffer decode(const std::vector<float> & latents);
    void release_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::fireredtts3
