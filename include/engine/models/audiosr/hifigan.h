#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/audiosr/assets.h"

#include <memory>
#include <vector>

namespace engine::models::audiosr {

class AudioSRHiFiGanRuntime {
public:
    AudioSRHiFiGanRuntime(
        std::shared_ptr<const AudioSRAssets> assets,
        engine::core::ExecutionContext & execution,
        engine::assets::TensorStorageType weight_type);
    ~AudioSRHiFiGanRuntime();

    engine::runtime::AudioBuffer synthesize(const std::vector<float> & mel, int64_t frames);
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::audiosr
