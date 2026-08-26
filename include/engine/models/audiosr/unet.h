#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/models/audiosr/assets.h"
#include "engine/models/audiosr/autoencoder.h"

#include <memory>
#include <vector>

namespace engine::models::audiosr {

class AudioSRUnetRuntime {
public:
    AudioSRUnetRuntime(
        std::shared_ptr<const AudioSRAssets> assets,
        engine::core::ExecutionContext & execution,
        engine::assets::TensorStorageType weight_type);
    ~AudioSRUnetRuntime();

    std::vector<float> predict_v(
        const AudioSRLatent & noisy,
        int64_t timestep,
        const AudioSRLatent & condition);

    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::audiosr
