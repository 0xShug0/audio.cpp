#pragma once

#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/text_encoders/t5_gemma_encoder.h"
#include "engine/framework/modules/text_encoders/t5_encoder_runtime.h"
#include "engine/models/stable_audio/assets.h"
#include "engine/models/stable_audio/conditioner.h"

#include <ggml-backend.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::core {
class BackendWeightStore;
}

namespace engine::models::stable_audio {

struct StableAudioT5Weights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::T5GemmaEncoderWeights encoder;
};

StableAudioT5Weights load_stable_audio_t5_weights(
    const StableAudioAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type);

struct StableAudioT5Encoding {
    int64_t batch = 0;
    int64_t tokens = 0;
    int64_t hidden_size = 0;
    std::vector<float> values;
};

class StableAudioT5EncoderRuntime {
public:
    StableAudioT5EncoderRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const StableAudioAssets> assets,
        assets::TensorStorageType weight_storage_type);
    ~StableAudioT5EncoderRuntime();

    StableAudioT5EncoderRuntime(const StableAudioT5EncoderRuntime &) = delete;
    StableAudioT5EncoderRuntime & operator=(const StableAudioT5EncoderRuntime &) = delete;

    void prepare() const;
    StableAudioT5Encoding encode(const StableAudioTokenBatch & tokens) const;
    void release_runtime_graphs() const;

private:
    core::ExecutionContext * execution_ = nullptr;
    std::shared_ptr<const StableAudioAssets> assets_;
    assets::TensorStorageType weight_storage_type_ = assets::TensorStorageType::Native;
    mutable std::unique_ptr<modules::T5EncoderRuntime> encoder_;
};

}  // namespace engine::models::stable_audio
