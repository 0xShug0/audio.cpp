#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/text_encoders/t5_base_encoder.h"
#include "engine/models/stable_audio/assets.h"
#include "engine/models/stable_audio/conditioner.h"
#include "engine/models/stable_audio/conditioner_runtime.h"

#include <ggml-backend.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::core {
class BackendWeightStore;
class ExecutionContext;
}

namespace engine::models::stable_audio::foundation {

struct T5BaseRuntimeWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::T5BaseEncoderWeights encoder;
};

struct FoundationNumberConditionerWeights {
    std::vector<float> frequencies;
    std::vector<float> projection_weight;
    std::vector<float> projection_bias;
};

T5BaseRuntimeWeights load_t5_base_runtime_weights(
    const StableAudioAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type);

class FoundationConditionerRuntime {
public:
    FoundationConditionerRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const StableAudioAssets> assets,
        assets::TensorStorageType weight_storage_type,
        int64_t max_batch);
    ~FoundationConditionerRuntime();

    FoundationConditionerRuntime(const FoundationConditionerRuntime &) = delete;
    FoundationConditionerRuntime & operator=(const FoundationConditionerRuntime &) = delete;

    void prepare() const;
    StableAudioConditioningInputs encode(const StableAudioConditioningBatch & inputs) const;

private:
    class T5Graph;

    StableAudioConditionedInput encode_one(
        const StableAudioTokenBatch & tokens,
        const std::vector<float> & normalized_start_seconds,
        const std::vector<float> & normalized_total_seconds) const;

    std::shared_ptr<const StableAudioAssets> assets_;
    core::ExecutionContext * execution_ = nullptr;
    assets::TensorStorageType weight_storage_type_ = assets::TensorStorageType::Native;
    int64_t max_batch_ = 1;
    mutable std::unique_ptr<T5Graph> t5_graph_;
    FoundationNumberConditionerWeights seconds_start_;
    FoundationNumberConditionerWeights seconds_total_;
};

}  // namespace engine::models::stable_audio::foundation
