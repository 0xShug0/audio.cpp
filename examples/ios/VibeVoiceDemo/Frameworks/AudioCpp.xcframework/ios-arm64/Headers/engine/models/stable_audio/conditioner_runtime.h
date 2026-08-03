#pragma once

#include "engine/models/stable_audio/assets.h"
#include "engine/models/stable_audio/conditioner.h"
#include "engine/models/stable_audio/t5gemma.h"

#include <memory>
#include <vector>

namespace engine::core {
class ExecutionContext;
}

namespace engine::models::stable_audio {

struct StableAudioConditionedInput {
    int64_t batch = 0;
    int64_t cross_tokens = 0;
    int64_t cond_dim = 0;
    std::vector<float> cross_attention;
    std::vector<uint8_t> cross_attention_mask;
    std::vector<float> global;
};

struct StableAudioConditioningInputs {
    StableAudioConditionedInput positive;
    StableAudioConditionedInput negative;
    bool has_negative = false;
};

class StableAudioConditionerRuntime {
public:
    StableAudioConditionerRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const StableAudioAssets> assets,
        assets::TensorStorageType weight_storage_type,
        int64_t max_batch);
    ~StableAudioConditionerRuntime();

    StableAudioConditionerRuntime(const StableAudioConditionerRuntime &) = delete;
    StableAudioConditionerRuntime & operator=(const StableAudioConditionerRuntime &) = delete;

    void prepare() const;
    StableAudioConditioningInputs encode(const StableAudioConditioningBatch & inputs) const;
    void release_runtime_graphs() const;

private:
    StableAudioConditionedInput encode_one(
        const StableAudioTokenBatch & tokens,
        const std::vector<float> & normalized_seconds,
        const std::vector<float> & seconds_fourier_features) const;

    std::shared_ptr<const StableAudioAssets> assets_;
    StableAudioT5EncoderRuntime t5_;
    std::vector<float> prompt_padding_embedding_;
    std::vector<float> seconds_projection_weight_;
    std::vector<float> seconds_projection_bias_;
};

}  // namespace engine::models::stable_audio
