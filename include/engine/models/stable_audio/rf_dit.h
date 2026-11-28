#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/models/stable_audio/assets.h"
#include "engine/models/stable_audio/conditioner_runtime.h"
#include "engine/models/stable_audio/sampler.h"

#include <ggml-backend.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace engine::core {
class BackendWeightStore;
class ExecutionContext;
}

namespace engine::modules { class DitRuntime; }

namespace engine::models::stable_audio {

class StableAudioRFDiTRuntime {
public:
    StableAudioRFDiTRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const StableAudioAssets> assets,
        assets::TensorStorageType weight_storage_type);
    ~StableAudioRFDiTRuntime();

    StableAudioRFDiTRuntime(const StableAudioRFDiTRuntime &) = delete;
    StableAudioRFDiTRuntime & operator=(const StableAudioRFDiTRuntime &) = delete;

    void prepare(const StableAudioSamplingState & sampling, float cfg_scale) const;
    std::vector<float> sample(
        const StableAudioSamplingState & sampling,
        const StableAudioConditioningInputs & conditioning,
        uint64_t seed,
        uint64_t & rng_offset_blocks,
        const engine::sampling::TorchCudaSamplingPolicy & rng_policy,
        std::string_view sampler_type,
        float cfg_scale,
        float apg_scale) const;
    void release_runtime_graphs() const;

private:
    core::ExecutionContext * execution_ = nullptr;
    std::shared_ptr<const StableAudioAssets> assets_;
    assets::TensorStorageType weight_storage_type_ = assets::TensorStorageType::Native;
    mutable std::unique_ptr<modules::DitRuntime> runtime_;
};

}  // namespace engine::models::stable_audio
