#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/ace_step/assets.h"
#include "engine/models/ace_step/dit_weights_runtime.h"
#include "engine/models/ace_step/types.h"

#include <memory>
#include <vector>

namespace engine::models::ace_step {

class AceStepCoverTokenizerRuntime {
public:
    AceStepCoverTokenizerRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const AceStepAssets> assets,
        assets::TensorStorageType storage_type = assets::TensorStorageType::Native);
    ~AceStepCoverTokenizerRuntime();

    std::vector<int32_t> encode_audio_codes(
        const AceStepLatents & latents,
        const std::vector<float> & silence_latent,
        int64_t silence_frames,
        int64_t silence_channels) const;
    void release_runtime_graphs() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::ace_step
