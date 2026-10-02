#pragma once

#include "engine/community_models/latentsync/unet.h"
#include "engine/community_models/latentsync/sampling.h"
#include "engine/framework/core/execution_context.h"

#include <memory>

namespace engine::models::latentsync {

struct LatentSyncLatentClip {
    int64_t frames = 0;
    int64_t height = 0;
    int64_t width = 0;
    int64_t audio_tokens = 50;
    // All image tensors are frame-major NCHW. Noise and image latents have four channels; mask has one.
    std::vector<float> noise, mask, masked_image, reference_image, audio;
};

class LatentSyncUnetRuntime {
public:
    LatentSyncUnetRuntime(core::ExecutionContext & execution, const assets::TensorSource & source,
        LatentSyncUnetConfig config, assets::TensorStorageType weight_type);
    ~LatentSyncUnetRuntime();

    std::vector<float> predict(const std::vector<float> & input, const std::vector<float> & audio,
        const std::vector<float> & timestep_features, int64_t batch, int64_t frames,
        int64_t height, int64_t width, int64_t audio_tokens, int64_t cache_step = -1,
        int64_t cache_slot = 0);

    std::vector<float> denoise(const LatentSyncLatentClip & clip, int64_t steps,
        float guidance_scale, LatentSyncDdimConfig schedule_config = {}, int64_t cache_interval = 1);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::latentsync
