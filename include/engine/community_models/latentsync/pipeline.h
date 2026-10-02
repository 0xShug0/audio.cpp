#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/assets/tensor_source.h"
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

namespace engine::models::latentsync {

struct LatentSyncPipelineSources {
    std::shared_ptr<const assets::TensorSource> unet, vae, whisper, detector, landmarks;
    std::filesystem::path detector_config, landmark_config;
};

struct LatentSyncGenerationOptions {
    int64_t steps = 20;
    int64_t cache_interval = 1;
    float guidance_scale = 1.5F;
    int64_t frames_per_chunk = 16;
    double fps = 25;
    uint64_t seed = 1247;
};

class LatentSyncPipeline {
public:
    LatentSyncPipeline(core::ExecutionContext & execution, const LatentSyncPipelineSources & sources,
        assets::TensorStorageType weight_type, bool vae_cudnn_f16 = false,
        bool unet_cudnn_f16 = false, bool mem_saver = false);
    ~LatentSyncPipeline();
    // Frames and callback pixels are RGB interleaved bytes; mask is RGB planar 512x512 in [0,1].
    void run(const std::vector<std::vector<uint8_t>> & frames, int height, int width,
        const std::vector<float> & audio_16khz, const std::vector<float> & mask,
        const LatentSyncGenerationOptions & options,
        const std::function<void(size_t, const std::vector<uint8_t> &)> & emit);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::latentsync
