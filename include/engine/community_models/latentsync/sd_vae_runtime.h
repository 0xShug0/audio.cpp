#pragma once

#include "engine/community_models/latentsync/sd_vae.h"
#include "engine/framework/core/execution_context.h"
#include <memory>

namespace engine::models::latentsync {

class LatentSyncSdVaeRuntime {
public:
    LatentSyncSdVaeRuntime(core::ExecutionContext & execution, const assets::TensorSource & source,
        assets::TensorStorageType weight_type, int64_t resolution = 512,
        bool cuda_cudnn = false, bool cuda_f16_activations = false,
        bool share_graph_workspace = false);
    ~LatentSyncSdVaeRuntime();
    std::vector<float> encode_moments(const std::vector<float> & image);
    std::vector<float> encode(const std::vector<float> & image, const std::vector<float> & posterior_noise);
    std::vector<float> decode(const std::vector<float> & scaled_latent);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::latentsync
