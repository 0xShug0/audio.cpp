#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"

#include <filesystem>
#include <vector>

namespace engine::models::latentsync {

struct LatentSyncLandmarkLayer {
    modules::Conv2dConfig config;
    bool depthwise = false;
    modules::Conv2dWeights convolution;
    core::TensorValue slope, positive_scale;
};

struct LatentSyncLandmarkWeights {
    core::TensorValue input_bias, input_scale;
    std::vector<LatentSyncLandmarkLayer> layers;
    modules::LinearWeights output;
};

LatentSyncLandmarkWeights load_landmark_weights(core::BackendWeightStore & store,
    const assets::TensorSource & source, const std::filesystem::path & config_path,
    assets::TensorStorageType type);

class LatentSyncDepthwiseCnnLandmarks {
public:
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & image,
        const LatentSyncLandmarkWeights & weights) const;
};

}  // namespace engine::models::latentsync
