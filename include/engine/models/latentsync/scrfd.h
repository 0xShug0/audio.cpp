#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/conv_modules.h"

#include <filesystem>
#include <array>
#include <vector>

namespace engine::models::latentsync {

struct LatentSyncScrfdNode {
    std::string op, output;
    std::vector<std::string> inputs;
    core::TensorShape shape;
    std::array<int, 4> axes{0, 1, 2, 3};
    modules::Conv2dConfig convolution_config;
    modules::Conv2dWeights convolution;
    core::TensorValue scale;
};

struct LatentSyncScrfdWeights {
    std::string input;
    core::TensorShape input_shape;
    std::vector<LatentSyncScrfdNode> nodes;
    std::vector<std::string> outputs;
    core::TensorValue negative_one;
};

LatentSyncScrfdWeights load_scrfd_weights(core::BackendWeightStore & store,
    const assets::TensorSource & source, const std::filesystem::path & config_path,
    assets::TensorStorageType type);

class LatentSyncScrfdDetector {
public:
    // Nine heads: scores at strides 8/16/32, then boxes, then five-point landmarks.
    std::vector<core::TensorValue> build(core::ModuleBuildContext & ctx,
        const core::TensorValue & input, const LatentSyncScrfdWeights & weights) const;
};

struct LatentSyncFaceDetection {
    std::array<float, 4> box;
    float score;
    std::array<float, 10> landmarks;
};

std::vector<LatentSyncFaceDetection> decode_scrfd_faces(
    const std::array<std::vector<float>, 9> & heads, float resize_scale,
    float score_threshold = 0.5F, float nms_threshold = 0.4F);

}  // namespace engine::models::latentsync
