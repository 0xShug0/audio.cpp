#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include <array>
#include <filesystem>
#include <vector>

namespace engine::modules {
// Normalized SCRFD 512px detector and 106-point depthwise landmark graph builders.
// The caller owns weight storage, graph contexts, allocation and execution.
struct ScrfdNode {
    std::string op, output;
    std::vector<std::string> inputs;
    core::TensorShape shape;
    std::array<int, 4> axes{0, 1, 2, 3};
    Conv2dConfig convolution_config;
    Conv2dWeights convolution;
    core::TensorValue scale;
};

struct ScrfdWeights {
    std::string input;
    core::TensorShape input_shape;
    std::vector<ScrfdNode> nodes;
    std::vector<std::string> outputs;
    core::TensorValue negative_one;
};

ScrfdWeights load_scrfd_weights(core::BackendWeightStore & store, const assets::TensorSource & source,
                                const std::filesystem::path & config_path, assets::TensorStorageType type);

class ScrfdDetectorModule {
    public:
    // Nine heads: scores at strides 8/16/32, then boxes, then five-point landmarks.
    std::vector<core::TensorValue> build(core::ModuleBuildContext & ctx, const core::TensorValue & input,
                                         const ScrfdWeights & weights) const;
};

struct ScrfdFaceDetection {
    std::array<float, 4> box;
    float score;
    std::array<float, 10> landmarks;
};

std::vector<ScrfdFaceDetection> decode_scrfd_faces(const std::array<std::vector<float>, 9> & heads, float resize_scale,
                                                   float score_threshold = 0.5F, float nms_threshold = 0.4F);

struct Landmark106Layer {
    Conv2dConfig config;
    bool depthwise = false;
    Conv2dWeights convolution;
    core::TensorValue slope, positive_scale;
};

struct Landmark106Weights {
    core::TensorValue input_bias, input_scale;
    std::vector<Landmark106Layer> layers;
    LinearWeights output;
};

Landmark106Weights load_landmark106_weights(core::BackendWeightStore & store, const assets::TensorSource & source,
                                            const std::filesystem::path & config_path, assets::TensorStorageType type);

class Landmark106Module {
    public:
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & image,
                            const Landmark106Weights & weights) const;
};

} // namespace engine::modules
