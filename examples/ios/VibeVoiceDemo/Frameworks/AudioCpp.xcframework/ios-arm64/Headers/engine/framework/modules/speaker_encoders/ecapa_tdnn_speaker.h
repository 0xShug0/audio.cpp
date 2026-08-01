#pragma once

#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::modules::ecapa_tdnn {

struct BatchNorm1dWeights {
    std::vector<float> weight;
    std::vector<float> bias;
    std::vector<float> running_mean;
    std::vector<float> running_var;
};

struct Conv1dWeights {
    std::string weight_name;
    std::vector<int64_t> weight_source_shape;
    int64_t out_channels = 0;
    int64_t in_channels = 0;
    int64_t kernel = 0;
    int64_t stride = 1;
    int64_t padding = 0;
    int64_t dilation = 1;
    bool use_bias = true;
    std::optional<std::string> bias_name;
};

struct TDNNBlockWeights {
    Conv1dWeights conv;
    BatchNorm1dWeights norm;
};

struct Res2NetBlockWeights {
    std::vector<TDNNBlockWeights> blocks;
    int64_t scale = 8;
    int64_t width = 0;
};

struct SEBlockWeights {
    Conv1dWeights conv1;
    Conv1dWeights conv2;
};

struct SERes2NetBlockWeights {
    TDNNBlockWeights tdnn1;
    Res2NetBlockWeights res2net;
    TDNNBlockWeights tdnn2;
    SEBlockWeights se;
    bool use_shortcut = false;
    Conv1dWeights shortcut;
};

struct AspWeights {
    TDNNBlockWeights tdnn;
    Conv1dWeights conv;
};

struct EcapaWeights {
    std::shared_ptr<const assets::TensorSource> source;
    TDNNBlockWeights block0;
    std::vector<SERes2NetBlockWeights> se_blocks;
    TDNNBlockWeights mfa;
    AspWeights asp;
    BatchNorm1dWeights asp_bn;
    Conv1dWeights fc;
    std::vector<float> classifier_weight;
    std::vector<float> embedding_global_mean;
};

}  // namespace engine::modules::ecapa_tdnn
