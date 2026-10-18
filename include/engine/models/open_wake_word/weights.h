#pragma once

#include "engine/framework/neon/operators.h"

#include <memory>
#include <variant>
#include <vector>

namespace engine::assets {
class TensorSource;
}
namespace engine::io::json {
class Value;
}

namespace engine::models::open_wake_word {

struct FrontendWeights {
    std::vector<float> window;
    std::vector<float> mel_filter;
};

struct ConvLayer {
    neon::Conv2dConfig config;
    std::vector<float> weight;
    std::vector<float> bias;
};
struct LeakyReluLayer {
    float slope;
};
struct ClampLayer {
    float minimum;
};
struct PoolLayer {
    int height, width, stride_height, stride_width;
};
using EmbeddingLayer = std::variant<ConvLayer, LeakyReluLayer, ClampLayer, PoolLayer>;

struct LinearLayer {
    int input_features, output_features;
    std::vector<float> weight;
    std::vector<float> bias;
};
struct ReluLayer {};
struct SigmoidLayer {};
using ClassifierLayer = std::variant<LinearLayer, ReluLayer, SigmoidLayer>;

struct Weights {
    FrontendWeights frontend;
    std::vector<EmbeddingLayer> embedding;
    std::vector<ClassifierLayer> classifier;
    std::vector<float> initial_features;
    int suppression_frames;
};

// This layer imports model names and architecture metadata into typed data.
std::shared_ptr<const Weights> load_weights(const assets::TensorSource &source, const io::json::Value &config);
FrontendWeights load_frontend_weights(const assets::TensorSource &source);

} // namespace engine::models::open_wake_word
