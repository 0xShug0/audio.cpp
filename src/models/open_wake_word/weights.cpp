#include "engine/models/open_wake_word/weights.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/json.h"
#include "engine/models/open_wake_word/graph_weights.h"

#include <limits>
#include <stdexcept>

namespace engine::models::open_wake_word {
namespace {
int dimension(int64_t value) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("OpenWakeWord dimension outside native index range");
    }
    return static_cast<int>(value);
}
} // namespace

FrontendWeights load_frontend_weights(const assets::TensorSource &source) {
    return {source.require_f32("frontend.window", {512}), source.require_f32("frontend.mel_filter", {32, 257})};
}

std::shared_ptr<const Weights> load_weights(const assets::TensorSource &source, const io::json::Value &config) {
    auto weights = std::make_shared<Weights>();
    weights->frontend = load_frontend_weights(source);
    weights->initial_features = source.require_f32("runtime.initial_features", {16, 96});
    weights->suppression_frames = config.require("initial_suppression_frames").as_i64();
    for (const auto &node : config.require("embedding_graph").as_array()) {
        const auto &op = node.require("op").as_string();
        const auto &args = node.require("input").as_array();
        const auto &attrs = node.require("attributes");
        if (op == "Reshape") {
            continue;
        }
        if (op == "Conv") {
            if (args.size() != 2 && args.size() != 3) {
                throw std::runtime_error("invalid OpenWakeWord convolution inputs");
            }
            const auto name = "embedding." + args[1].as_string();
            const auto shape = source.require_metadata(name).shape;
            if (shape.size() != 4) {
                throw std::runtime_error("invalid OpenWakeWord convolution rank");
            }
            if (const auto *group = attrs.find("group"); group && group->as_i64() != 1) {
                throw std::runtime_error("OpenWakeWord requires ungrouped convolution");
            }
            ConvLayer layer;
            auto &conv = layer.config;
            conv.output_channels = dimension(shape[0]);
            conv.input_channels = dimension(shape[1]);
            conv.kernel_height = dimension(shape[2]);
            conv.kernel_width = dimension(shape[3]);
            const auto &strides = attrs.require("strides").as_array();
            const auto &dilation = attrs.require("dilations").as_array();
            if (strides.size() != 2 || dilation.size() != 2) {
                throw std::runtime_error("invalid convolution geometry");
            }
            conv.stride_height = dimension(strides[0].as_i64());
            conv.stride_width = dimension(strides[1].as_i64());
            conv.dilation_height = dimension(dilation[0].as_i64());
            conv.dilation_width = dimension(dilation[1].as_i64());
            if (const auto *pads = attrs.find("pads")) {
                const auto &values = pads->as_array();
                if (values.size() != 4 || values[0].as_i64() != values[2].as_i64() ||
                    values[1].as_i64() != values[3].as_i64() || values[0].as_i64() < 0 || values[1].as_i64() < 0 ||
                    values[0].as_i64() > std::numeric_limits<int>::max() ||
                    values[1].as_i64() > std::numeric_limits<int>::max()) {
                    throw std::runtime_error("invalid convolution padding");
                }
                conv.padding_height = static_cast<int>(values[0].as_i64());
                conv.padding_width = static_cast<int>(values[1].as_i64());
            }
            layer.weight = source.require_f32(name, shape);
            if (args.size() == 3) {
                layer.bias = source.require_f32("embedding." + args[2].as_string(), {shape[0]});
            }
            weights->embedding.emplace_back(std::move(layer));
        } else if (op == "LeakyRelu") {
            weights->embedding.emplace_back(LeakyReluLayer{attrs.require("alpha").as_f32()});
        } else if (op == "Max") {
            if (args.size() != 2) {
                throw std::runtime_error("invalid OpenWakeWord activation bound");
            }
            const auto bound = source.require_f32("embedding." + args[1].as_string());
            if (bound.size() != 1) {
                throw std::runtime_error("OpenWakeWord requires scalar activation bound");
            }
            weights->embedding.emplace_back(ClampLayer{bound[0]});
        } else if (op == "MaxPool") {
            const auto &kernel = attrs.require("kernel_shape").as_array();
            const auto &strides = attrs.require("strides").as_array();
            if (kernel.size() != 2 || strides.size() != 2) {
                throw std::runtime_error("invalid OpenWakeWord pooling geometry");
            }
            weights->embedding.emplace_back(PoolLayer{dimension(kernel[0].as_i64()), dimension(kernel[1].as_i64()),
                                                      dimension(strides[0].as_i64()), dimension(strides[1].as_i64())});
        } else {
            throw std::runtime_error("unsupported OpenWakeWord embedding op: " + op);
        }
    }
    int features = 1536;
    for (const auto &node : config.require("classifier_graph").as_array()) {
        const auto &op = node.require("op").as_string();
        if (op == "Flatten") {
            continue;
        }
        if (op == "Gemm") {
            const auto &args = node.require("input").as_array();
            if (args.size() != 3) {
                throw std::runtime_error("invalid OpenWakeWord classifier inputs");
            }
            const auto name = "classifier." + args[1].as_string();
            const auto shape = source.require_metadata(name).shape;
            if (shape.size() != 2 || shape[1] != features) {
                throw std::runtime_error("classifier weight shape mismatch");
            }
            LinearLayer layer{dimension(shape[1]), dimension(shape[0]), source.require_f32(name, shape),
                              source.require_f32("classifier." + args[2].as_string(), {shape[0]})};
            features = layer.output_features;
            weights->classifier.emplace_back(std::move(layer));
        } else if (op == "Relu") {
            weights->classifier.emplace_back(ReluLayer{});
        } else if (op == "Sigmoid") {
            weights->classifier.emplace_back(SigmoidLayer{});
        } else {
            throw std::runtime_error("unsupported OpenWakeWord classifier op: " + op);
        }
    }
    if (features != 1) {
        throw std::runtime_error("OpenWakeWord classifier must output one probability");
    }
    return weights;
}

GraphWeights bind_graph_weights(const Weights &weights, const core::ExecutionContext &execution, bool embedding) {
    GraphWeights bound;
    bound.store = std::make_shared<core::BackendWeightStore>(execution.backend(), execution.backend_type(),
                                                             "open_wake_word", 1024 * 1024);
    bound.embedding.resize(weights.embedding.size());
    for (size_t i = 0; embedding && i < weights.embedding.size(); ++i) {
        if (const auto *conv = std::get_if<ConvLayer>(&weights.embedding[i])) {
            const auto &c = conv->config;
            modules::Conv2dWeights layer;
            layer.weight = bound.store->make_f32(
                core::TensorShape::from_dims({c.output_channels, c.input_channels, c.kernel_height, c.kernel_width}),
                conv->weight);
            if (!conv->bias.empty()) {
                layer.bias = bound.store->make_f32(core::TensorShape::from_dims({c.output_channels}), conv->bias);
            }
            bound.embedding[i] = std::move(layer);
        }
    }
    bound.classifier.resize(weights.classifier.size());
    for (size_t i = 0; i < weights.classifier.size(); ++i) {
        if (const auto *linear = std::get_if<LinearLayer>(&weights.classifier[i])) {
            bound.classifier[i] = modules::LinearWeights{
                bound.store->make_f32(core::TensorShape::from_dims({linear->output_features, linear->input_features}),
                                      linear->weight),
                bound.store->make_f32(core::TensorShape::from_dims({linear->output_features}), linear->bias)};
        }
    }
    bound.store->upload();
    return bound;
}

} // namespace engine::models::open_wake_word
