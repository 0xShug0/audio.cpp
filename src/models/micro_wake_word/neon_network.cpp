#include "engine/models/micro_wake_word/neon_network.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace engine::models::micro_wake_word {
namespace {
using Value = neon::ExecutionPlan::Value;

neon::Shape shape(const NetworkTensorInfo &info) {
    const auto &s = info.tflite_shape;
    if (s.size() == 4 && s[0] == 1 && s[2] == 1) return neon::Shape({int(s[3]), int(s[1])});
    if (s.size() == 3 && s[0] == 1) return neon::Shape({int(s[2]), int(s[1])});
    if (s.size() == 2 && s[0] == 1) return neon::Shape({int(s[1]), 1});
    if (s.size() == 1) return neon::Shape({int(s[0]), 1});
    throw std::runtime_error("microWakeWord NEON requires batch one and a unit spatial width");
}

int axis(int original, size_t rank) {
    if (original < 0) original += int(rank);
    if (original == 1 && rank >= 3) return 1;
    if (original == int(rank) - 1) return 0;
    throw std::runtime_error("microWakeWord NEON operator axis is not a channel or time axis");
}

bool same_quantization(const NetworkTensorInfo &a, const NetworkTensorInfo &b) {
    return a.type == b.type && a.scale == b.scale && a.zero_point == b.zero_point &&
        a.per_channel_quantized == b.per_channel_quantized;
}
} // namespace

NeonNetwork::NeonNetwork(const assets::TensorSource &source, const io::json::Value &config,
                         const std::vector<NetworkTensorInfo> &tensors, int threads)
    : threads_(threads) {
    namespace json = io::json;
    const int input_index = json::require_i32(config, "input_tensor");
    const int output_index = json::require_i32(config, "output_tensor");
    input_shape_ = shape(tensors.at(input_index));
    std::unordered_map<int, Value> values;
    values.emplace(input_index, plan_.input());
    std::vector<int> state_assigns;
    for (const auto &state : config.require("states").as_array()) {
        const auto read = json::require_i32(state, "read");
        const auto state_shape = shape(tensors.at(read));
        states_.push_back({state_shape, std::vector<float>(state_shape.elements(), 0.0F)});
        values.emplace(read, plan_.input());
        state_assigns.push_back(json::require_i32(state, "assign"));
    }
    inputs_.reserve(1 + states_.size());

    const auto quantize = [&](Value value, int index) {
        const auto &info = tensors.at(index);
        if (info.per_channel_quantized) throw std::runtime_error("microWakeWord NEON activation quantization must be per-tensor");
        if (!info.scale) return value;
        if (info.type != "i8" && info.type != "u8") throw std::runtime_error("microWakeWord NEON activation must use i8/u8 quantization");
        return plan_.add(neon::QuantizeDequantize(*info.scale, int(info.zero_point),
            info.type == "i8" ? -128 : 0, info.type == "i8" ? 127 : 255), {value});
    };
    const auto finish = [&](Value value, int index, const std::string &activation) {
        if (activation == "relu") value = plan_.add(neon::Clamp(0, std::numeric_limits<float>::infinity()), {value});
        else if (activation == "relu6") value = plan_.add(neon::Clamp(0, 6), {value});
        else if (activation != "none") throw std::runtime_error("microWakeWord NEON activation is unsupported");
        values.insert_or_assign(index, quantize(value, index));
    };
    const auto passthrough = [&](Value value, int input, int output) {
        if (!same_quantization(tensors.at(input), tensors.at(output))) {
            throw std::runtime_error("microWakeWord NEON view operator changes quantization");
        }
        values.insert_or_assign(output, value);
    };

    for (const auto &op : config.require("operators").as_array()) {
        const auto type = json::require_string(op, "type");
        const auto in = json::require_i64_array(op, "inputs");
        const auto out = json::require_i64_array(op, "outputs");
        auto value = values.at(int(in.at(0)));
        const int input = int(in[0]), output = int(out.at(0));
        if (type == "reshape" || type == "expand_dims") {
            const auto &before = tensors.at(input).tflite_shape;
            const auto &after = tensors.at(output).tflite_shape;
            // TFLite flattens NHWC storage, whereas the native network uses C,T.
            if (before.size() == 4 && after.size() == 2) value = plan_.add(neon::Transpose2d(), {value});
            if (!(shape(tensors.at(input)) == shape(tensors.at(output))) || (before.size() == 4 && after.size() == 2)) {
                value = plan_.add(neon::Reshape(shape(tensors.at(output))), {value});
            }
            passthrough(value, input, output);
        } else if (type == "concatenation") {
            if (json::require_string(op, "activation") != "none") throw std::runtime_error("microWakeWord NEON concat activation unsupported");
            const int dimension = axis(json::require_i32(op, "axis"), tensors.at(input).tflite_shape.size());
            for (size_t i = 0; i < in.size(); ++i) {
                if (!same_quantization(tensors.at(int(in[i])), tensors.at(output))) {
                    throw std::runtime_error("microWakeWord NEON concat changes quantization");
                }
                if (i) value = plan_.add(neon::Concat(dimension), {value, values.at(int(in[i]))});
            }
            values.insert_or_assign(output, value);
        } else if (type == "strided_slice") {
            const auto begin = json::require_i64_array(op, "begin"), end = json::require_i64_array(op, "end");
            const auto &source_shape = tensors.at(input).tflite_shape;
            for (size_t a = 0; a < begin.size(); ++a) {
                if (begin[a] || end[a] != source_shape.at(a)) {
                    value = plan_.add(neon::Slice(axis(int(a), begin.size()), int(begin[a]), int(end[a] - begin[a])), {value});
                }
            }
            passthrough(value, input, output);
        } else if (type == "split_v") {
            const auto sizes = json::require_i64_array(op, "sizes");
            if (sizes.size() != out.size()) throw std::runtime_error("microWakeWord NEON split size mismatch");
            const int dimension = axis(json::require_i32(op, "axis"), tensors.at(input).tflite_shape.size());
            int offset = 0;
            for (size_t i = 0; i < sizes.size(); ++i) {
                passthrough(plan_.add(neon::Slice(dimension, offset, int(sizes[i])), {value}), input, int(out[i]));
                offset += int(sizes[i]);
            }
        } else if (type == "conv_2d" || type == "depthwise_conv_2d") {
            const auto name = json::require_string(op, "weight");
            const auto dimensions = source.require_metadata(name).shape;
            const auto weights = source.require_f32(name, dimensions);
            const auto bias = source.require_f32(json::require_string(op, "bias"), {dimensions.at(0)});
            const auto stride = json::require_i64_array(op, "stride"), dilation = json::require_i64_array(op, "dilation");
            const auto input_shape = shape(tensors.at(input)), output_shape = shape(tensors.at(output));
            if (dimensions.size() != 4 || dimensions[3] != 1 || stride.at(1) != 1 || dilation.at(1) != 1) {
                throw std::runtime_error("microWakeWord NEON convolution must have unit spatial width");
            }
            const int padding_mode = json::require_i32(op, "padding");
            if (padding_mode != 0 && padding_mode != 1) throw std::runtime_error("microWakeWord NEON convolution padding invalid");
            const int total_padding = padding_mode == 1 ? 0 : std::max<int64_t>(0,
                (output_shape.dimensions[1] - 1) * stride[0] + dilation[0] * (dimensions[2] - 1) + 1 - input_shape.dimensions[1]);
            if (total_padding % 2) throw std::runtime_error("microWakeWord NEON asymmetric SAME padding unsupported");
            neon::Conv1dConfig c{input_shape.dimensions[0], output_shape.dimensions[0], int(dimensions[2]),
                int(stride[0]), int(dilation[0]), total_padding / 2, total_padding / 2};
            if (type == "depthwise_conv_2d") {
                value = plan_.add(neon::DepthwiseConv1d(c, weights, bias), {value});
            } else if (!total_padding && dilation[0] == 1 && input_shape.dimensions[1] == c.kernel_size && output_shape.dimensions[1] == 1) {
                // A valid convolution with one output is exactly one matrix-vector product.
                value = plan_.add(neon::Reshape(neon::Shape({c.input_channels * c.kernel_size, 1})), {value});
                value = plan_.add(neon::Linear(c.input_channels * c.kernel_size, c.output_channels, weights, bias), {value});
            } else {
                value = plan_.add(neon::Conv1d(c, weights, bias), {value});
            }
            finish(value, output, json::require_string(op, "activation"));
        } else if (type == "fully_connected") {
            const auto name = json::require_string(op, "weight");
            const auto dimensions = source.require_metadata(name).shape;
            value = plan_.add(neon::Linear(int(dimensions.at(1)), int(dimensions.at(0)), source.require_f32(name, dimensions),
                source.require_f32(json::require_string(op, "bias"), {dimensions[0]})), {value});
            finish(value, output, json::require_string(op, "activation"));
        } else if (type == "logistic") {
            finish(plan_.add(neon::Activation(neon::ActivationKind::Sigmoid), {value}), output, "none");
        } else if (type == "quantize") {
            finish(value, output, "none");
        } else throw std::runtime_error("microWakeWord NEON operator unsupported: " + type);
    }

    for (size_t i = 0; i < state_assigns.size(); ++i) {
        const auto value = plan_.add(neon::Reshape(states_[i].shape), {values.at(state_assigns[i])});
        plan_.retain(value);
        state_outputs_.push_back(value);
    }
    plan_.set_output(values.at(output_index));
    if (tensors.at(output_index).type == "u8" && tensors[output_index].scale) output_scale_ = 256.0F / 255.0F;
}

float NeonNetwork::infer(const std::vector<float> &input) {
    inputs_.clear();
    inputs_.emplace_back(input, input_shape_);
    for (const auto &state : states_) inputs_.emplace_back(state.values, state.shape);
    const auto &output = plan_.run(inputs_, threads_);
    if (output.values.size() != 1) throw std::runtime_error("microWakeWord NEON probability shape invalid");
    // Copy all retained new state only after every consumer of the old state has finished.
    for (size_t i = 0; i < states_.size(); ++i) {
        const auto view = plan_.view(state_outputs_[i]);
        std::copy_n(view.data, view.size, states_[i].values.data());
    }
    return output.values[0] * output_scale_;
}

void NeonNetwork::reset() {
    for (auto &state : states_) std::fill(state.values.begin(), state.values.end(), 0.0F);
}
} // namespace engine::models::micro_wake_word
