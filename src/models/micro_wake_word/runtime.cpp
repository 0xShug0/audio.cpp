#include "engine/models/micro_wake_word/runtime.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace engine::models::micro_wake_word {
namespace json = engine::io::json;
namespace {

struct TensorInfo {
    std::vector<int64_t> tflite_shape;
    std::string type;
    std::optional<float> scale;
    int64_t zero_point = 0;
    bool per_channel_quantized = false;
};

struct StateInfo {
    std::string name;
    int read = -1;
    int assign = -1;
};

std::vector<int> ints(const json::Value & object, const std::string & key) {
    const auto values = json::require_i64_array(object, key);
    return std::vector<int>(values.begin(), values.end());
}

core::TensorShape native_shape(const std::vector<int64_t> & shape) {
    if (shape.size() == 4) {
        return core::TensorShape::from_dims({shape[0], shape[3], shape[1], shape[2]});
    }
    if (shape.size() == 3) {
        return core::TensorShape::from_dims({shape[0], shape[2], shape[1]});
    }
    if (shape.size() == 2) return core::TensorShape::from_dims({shape[0], shape[1]});
    if (shape.size() == 1) return core::TensorShape::from_dims({shape[0]});
    throw std::runtime_error("microWakeWord tensor rank is unsupported");
}

int native_axis(int axis, size_t rank) {
    if (axis < 0) axis += static_cast<int>(rank);
    if (axis < 0 || axis >= static_cast<int>(rank)) {
        throw std::runtime_error("microWakeWord operator axis is out of range");
    }
    if (rank == 4) {
        constexpr int axes[] = {0, 2, 3, 1};
        return axes[axis];
    }
    if (rank == 3) {
        constexpr int axes[] = {0, 2, 1};
        return axes[axis];
    }
    return axis;
}

core::TensorValue fake_quantize(
    core::ModuleBuildContext & ctx,
    core::TensorValue value,
    const TensorInfo & info) {
    if (info.per_channel_quantized) {
        throw std::runtime_error("microWakeWord activation uses unsupported per-channel quantization");
    }
    if (!info.scale.has_value()) return value;
    if (info.type != "i8" && info.type != "u8") {
        throw std::runtime_error("microWakeWord activation quantization requires i8 or u8 tensors");
    }
    value = core::ensure_backend_addressable_layout(ctx, value);
    const float scale = *info.scale;
    const float minimum = info.type == "i8" ? -128.0F : 0.0F;
    const float maximum = info.type == "i8" ? 127.0F : 255.0F;
    // zero_point is integral, so round(x / scale + zero_point) - zero_point
    // equals round(x / scale). Shift the clamp bounds instead of materializing
    // scalar add nodes.
    auto * quantized = ggml_scale(ctx.ggml, value.tensor, 1.0F / scale);
    quantized = ggml_round(ctx.ggml, quantized);
    quantized = ggml_clamp(ctx.ggml, quantized,
                           minimum - static_cast<float>(info.zero_point),
                           maximum - static_cast<float>(info.zero_point));
    quantized = ggml_scale(ctx.ggml, quantized, scale);
    return core::wrap_tensor(quantized, value.shape, GGML_TYPE_F32);
}

core::TensorValue apply_activation(
    core::ModuleBuildContext & ctx,
    core::TensorValue value,
    const std::string & activation) {
    if (activation == "none") return value;
    if (activation == "relu") return modules::ReluModule{}.build(ctx, value);
    if (activation == "relu6") {
        return core::wrap_tensor(ggml_clamp(ctx.ggml, value.tensor, 0.0F, 6.0F), value.shape, GGML_TYPE_F32);
    }
    throw std::runtime_error("unsupported microWakeWord activation: " + activation);
}

}  // namespace

struct Runtime::Impl {
    struct ContextDeleter { void operator()(ggml_context * value) const { ggml_free(value); } };
    struct AllocatorDeleter { void operator()(ggml_gallocr_t value) const { ggml_gallocr_free(value); } };

    ModelConfig model;
    core::ExecutionContext & execution;
    core::BackendWeightStore store;
    std::unique_ptr<ggml_context, ContextDeleter> context;
    std::unique_ptr<ggml_gallocr, AllocatorDeleter> allocator;
    core::HostGraphPlan plan;
    core::TensorValue input;
    core::TensorValue output;
    std::vector<core::TensorValue> state_inputs;
    std::vector<core::TensorValue> state_outputs;
    std::vector<std::vector<float>> state_values;
    float input_scale = 0.0F;
    int64_t input_zero_point = 0;
    ggml_cgraph * graph = nullptr;

    Impl(const assets::TensorSource & source, const json::Value & root,
         core::ExecutionContext & execution_context)
        : execution(execution_context),
          store(execution.backend(), execution.backend_type(), "micro_wake_word.weights", 1024 * 1024) {
        if (json::require_string(root, "format") != "micro_wake_word_tflite_v1") {
            throw std::runtime_error("unsupported microWakeWord conversion format");
        }
        model.sample_rate = json::require_i32(root, "sample_rate");
        model.feature_dim = json::require_i32(root, "feature_dim");
        model.input_frames = json::require_i32(root, "input_frames");
        model.sliding_window_size = json::require_i32(root, "sliding_window_size");
        model.probability_cutoff = json::require_f32(root, "probability_cutoff");
        model.wake_phrase = json::require_string(root, "wake_phrase");
        if (model.sample_rate != 16000 || model.feature_dim != 40 || model.input_frames <= 0 ||
            model.sliding_window_size <= 0 || model.wake_phrase.empty() ||
            !std::isfinite(model.probability_cutoff) || model.probability_cutoff < 0.0F ||
            model.probability_cutoff > 1.0F) {
            throw std::runtime_error("invalid microWakeWord model configuration");
        }

        std::vector<TensorInfo> infos;
        for (const auto & item : root.require("tensors").as_array()) {
            TensorInfo info;
            info.tflite_shape = json::require_i64_array(item, "shape");
            info.type = json::require_string(item, "type");
            if (const auto * quant = item.find("quantization"); quant != nullptr && !quant->is_null()) {
                const auto scales = json::require_i64_array(*quant, "zero_points");
                const auto scale_values = quant->require("scales").as_array();
                if (scale_values.size() == 1 && scales.size() == 1) {
                    info.scale = scale_values[0].as_f32();
                    info.zero_point = scales[0];
                } else {
                    info.per_channel_quantized = true;
                }
            }
            infos.push_back(std::move(info));
        }
        const int input_index = json::require_i32(root, "input_tensor");
        const int output_index = json::require_i32(root, "output_tensor");
        if (input_index < 0 || output_index < 0 || input_index >= static_cast<int>(infos.size()) ||
            output_index >= static_cast<int>(infos.size())) {
            throw std::runtime_error("microWakeWord input/output tensor index is invalid");
        }

        context.reset(ggml_init({8 * 1024 * 1024, nullptr, true}));
        if (!context) throw std::runtime_error("microWakeWord graph context allocation failed");
        core::ModuleBuildContext ctx{context.get(), "micro_wake_word", execution.backend_type()};
        input = core::make_tensor(ctx, GGML_TYPE_F32, native_shape(infos[input_index].tflite_shape));
        ggml_set_input(input.tensor);
        if (!infos[input_index].scale.has_value() || infos[input_index].per_channel_quantized ||
            infos[input_index].type != "i8") {
            throw std::runtime_error("microWakeWord input quantization metadata is unsupported");
        }
        input_scale = *infos[input_index].scale;
        input_zero_point = infos[input_index].zero_point;
        std::unordered_map<int, core::TensorValue> values;
        values.emplace(input_index, input);

        std::vector<StateInfo> states;
        for (const auto & item : root.require("states").as_array()) {
            StateInfo state{json::require_string(item, "name"), json::require_i32(item, "read"),
                            json::require_i32(item, "assign")};
            if (state.read < 0 || state.assign < 0 || state.read >= static_cast<int>(infos.size()) ||
                state.assign >= static_cast<int>(infos.size())) {
                throw std::runtime_error("microWakeWord state tensor index is invalid");
            }
            auto state_input = core::make_tensor(ctx, GGML_TYPE_F32, native_shape(infos[state.read].tflite_shape));
            ggml_set_input(state_input.tensor);
            state_inputs.push_back(state_input);
            state_values.emplace_back(static_cast<size_t>(state_input.shape.num_elements()), 0.0F);
            values.emplace(state.read, state_input);
            states.push_back(std::move(state));
        }

        const auto require_value = [&](int index) -> core::TensorValue {
            const auto it = values.find(index);
            if (it == values.end()) throw std::runtime_error("microWakeWord operator references unavailable tensor");
            return it->second;
        };
        const auto finish = [&](int index, core::TensorValue value) {
            values[index] = fake_quantize(ctx, value, infos[index]);
        };
        const auto passthrough = [&](int input_index_value, int output_index_value,
                                     core::TensorValue value) {
            const auto & source_info = infos[input_index_value];
            const auto & output_info = infos[output_index_value];
            if (source_info.type != output_info.type || source_info.scale != output_info.scale ||
                source_info.zero_point != output_info.zero_point ||
                source_info.per_channel_quantized != output_info.per_channel_quantized) {
                throw std::runtime_error("microWakeWord view operator changes quantization metadata");
            }
            values[output_index_value] = value;
        };

        for (const auto & op : root.require("operators").as_array()) {
            const std::string type = json::require_string(op, "type");
            const auto op_inputs = ints(op, "inputs");
            const auto op_outputs = ints(op, "outputs");
            if (op_outputs.empty()) throw std::runtime_error("microWakeWord operator has no output");
            if (type == "reshape" || type == "expand_dims") {
                auto value = require_value(op_inputs[0]);
                const auto target = infos[op_outputs[0]].tflite_shape;
                if (value.shape.rank == 4 && target.size() == 2) {
                    value = modules::TransposeModule({{0, 2, 3, 1}, 4}).build(ctx, value);
                    value = core::wrap_tensor(ggml_cont(ctx.ggml, value.tensor), value.shape, value.type);
                }
                passthrough(op_inputs[0], op_outputs[0],
                            core::reshape_tensor(ctx, value, native_shape(target)));
            } else if (type == "concatenation") {
                if (json::require_string(op, "activation") != "none") {
                    throw std::runtime_error("microWakeWord concat activation is unsupported");
                }
                auto value = require_value(op_inputs[0]);
                const int axis = native_axis(json::require_i32(op, "axis"), infos[op_inputs[0]].tflite_shape.size());
                for (size_t i = 1; i < op_inputs.size(); ++i) {
                    value = modules::ConcatModule({axis}).build(ctx, value, require_value(op_inputs[i]));
                }
                for (const int input_id : op_inputs) {
                    const auto & source_info = infos[input_id];
                    const auto & output_info = infos[op_outputs[0]];
                    if (source_info.type != output_info.type || source_info.scale != output_info.scale ||
                        source_info.zero_point != output_info.zero_point) {
                        throw std::runtime_error("microWakeWord concat changes quantization metadata");
                    }
                }
                values[op_outputs[0]] = value;
            } else if (type == "strided_slice") {
                auto value = require_value(op_inputs[0]);
                const auto begin = ints(op, "begin");
                const auto end = ints(op, "end");
                for (size_t axis = 0; axis < begin.size(); ++axis) {
                    if (begin[axis] != 0 || end[axis] != infos[op_inputs[0]].tflite_shape[axis]) {
                        value = modules::SliceModule({native_axis(static_cast<int>(axis), begin.size()),
                                                      begin[axis], end[axis] - begin[axis]}).build(ctx, value);
                    }
                }
                passthrough(op_inputs[0], op_outputs[0], value);
            } else if (type == "split_v") {
                const auto source_value = require_value(op_inputs[0]);
                const auto sizes = ints(op, "sizes");
                const int axis = native_axis(json::require_i32(op, "axis"), infos[op_inputs[0]].tflite_shape.size());
                int64_t offset = 0;
                if (sizes.size() != op_outputs.size()) throw std::runtime_error("microWakeWord split size mismatch");
                for (size_t i = 0; i < sizes.size(); ++i) {
                    passthrough(op_inputs[0], op_outputs[i],
                                modules::SliceModule({axis, offset, sizes[i]}).build(ctx, source_value));
                    offset += sizes[i];
                }
            } else if (type == "conv_2d" || type == "depthwise_conv_2d") {
                const auto input_value = require_value(op_inputs[0]);
                const auto weight_shape = source.require_metadata(json::require_string(op, "weight")).shape;
                const auto stride = ints(op, "stride");
                const auto dilation = ints(op, "dilation");
                const int padding_mode = json::require_i32(op, "padding");
                if ((padding_mode != 0 && padding_mode != 1) || stride.size() != 2 || dilation.size() != 2) {
                    throw std::runtime_error("microWakeWord convolution has invalid padding or stride metadata");
                }
                const auto output_shape = native_shape(infos[op_outputs[0]].tflite_shape);
                const auto same_padding = [&](int axis, int64_t kernel, int stride_value, int dilation_value) {
                    if (padding_mode == 1) return 0;
                    const int64_t total = std::max<int64_t>(
                        0, (output_shape.dims[axis] - 1) * stride_value +
                               dilation_value * (kernel - 1) + 1 - input_value.shape.dims[axis]);
                    if (total % 2 != 0) {
                        throw std::runtime_error("microWakeWord asymmetric SAME convolution padding is unsupported");
                    }
                    return static_cast<int>(total / 2);
                };
                const int pad_height = same_padding(2, weight_shape[2], stride[0], dilation[0]);
                const int pad_width = same_padding(3, weight_shape[3], stride[1], dilation[1]);
                modules::Conv2dWeights weights;
                weights.weight = store.load_f32_tensor(source, json::require_string(op, "weight"), weight_shape);
                weights.bias = store.load_f32_tensor(source, json::require_string(op, "bias"), {weight_shape[0]});
                core::TensorValue value;
                if (type == "conv_2d") {
                    value = modules::Conv2dModule({weight_shape[1], weight_shape[0], weight_shape[2], weight_shape[3],
                                                   stride[0], stride[1], pad_height, pad_width,
                                                   dilation[0], dilation[1], true})
                                .build(ctx, input_value, weights);
                } else {
                    value = modules::DepthwiseConv2dModule({weight_shape[0], weight_shape[2], weight_shape[3],
                                                            stride[0], stride[1], pad_height, pad_width,
                                                            dilation[0], dilation[1], true})
                                .build(ctx, input_value, weights);
                }
                value = apply_activation(ctx, value, json::require_string(op, "activation"));
                finish(op_outputs[0], value);
            } else if (type == "fully_connected") {
                const auto weight_shape = source.require_metadata(json::require_string(op, "weight")).shape;
                modules::LinearWeights weights;
                weights.weight = store.load_f32_tensor(source, json::require_string(op, "weight"), weight_shape);
                weights.bias = store.load_f32_tensor(source, json::require_string(op, "bias"), {weight_shape[0]});
                auto value = modules::LinearModule({weight_shape[1], weight_shape[0], true, GGML_PREC_F32})
                                 .build(ctx, require_value(op_inputs[0]), weights);
                value = apply_activation(ctx, value, json::require_string(op, "activation"));
                finish(op_outputs[0], value);
            } else if (type == "logistic") {
                finish(op_outputs[0], modules::SigmoidModule{}.build(ctx, require_value(op_inputs[0])));
            } else if (type == "quantize") {
                finish(op_outputs[0], require_value(op_inputs[0]));
            } else {
                throw std::runtime_error("unsupported microWakeWord converted operator: " + type);
            }
        }

        output = require_value(output_index);
        // The reference microWakeWord runtime maps the uint8 sigmoid result by
        // q / 255, while the TFLite tensor itself uses scale 1 / 256.
        if (infos[output_index].type == "u8" && infos[output_index].scale.has_value()) {
            output = core::wrap_tensor(ggml_scale(ctx.ggml, output.tensor, 256.0F / 255.0F), output.shape, GGML_TYPE_F32);
        }
        ggml_set_output(output.tensor);
        for (const auto & state : states) {
            auto value = require_value(state.assign);
            state_outputs.push_back(value);
            ggml_set_output(value.tensor);
        }
        store.upload();
        graph = ggml_new_graph_custom(context.get(), 2048, false);
        ggml_build_forward_expand(graph, output.tensor);
        for (const auto & value : state_outputs) ggml_build_forward_expand(graph, value.tensor);
        core::validate_backend_graph_supported(execution.backend(), graph, "micro_wake_word");
        allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
            throw std::runtime_error("microWakeWord graph allocation failed");
        }
        core::prepare_host_graph_plan(execution, graph, plan);
    }

    ~Impl() { core::release_backend_graph_resources(execution.backend(), graph, true); }

    float run(const std::vector<float> & features) {
        if (features.size() != static_cast<size_t>(model.input_frames * model.feature_dim)) {
            throw std::runtime_error("microWakeWord feature input has the wrong size");
        }
        std::vector<float> channel_major(features.size());
        for (int frame = 0; frame < model.input_frames; ++frame) {
            for (int channel = 0; channel < model.feature_dim; ++channel) {
                const float source = features[static_cast<size_t>(frame * model.feature_dim + channel)];
                // Match the reference wrapper's NumPy astype(int8): scale,
                // add the integer zero point, then truncate toward zero.
                const int quantized = static_cast<int>(source / input_scale + static_cast<float>(input_zero_point));
                channel_major[static_cast<size_t>(channel * model.input_frames + frame)] =
                    (static_cast<float>(quantized) - static_cast<float>(input_zero_point)) * input_scale;
            }
        }
        core::write_tensor_f32(input, channel_major);
        for (size_t i = 0; i < state_inputs.size(); ++i) core::write_tensor_f32(state_inputs[i], state_values[i]);
        const auto started = std::chrono::steady_clock::now();
        if (core::compute_graph(execution, graph, plan, "micro_wake_word") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("microWakeWord graph compute failed");
        }
        debug::timing_log_scalar("micro_wake_word.network_ms", debug::elapsed_ms(started));
        for (size_t i = 0; i < state_outputs.size(); ++i) {
            core::read_tensor_f32_into(state_outputs[i].tensor, state_values[i]);
        }
        const auto result = core::read_tensor_f32(output.tensor);
        if (result.size() != 1 || !std::isfinite(result[0])) {
            throw std::runtime_error("microWakeWord produced an invalid probability");
        }
        debug::trace_log_scalar("micro_wake_word.probability", result[0]);
        return result[0];
    }

    void clear() {
        for (auto & state : state_values) std::fill(state.begin(), state.end(), 0.0F);
    }
};

Runtime::Runtime(std::shared_ptr<const assets::TensorSource> source,
                 const io::json::Value & config,
                 core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(*source, config, execution)) {}
Runtime::~Runtime() = default;
const ModelConfig & Runtime::config() const noexcept { return impl_->model; }
float Runtime::infer(const std::vector<float> & features) { return impl_->run(features); }
void Runtime::reset() { impl_->clear(); }

}  // namespace engine::models::micro_wake_word
