#include "engine/community_models/kroko_asr/zipformer.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/zipformer_modules.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace engine::models::kroko_asr {
namespace {

constexpr int64_t kInputDim = 192;
constexpr int64_t kPosDim = 48;
constexpr int64_t kPosHeadDim = 4;
constexpr size_t kWeightContextBytes = 2ull * 1024ull * 1024ull;
constexpr size_t kGraphContextBytes = 32ull * 1024ull * 1024ull;
constexpr size_t kGraphNodes = 16384;

struct Param {
    std::vector<int64_t> shape;
    std::vector<float> host;
    core::TensorValue tensor;
};

struct GraphConstant {
    ggml_tensor * tensor = nullptr;
    std::vector<float> values;
};

struct StateTensor {
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    core::TensorValue value;
    std::vector<float> host;
};

struct LayerState {
    StateTensor key;
    StateTensor nonlin;
    StateTensor val1;
    StateTensor val2;
    StateTensor conv1;
    StateTensor conv2;
};

struct LayerStateValue {
    core::TensorValue key;
    core::TensorValue nonlin;
    core::TensorValue val1;
    core::TensorValue val2;
    core::TensorValue conv1;
    core::TensorValue conv2;
};

struct ValueResult {
    core::TensorValue output;
    core::TensorValue state;
};

struct LayerResult {
    core::TensorValue output;
    LayerStateValue state;
};

struct StackMask {
    ggml_tensor * input = nullptr;
    core::TensorValue value;
    int64_t frames = 0;
    int64_t left_context = 0;
    int64_t downsampling = 1;
    std::vector<float> host;
};

const Param & param(
    const std::unordered_map<std::string, Param> & params,
    const std::string & name) {
    const auto it = params.find(name);
    if (it == params.end()) {
        throw std::runtime_error(
            "missing Kroko Zipformer tensor: " + name);
    }
    return it->second;
}

core::TensorValue contiguous(
    core::ModuleBuildContext & context,
    const core::TensorValue & value) {
    return core::ensure_backend_addressable_layout(context, value);
}

core::TensorValue scale(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    float value) {
    // Normalize singleton strides too: ggml_scale requires padded-1D layout.
    const auto source = core::reshape_tensor(context, contiguous(context, input), input.shape);
    return core::wrap_tensor(
        ggml_scale(context.ggml, source.tensor, value),
        input.shape,
        GGML_TYPE_F32);
}

core::TensorValue linear(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix) {
    const std::string weight_name = prefix + ".weight";
    const std::string bias_name = prefix + ".bias";
    const auto & weight = param(params, weight_name);
    const auto & bias = param(params, bias_name);
    if (weight.shape.size() != 2 ||
        weight.shape[1] != input.shape.last_dim()) {
        throw std::runtime_error(
            "Kroko Zipformer linear shape mismatch: " + prefix);
    }
    return modules::LinearModule(
        {weight.shape[1], weight.shape[0], true})
        .build(
            context,
            input,
            {weight.tensor, bias.tensor});
}

modules::ZipformerLinearWeights zipformer_linear_weights(
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix) {
    return {
        param(params, prefix + ".weight").tensor,
        param(params, prefix + ".bias").tensor,
    };
}

modules::DepthwiseConv1dWeights zipformer_depthwise_conv_weights(
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix) {
    return {
        param(params, prefix + ".weight").tensor,
        param(params, prefix + ".bias").tensor,
    };
}

modules::ZipformerRelativeAttentionWeights relative_attention_weights(
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix) {
    return {
        zipformer_linear_weights(
            params,
            prefix + ".self_attn_weights.in_proj"),
        param(params, prefix + ".self_attn_weights.linear_pos.weight").host,
    };
}

modules::ZipformerNonlinearAttentionWeights nonlinear_attention_weights(
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix) {
    const std::string base = prefix + ".nonlin_attention";
    return {
        zipformer_linear_weights(params, base + ".in_proj"),
        zipformer_linear_weights(params, base + ".out_proj"),
    };
}

modules::ZipformerConvolutionWeights convolution_weights(
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix,
    const std::string & module) {
    const std::string base = prefix + "." + module;
    return {
        zipformer_linear_weights(params, base + ".in_proj"),
        zipformer_depthwise_conv_weights(
            params,
            base + ".depthwise_conv.causal_conv"),
        zipformer_depthwise_conv_weights(
            params,
            base + ".depthwise_conv.chunkwise_conv"),
        param(params, base + ".depthwise_conv.chunk_scale_left").host,
        param(params, base + ".depthwise_conv.chunk_scale_right").host,
        zipformer_linear_weights(params, base + ".out_proj"),
    };
}

core::TensorValue graph_constant(
    core::ModuleBuildContext & context,
    std::vector<GraphConstant> & constants,
    const core::TensorShape & shape,
    std::vector<float> values) {
    auto result =
        core::make_tensor(context, GGML_TYPE_F32, shape);
    ggml_set_input(result.tensor);
    constants.push_back(
        GraphConstant{result.tensor, std::move(values)});
    return result;
}

size_t element_count(const core::TensorShape & shape) {
    size_t result = 1;
    for (size_t axis = 0; axis < shape.rank; ++axis) {
        result *= static_cast<size_t>(shape.dims[axis]);
    }
    return result;
}

StateTensor state_tensor(
    core::ModuleBuildContext & context,
    const core::TensorShape & shape) {
    StateTensor result;
    result.value = core::make_tensor(
        context, GGML_TYPE_F32, shape);
    result.input = result.value.tensor;
    ggml_set_input(result.input);
    result.host.assign(element_count(shape), 0.0F);
    return result;
}

ValueResult self_attention(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    const core::TensorValue & attention,
    const core::TensorValue & cached_value,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix,
    const std::string & module,
    int64_t heads,
    int64_t value_dim) {
    const std::string base = prefix + "." + module;
    auto values = linear(
        context, input, params, base + ".in_proj");
    const int64_t seq = values.shape.dims[0];
    core::TensorValue new_state;
    if (cached_value.valid()) {
        const int64_t left_context = cached_value.shape.dims[0];
        values = modules::ConcatModule({0}).build(context, cached_value, values);
        new_state = modules::SliceModule({0, seq, left_context}).build(context, values);
    }
    values = core::reshape_tensor(context, contiguous(context, values),
        core::TensorShape::from_dims({values.shape.dims[0], heads, value_dim}));
    values = modules::TransposeModule({{1, 0, 2, 3}, 3}).build(context, values);
    auto joined = modules::MatMulModule().build(context, attention, values);
    joined = modules::TransposeModule({{1, 0, 2, 3}, 3}).build(context, joined);
    joined = core::reshape_tensor(context, contiguous(context, joined),
        core::TensorShape::from_dims({seq, 1, heads * value_dim}));
    return {
        linear(context, joined, params, base + ".out_proj"),
        new_state};
}

core::TensorValue feed_forward(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix,
    const std::string & module) {
    auto result = linear(
        context,
        input,
        params,
        prefix + "." + module + ".in_proj");
    result = modules::SwooshLModule().build(context, result);
    return linear(
        context,
        result,
        params,
        prefix + "." + module + ".out_proj");
}

modules::ZipformerStatefulOutputs offline_convolution(
    core::ModuleBuildContext & context, const core::TensorValue & input,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix, int64_t kernel) {
    const int64_t channels = input.shape.last_dim();
    auto projected = linear(context, input, params, prefix + ".in_proj");
    auto value = modules::SliceModule({2, 0, channels}).build(context, projected);
    auto gate = modules::SliceModule({2, channels, channels}).build(context, projected);
    value = modules::MulModule(false, context.backend_type == core::BackendType::Cpu).build(context, value, modules::SigmoidModule().build(context, gate));
    const auto weights = zipformer_depthwise_conv_weights(params, prefix + ".depthwise_conv");
    if (context.backend_type == core::BackendType::Cpu) {
        value = core::reshape_tensor(context, contiguous(context, value),
            core::TensorShape::from_dims({1, 1, input.shape.dims[0], channels}));
        value = modules::Pad2dModule({0, 0, kernel / 2, kernel / 2}).build(context, value);
        value = core::reshape_tensor(context, value,
            core::TensorShape::from_dims({1, input.shape.dims[0] + kernel - 1, channels}));
        value = modules::DepthwiseConv1dModule({channels, kernel, 1, 0, 1, false})
            .build_time_major(context, value, {weights.weight, std::nullopt});
        value = modules::AddModule(false, true).build(context, value,
            core::reshape_tensor(context, *weights.bias, core::TensorShape::from_dims({1, 1, channels})));
        value = core::reshape_tensor(context, value, input.shape);
    } else {
        value = modules::TransposeModule({{1, 2, 0, 3}, 3}).build(context, value);
        value = modules::DepthwiseConv1dModule({channels, kernel, 1, static_cast<int>(kernel / 2), 1, true})
            .build(context, value, weights);
        value = modules::TransposeModule({{2, 0, 1, 3}, 3}).build(context, value);
    }
    value = core::reshape_tensor(context, contiguous(context, value), input.shape);
    value = modules::SwooshRModule().build(context, value);
    return {linear(context, value, params, prefix + ".out_proj"), {}};
}

LayerResult zipformer_layer(
    core::ModuleBuildContext & context,
    core::ModuleBuildContext & constant_context,
    std::vector<GraphConstant> & constants,
    const core::TensorValue & input,
    const LayerStateValue & state,
    const core::TensorValue & padding_mask,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix,
    int64_t heads,
    int64_t query_dim,
    int64_t value_dim,
    int64_t kernel) {
    const auto original = input;
    modules::ZipformerConstantFactory constant_factory =
        [&](const core::TensorShape & shape, std::vector<float> values) {
            return graph_constant(
                constant_context,
                constants,
                shape,
                std::move(values));
        };
    const auto attention =
        modules::ZipformerRelativeAttentionModule({
            heads,
            query_dim,
            kPosDim,
            kPosHeadDim,
            modules::ZipformerAttentionImpl::Batched,
        }).build(
        context,
        constant_factory,
        input,
        state.key,
        padding_mask,
        relative_attention_weights(params, prefix));
    auto output = modules::AddModule().build(
        context,
        input,
        feed_forward(
            context, input, params, prefix, "feed_forward1"));
    const auto nonlinear =
        modules::ZipformerNonlinearAttentionModule().build(
        context,
        output,
        attention.weights.front(),
        state.nonlin,
        nonlinear_attention_weights(params, prefix));
    output = modules::AddModule().build(
        context, output, nonlinear.output);
    const auto attention1 = self_attention(
        context,
        output,
        attention.batched_weights,
        state.val1,
        params,
        prefix,
        "self_attn1",
        heads,
        value_dim);
    output = modules::AddModule().build(
        context, output, attention1.output);
    const auto convolution1 = !state.key.valid()
        ? offline_convolution(context, output, params, prefix + ".conv_module1", kernel)
        : modules::ZipformerConvolutionModule({kernel}).build(
        context,
        constant_factory,
        output,
        state.conv1,
        convolution_weights(params, prefix, "conv_module1"));
    output = modules::AddModule().build(
        context, output, convolution1.output);
    output = modules::AddModule().build(
        context,
        output,
        feed_forward(
            context, output, params, prefix, "feed_forward2"));
    output = modules::ScaledBypassModule().build(
        context,
        original,
        output,
        param(params, prefix + ".bypass_mid.bypass_scale").tensor);
    const auto attention2 = self_attention(
        context,
        output,
        attention.batched_weights,
        state.val2,
        params,
        prefix,
        "self_attn2",
        heads,
        value_dim);
    output = modules::AddModule().build(
        context, output, attention2.output);
    const auto convolution2 = !state.key.valid()
        ? offline_convolution(context, output, params, prefix + ".conv_module2", kernel)
        : modules::ZipformerConvolutionModule({kernel}).build(
        context,
        constant_factory,
        output,
        state.conv2,
        convolution_weights(params, prefix, "conv_module2"));
    output = modules::AddModule().build(
        context, output, convolution2.output);
    output = modules::AddModule().build(
        context,
        output,
        feed_forward(
            context, output, params, prefix, "feed_forward3"));
    const std::string norm_bias_name = prefix + ".norm.bias";
    const std::string norm_log_scale_name = prefix + ".norm.log_scale";
    const auto & norm_bias = param(params, norm_bias_name);
    const auto & norm_log_scale = param(params, norm_log_scale_name);
    if (norm_log_scale.host.size() != 1) {
        throw std::runtime_error(
            "Kroko BiasNorm scale is unavailable: " + prefix + ".norm");
    }
    output = modules::BiasNormModule({output.shape.last_dim()}).build(
        context,
        output,
        {norm_bias.tensor, norm_log_scale.host.front()});
    return {
        modules::ScaledBypassModule().build(
            context,
            original,
            output,
            param(params, prefix + ".bypass.bypass_scale").tensor),
        {
            attention.key_state,
            nonlinear.state,
            attention1.state,
            attention2.state,
            convolution1.state,
            convolution2.state,
        }};
}

core::TensorValue convert_channels(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    int64_t channels) {
    const int64_t current = input.shape.last_dim();
    if (current == channels) {
        return input;
    }
    if (current > channels) {
        return modules::SliceModule(
            {2, 0, channels})
            .build(context, input);
    }
    std::array<int32_t, 4> left = {0, 0, 0, 0};
    std::array<int32_t, 4> right = {0, 0, 0, 0};
    right[core::logical_axis_to_ggml_axis(3, 2)] =
        static_cast<int32_t>(channels - current);
    const auto source = contiguous(context, input);
    auto shape = input.shape;
    shape.dims[2] = channels;
    return core::wrap_tensor(
        ggml_pad_ext(
            context.ggml,
            source.tensor,
            left[0],
            right[0],
            left[1],
            right[1],
            left[2],
            right[2],
            left[3],
            right[3]),
        shape,
        GGML_TYPE_F32);
}

core::TensorValue downsample(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    const Param & weights) {
    if (weights.host.empty()) {
        throw std::runtime_error(
            "Kroko downsample weights are unavailable");
    }
    const int64_t factor =
        static_cast<int64_t>(weights.host.size());
    const int64_t output_frames =
        (input.shape.dims[0] + factor - 1) / factor;
    const int64_t channels = input.shape.last_dim();
    const int64_t padding = output_frames * factor - input.shape.dims[0];
    auto padded = input;
    if (padding != 0) {
        auto last = modules::SliceModule({0, input.shape.dims[0] - 1, 1})
                        .build(context, input);
        last = modules::RepeatModule({core::TensorShape::from_dims({padding, 1, channels})})
                   .build(context, last);
        padded = modules::ConcatModule({0}).build(context, input, last);
    }
    auto grouped = core::reshape_tensor(
        context, contiguous(context, padded),
        core::TensorShape::from_dims({output_frames, factor, channels}));
    core::TensorValue output;
    // Preserve the original ordered weighted sum, but process all frames together.
    for (int64_t index = 0; index < factor; ++index) {
        auto phase = modules::SliceModule({1, index, 1}).build(context, grouped);
        phase = scale(context, phase, weights.host[static_cast<size_t>(index)]);
        output = output.valid() ? modules::AddModule().build(context, output, phase) : phase;
    }
    return output;
}

core::TensorValue upsample(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    int64_t factor,
    int64_t target_frames) {
    auto repeated = core::reshape_tensor(
        context, contiguous(context, input),
        core::TensorShape::from_dims({input.shape.dims[0], 1, 1, input.shape.last_dim()}));
    repeated = modules::RepeatModule({core::TensorShape::from_dims(
        {input.shape.dims[0], factor, 1, input.shape.last_dim()})}).build(context, repeated);
    repeated = core::reshape_tensor(
        context, repeated,
        core::TensorShape::from_dims({input.shape.dims[0] * factor, 1, input.shape.last_dim()}));
    return modules::SliceModule({0, 0, target_frames}).build(context, repeated);
}

class ZipformerGraph {
public:
    ZipformerGraph(
        core::ExecutionContext & execution,
        const KrokoASRConfig & config,
        const std::unordered_map<std::string, Param> & params,
        int64_t input_frames = 0)
        : backend_(execution.backend()),
          input_frames_(config.streaming ? config.chunk_shift / 2 : input_frames),
          prepared_(execution, {"kroko_asr.zipformer", kGraphContextBytes, kGraphNodes, 1024 * 1024}) {
        if (input_frames_ <= 0) {
            throw std::runtime_error(
                "Kroko Zipformer input frame count is invalid");
        }
        auto & build = prepared_.build_context();
        auto input = core::make_tensor(
            build,
            GGML_TYPE_F32,
            core::TensorShape::from_dims(
                {input_frames_, 1, kInputDim}));
        input_ = input.tensor;
        ggml_set_input(input_);
        auto current = input;
        std::vector<core::TensorValue> stack_outputs;
        int64_t layer_offset = 0;
        for (size_t stack = 0;
             stack < config.encoder_dims.size();
             ++stack) {
            const int64_t channels = config.encoder_dims[stack];
            const int64_t factor =
                config.downsampling_factors[stack];
            current = convert_channels(build, current, channels);
            const auto stack_input = current;
            if (factor > 1) {
                current = downsample(
                    build,
                    current,
                    param(
                        params,
                        "encoder.encoder.encoders." +
                            std::to_string(stack) +
                            ".downsample.weights"));
            }
            const int64_t left_context =
                config.left_context_len[stack];
            core::TensorValue padding_mask;
            if (config.streaming) {
                StackMask stack_mask;
                stack_mask.value = core::make_tensor(
                    build,
                    GGML_TYPE_F32,
                    core::TensorShape::from_dims(
                        {1,
                         current.shape.dims[0],
                         left_context + current.shape.dims[0]}));
                stack_mask.input = stack_mask.value.tensor;
                ggml_set_input(stack_mask.input);
                stack_mask.frames = current.shape.dims[0];
                stack_mask.left_context = left_context;
                stack_mask.downsampling = factor;
                stack_mask.host.assign(
                    element_count(stack_mask.value.shape), 0.0F);
                padding_mask = stack_mask.value;
                stack_masks_.push_back(std::move(stack_mask));
            }
            for (int64_t layer = 0;
                 layer < config.num_encoder_layers[stack];
                 ++layer) {
                const std::string prefix =
                    "encoder.encoder.encoders." +
                    std::to_string(stack) +
                    (stack == 0 ? ".layers." : ".encoder.layers.") +
                    std::to_string(layer);
                LayerState layer_state;
                if (config.streaming) {
                    layer_state.key = state_tensor(
                        build,
                        core::TensorShape::from_dims(
                            {left_context,
                             1,
                             config.num_heads[stack] *
                                 config.query_head_dims[stack]}));
                    layer_state.nonlin = state_tensor(
                        build,
                        core::TensorShape::from_dims(
                            {left_context, 1, 3 * channels / 4}));
                    layer_state.val1 = state_tensor(
                        build,
                        core::TensorShape::from_dims(
                            {left_context,
                             1,
                             config.num_heads[stack] *
                                 config.value_head_dims[stack]}));
                    layer_state.val2 = state_tensor(
                        build,
                        core::TensorShape::from_dims(
                            {left_context,
                             1,
                             config.num_heads[stack] *
                                 config.value_head_dims[stack]}));
                    layer_state.conv1 = state_tensor(
                        build,
                        core::TensorShape::from_dims(
                            {1,
                             channels,
                             config.cnn_module_kernels[stack] / 2}));
                    layer_state.conv2 = state_tensor(
                        build,
                        core::TensorShape::from_dims(
                            {1,
                             channels,
                             config.cnn_module_kernels[stack] / 2}));
                }
                const auto layer_result = zipformer_layer(
                    build,
                    prepared_.persistent_build_context(),
                    constants_,
                    current,
                    {
                        layer_state.key.value,
                        layer_state.nonlin.value,
                        layer_state.val1.value,
                        layer_state.val2.value,
                        layer_state.conv1.value,
                        layer_state.conv2.value,
                    },
                    padding_mask,
                    params,
                    prefix,
                    config.num_heads[stack],
                    config.query_head_dims[stack],
                    config.value_head_dims[stack],
                    config.cnn_module_kernels[stack]);
                current = layer_result.output;
                if (config.streaming) {
                    layer_state.key.output =
                        layer_result.state.key.tensor;
                    layer_state.nonlin.output =
                        layer_result.state.nonlin.tensor;
                    layer_state.val1.output =
                        layer_result.state.val1.tensor;
                    layer_state.val2.output =
                        layer_result.state.val2.tensor;
                    layer_state.conv1.output =
                        layer_result.state.conv1.tensor;
                    layer_state.conv2.output =
                        layer_result.state.conv2.tensor;
                    layer_states_.push_back(std::move(layer_state));
                }
                ++layer_offset;
            }
            if (factor > 1) {
                current = upsample(
                    build,
                    current,
                    factor,
                    stack_input.shape.dims[0]);
                current = modules::ScaledBypassModule().build(
                    build,
                    stack_input,
                    current,
                    param(
                        params,
                        "encoder.encoder.encoders." +
                            std::to_string(stack) +
                            ".out_combiner.bypass_scale").tensor);
            }
            stack_outputs.push_back(current);
        }
        if (layer_offset != 19) {
            throw std::runtime_error(
                "Kroko Zipformer layer count is not 19");
        }

        std::vector<core::TensorValue> full_pieces;
        full_pieces.push_back(stack_outputs.back());
        int64_t current_dim =
            config.encoder_dims.back();
        for (int64_t stack =
                 static_cast<int64_t>(stack_outputs.size()) - 2;
             stack >= 0;
             --stack) {
            const int64_t stack_dim =
                config.encoder_dims[static_cast<size_t>(stack)];
            if (stack_dim > current_dim) {
                full_pieces.push_back(
                    modules::SliceModule(
                        {2, current_dim, stack_dim - current_dim})
                        .build(
                            build,
                            stack_outputs[
                                static_cast<size_t>(stack)]));
                current_dim = stack_dim;
            }
        }
        current = full_pieces.front();
        for (size_t index = 1;
             index < full_pieces.size();
             ++index) {
            current = modules::ConcatModule({2}).build(
                build, current, full_pieces[index]);
        }
        current = downsample(
            build,
            current,
            param(
                params,
                "encoder.encoder.downsample_output.weights"));
        current = linear(
            build,
            current,
            params,
            "encoder.encoder_proj");
        output_frames_ = current.shape.dims[0];
        output_dim_ = current.shape.dims[2];
        output_ = current.tensor;
        ggml_set_output(output_);
        for (auto & state : layer_states_) {
            ggml_set_output(state.key.output);
            ggml_set_output(state.nonlin.output);
            ggml_set_output(state.val1.output);
            ggml_set_output(state.val2.output);
            ggml_set_output(state.conv1.output);
            ggml_set_output(state.conv2.output);
        }
        auto * graph = prepared_.graph();
        ggml_build_forward_expand(graph, output_);
        for (const auto & state : layer_states_) {
            ggml_build_forward_expand(graph, state.key.output);
            ggml_build_forward_expand(
                graph, state.nonlin.output);
            ggml_build_forward_expand(graph, state.val1.output);
            ggml_build_forward_expand(graph, state.val2.output);
            ggml_build_forward_expand(graph, state.conv1.output);
            ggml_build_forward_expand(graph, state.conv2.output);
        }
        prepared_.allocate_persistent();
        prepared_.allocate();
        for (const auto & constant : constants_) {
            ggml_backend_tensor_set(
                constant.tensor,
                constant.values.data(),
                0,
                constant.values.size() * sizeof(float));
        }
    }

    KrokoEncoderChunk run(
        const std::vector<float> & input) {
        if (input.size() !=
            static_cast<size_t>(input_frames_ * kInputDim)) {
            throw std::runtime_error(
                "Kroko Zipformer input shape does not match the package chunk size");
        }
        ggml_backend_tensor_set_async(
            backend_,
            input_,
            input.data(),
            0,
            input.size() * sizeof(float));
        const auto set_state = [&](StateTensor & state) {
            ggml_backend_tensor_set_async(
                backend_,
                state.input,
                state.host.data(),
                0,
                state.host.size() * sizeof(float));
        };
        for (auto & state : layer_states_) {
            set_state(state.key);
            set_state(state.nonlin);
            set_state(state.val1);
            set_state(state.val2);
            set_state(state.conv1);
            set_state(state.conv2);
        }
        for (auto & mask : stack_masks_) {
            const int64_t valid_left = std::min(
                mask.left_context,
                processed_frames_ / mask.downsampling);
            std::fill(
                mask.host.begin(), mask.host.end(), 0.0F);
            for (int64_t target = 0;
                 target < mask.frames;
                 ++target) {
                for (int64_t source = 0;
                     source < mask.left_context - valid_left;
                     ++source) {
                    mask.host[static_cast<size_t>(
                        target * mask.value.shape.last_dim() + source)] = -1000.0F;
                }
            }
            ggml_backend_tensor_set_async(
                backend_,
                mask.input,
                mask.host.data(),
                0,
                mask.host.size() * sizeof(float));
        }
        prepared_.compute();
        KrokoEncoderChunk result;
        result.frames = output_frames_;
        result.channels = output_dim_;
        result.values.resize(
            static_cast<size_t>(output_frames_ * output_dim_));
        ggml_backend_tensor_get_async(
            backend_,
            output_,
            result.values.data(),
            0,
            result.values.size() * sizeof(float));
        const auto get_state = [&](StateTensor & state) {
            ggml_backend_tensor_get_async(
                backend_,
                state.output,
                state.host.data(),
                0,
                state.host.size() * sizeof(float));
        };
        for (auto & state : layer_states_) {
            get_state(state.key);
            get_state(state.nonlin);
            get_state(state.val1);
            get_state(state.val2);
            get_state(state.conv1);
            get_state(state.conv2);
        }
        ggml_backend_synchronize(backend_);
        processed_frames_ += input_frames_;
        return result;
    }

    void reset() {
        processed_frames_ = 0;
        const auto clear = [](StateTensor & value) {
            std::fill(
                value.host.begin(),
                value.host.end(),
                0.0F);
        };
        for (auto & state : layer_states_) {
            clear(state.key);
            clear(state.nonlin);
            clear(state.val1);
            clear(state.val2);
            clear(state.conv1);
            clear(state.conv2);
        }
    }

private:
    ggml_backend_t backend_ = nullptr;
    std::vector<GraphConstant> constants_;
    std::vector<LayerState> layer_states_;
    std::vector<StackMask> stack_masks_;
    int64_t processed_frames_ = 0;
    int64_t input_frames_ = 0;
    int64_t output_frames_ = 0;
    int64_t output_dim_ = 0;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    core::PreparedGraph prepared_;
};

bool needs_host_values(std::string_view name) {
    const auto ends_with = [&](std::string_view suffix) {
        return name.size() >= suffix.size() &&
               name.substr(name.size() - suffix.size()) == suffix;
    };
    return ends_with(".linear_pos.weight") ||
           ends_with(".norm.log_scale") ||
           ends_with(".bypass_scale") ||
           name.find(".downsample.weights") !=
               std::string_view::npos ||
           name.find(".downsample_output.weights") !=
               std::string_view::npos ||
           ends_with(".chunk_scale_left") ||
           ends_with(".chunk_scale_right");
}

bool is_zipformer_tensor(std::string_view name) {
    const auto starts_with = [&](std::string_view prefix) {
        return name.size() >= prefix.size() &&
               name.substr(0, prefix.size()) == prefix;
    };
    return starts_with("encoder.encoder.") ||
           starts_with("encoder.encoder_proj.");
}

}  // namespace

struct KrokoZipformerRuntime::Impl {
    std::shared_ptr<const KrokoASRAssets> assets;
    std::shared_ptr<core::BackendWeightStore> store;
    std::unordered_map<std::string, Param> params;
    std::unique_ptr<ZipformerGraph> graph;
    core::ExecutionContext * execution = nullptr;
    int64_t input_frames = 0;
};

KrokoZipformerRuntime::KrokoZipformerRuntime(
    std::shared_ptr<const KrokoASRAssets> assets,
    core::ExecutionContext & execution_context)
    : impl_(std::make_unique<Impl>()) {
    if (assets == nullptr || assets->weights == nullptr) {
        throw std::runtime_error(
            "Kroko Zipformer requires tensor assets");
    }
    impl_->assets = std::move(assets);
    impl_->execution = &execution_context;

    impl_->store = std::make_shared<core::BackendWeightStore>(
        execution_context.backend(),
        execution_context.backend_type(),
        "Kroko Zipformer weights",
        kWeightContextBytes);
    const auto & source = *impl_->assets->weights;
    for (const auto & metadata : source.tensors()) {
        if (!is_zipformer_tensor(metadata.name)) {
            continue;
        }
        Param value;
        value.shape = metadata.shape.empty()
            ? std::vector<int64_t>{1}
            : metadata.shape;
        const bool keep_host_type =
            metadata.shape.size() == 2 &&
            !needs_host_values(metadata.name);
        value.tensor = keep_host_type
            ? impl_->store->load_tensor(
                  source,
                  metadata.name,
                  assets::TensorStorageType::Native,
                  value.shape)
            : impl_->store->load_f32_tensor(
                  source, metadata.name, value.shape);
        if (needs_host_values(metadata.name)) {
            value.host =
                source.require_f32(metadata.name, value.shape);
        }
        impl_->params.emplace(
            metadata.name, std::move(value));
    }
    impl_->store->upload();
    if (impl_->assets->config.streaming) impl_->graph = std::make_unique<ZipformerGraph>(
        execution_context,
        impl_->assets->config,
        impl_->params);
}

KrokoZipformerRuntime::~KrokoZipformerRuntime() = default;

KrokoEncoderChunk
KrokoZipformerRuntime::encode_chunk(
    const std::vector<float> & subsampled_features) const {
    if (!impl_->assets->config.streaming) {
        const auto frames = static_cast<int64_t>(subsampled_features.size() / kInputDim);
        if (!impl_->graph || impl_->input_frames != frames) {
            impl_->graph.reset();
            impl_->graph = std::make_unique<ZipformerGraph>(*impl_->execution, impl_->assets->config, impl_->params, frames);
            impl_->input_frames = frames;
        }
    }
    const auto start = std::chrono::steady_clock::now();
    auto result =
        impl_->graph->run(subsampled_features);
    engine::debug::timing_log_scalar(
        "kroko_asr.zipformer_ms",
        engine::debug::elapsed_ms(start));
    return result;
}

void KrokoZipformerRuntime::reset() const {
    if (impl_->graph) impl_->graph->reset();
}

}  // namespace engine::models::kroko_asr
