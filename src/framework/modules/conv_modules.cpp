#include "engine/framework/modules/conv_modules.h"
#include "tensor_layout_utils.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/modules/structural_modules.h"

#include <stdexcept>
#include <string>

namespace engine::modules {

namespace {

const core::ModulePortSpec kConvInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"weight", core::PortKind::Parameter, false},
    {"bias", core::PortKind::Parameter, true},
};

const core::ModulePortSpec kSingleOutput[] = {
    {"output", core::PortKind::Activation, false},
};

const core::ModuleSchema kConv1dSchema = {
    "Conv1d",
    "nn.conv",
    kConvInputs,
    3,
    kSingleOutput,
    1,
    "Applies a 1D convolution to channel-first inputs [batch, channels, frames].",
};

const core::ModuleSchema kConv2dSchema = {
    "Conv2d",
    "nn.conv",
    kConvInputs,
    3,
    kSingleOutput,
    1,
    "Applies a 2D convolution to channel-first inputs [batch, channels, height, width].",
};

const core::ModuleSchema kDepthwiseConv2dSchema = {
    "DepthwiseConv2d",
    "nn.conv",
    kConvInputs,
    3,
    kSingleOutput,
    1,
    "Applies a depthwise 2D convolution to channel-first inputs [batch, channels, height, width].",
};

const core::ModuleSchema kConvTranspose1dSchema = {
    "ConvTranspose1d",
    "nn.conv",
    kConvInputs,
    3,
    kSingleOutput,
    1,
    "Applies a 1D transposed convolution to channel-first inputs [batch, channels, frames].",
};

core::TensorValue ensure_f32(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value) {
    if (value.type == GGML_TYPE_F32) {
        return value;
    }
    return core::wrap_tensor(ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F32), value.shape, GGML_TYPE_F32);
}

core::TensorValue regular_conv_weight(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & weight,
    const char * module_name) {
    const auto contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, weight);
    if (contiguous.type == GGML_TYPE_F32 || contiguous.type == GGML_TYPE_F16) {
        return contiguous;
    }
    if (contiguous.type == GGML_TYPE_BF16) {
        return core::wrap_tensor(ggml_cast(ctx.ggml, contiguous.tensor, GGML_TYPE_F16), contiguous.shape, GGML_TYPE_F16);
    }
    if (ggml_is_quantized(contiguous.type)) {
        return core::wrap_tensor(ggml_cast(ctx.ggml, contiguous.tensor, GGML_TYPE_F32), contiguous.shape, GGML_TYPE_F32);
    }
    throw std::runtime_error(
        std::string(module_name) + " does not support weight type with the current ggml conv path: " +
        ggml_type_name(contiguous.type));
}

core::TensorValue conv_transpose1d_weight(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & weight) {
    const auto contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, weight);
    if (contiguous.type == GGML_TYPE_F32) {
        return contiguous;
    }
    if (contiguous.type == GGML_TYPE_F16 && core::uses_host_graph_plan(ctx.backend_type)) {
        return contiguous;
    }
    if (contiguous.type == GGML_TYPE_BF16 && core::uses_host_graph_plan(ctx.backend_type)) {
        return core::wrap_tensor(ggml_cast(ctx.ggml, contiguous.tensor, GGML_TYPE_F16), contiguous.shape, GGML_TYPE_F16);
    }
    if (contiguous.type == GGML_TYPE_F16 || contiguous.type == GGML_TYPE_BF16) {
        return core::wrap_tensor(ggml_cast(ctx.ggml, contiguous.tensor, GGML_TYPE_F32), contiguous.shape, GGML_TYPE_F32);
    }
    if (ggml_is_quantized(contiguous.type)) {
        return core::wrap_tensor(ggml_cast(ctx.ggml, contiguous.tensor, GGML_TYPE_F32), contiguous.shape, GGML_TYPE_F32);
    }
    throw std::runtime_error(
        std::string("ConvTranspose1dModule does not support weight type with the current ggml conv-transpose path: ") +
        ggml_type_name(contiguous.type));
}

core::TensorValue depthwise_conv2d_weight(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & weight) {
    const auto contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, weight);
    if (contiguous.type == GGML_TYPE_F32) {
        return contiguous;
    }
    if (contiguous.type == GGML_TYPE_F16 || contiguous.type == GGML_TYPE_BF16 || ggml_is_quantized(contiguous.type)) {
        return core::wrap_tensor(ggml_cast(ctx.ggml, contiguous.tensor, GGML_TYPE_F32), contiguous.shape, GGML_TYPE_F32);
    }
    throw std::runtime_error(
        std::string("DepthwiseConv2dModule does not support weight type with the current ggml depthwise conv path: ") +
        ggml_type_name(contiguous.type));
}

int64_t conv1d_output_frames(const Conv1dConfig & config, int64_t input_frames) {
    return (input_frames + 2 * config.padding - config.dilation * (config.kernel_size - 1) - 1) / config.stride + 1;
}

int64_t conv2d_output_dim(int64_t input, int kernel, int stride, int padding, int dilation) {
    return (input + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

int64_t conv_transpose1d_output_frames(const ConvTranspose1dConfig & config, int64_t input_frames) {
    return (input_frames - 1) * config.stride - 2 * config.padding + config.dilation * (config.kernel_size - 1) + 1;
}

core::TensorValue add_bias_if_needed(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & output,
    int64_t out_channels,
    const std::optional<core::TensorValue> & bias) {
    if (!bias.has_value()) {
        return output;
    }

    core::validate_shape(*bias, core::TensorShape::from_dims({out_channels}), "bias");
    const auto output_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, output);
    const auto bias_view = core::reshape_tensor(ctx, *bias, core::TensorShape::from_dims({1, out_channels, 1}));
    const auto bias_expanded = core::wrap_tensor(ggml_repeat(ctx.ggml, bias_view.tensor, output_contiguous.tensor), output.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_add(ctx.ggml, output_contiguous.tensor, bias_expanded.tensor), output.shape, GGML_TYPE_F32);
}

core::TensorValue add_4d_channel_bias_if_needed(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & output,
    int64_t channels,
    const std::optional<core::TensorValue> & bias) {
    if (!bias.has_value()) {
        return output;
    }
    core::validate_shape(*bias, core::TensorShape::from_dims({channels}), "bias");
    const auto output_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, output);
    const auto bias_view = core::reshape_tensor(ctx, *bias, core::TensorShape::from_dims({1, channels, 1, 1}));
    const auto bias_expanded =
        core::wrap_tensor(ggml_repeat(ctx.ggml, bias_view.tensor, output_contiguous.tensor), output.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_add(ctx.ggml, output_contiguous.tensor, bias_expanded.tensor), output.shape, GGML_TYPE_F32);
}

core::TensorValue build_conv_transpose1d_cuda_col2im_path(
    core::ModuleBuildContext & ctx,
    const ConvTranspose1dConfig & config,
    const core::TensorValue & input,
    const ConvTranspose1dWeights & weights,
    const core::TensorShape & output_shape) {
    if (!is_conv_transpose1d_col2im_fast_path_eligible(ctx, config)) {
        throw std::runtime_error("ConvTranspose1dModule CUDA col2im path was requested for an ineligible config");
    }
    const auto input_contiguous = core::ensure_backend_addressable_layout(ctx, input);
    auto weight_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, weights.weight);
    if (weight_contiguous.type != GGML_TYPE_F32) {
        weight_contiguous = core::wrap_tensor(ggml_cast(ctx.ggml, weight_contiguous.tensor, GGML_TYPE_F32), weight_contiguous.shape, GGML_TYPE_F32);
    }
    auto * weight_perm = ggml_reshape_2d(
        ctx.ggml,
        ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, weight_contiguous.tensor, 1, 2, 0, 3)),
        config.in_channels,
        config.kernel_size * config.out_channels);
    ggml_tensor * bias_matrix = nullptr;
    if (config.use_bias) {
        if (!weights.bias.has_value()) {
            throw std::runtime_error("ConvTranspose1dModule col2im fast path requires bias when use_bias is true");
        }
        core::validate_shape(*weights.bias, core::TensorShape::from_dims({config.out_channels}), "bias");
        bias_matrix = ggml_reshape_2d(ctx.ggml, weights.bias->tensor, 1, config.out_channels);
    }
    core::TensorValue output;
    for (int64_t batch_index = 0; batch_index < input.shape.dims[0]; ++batch_index) {
        auto * batch_input = ggml_view_2d(
            ctx.ggml,
            input_contiguous.tensor,
            input_contiguous.tensor->ne[0],
            input_contiguous.tensor->ne[1],
            input_contiguous.tensor->nb[1],
            static_cast<size_t>(batch_index) * input_contiguous.tensor->nb[2]);
        auto * transposed_input = ggml_cont(ctx.ggml, ggml_transpose(ctx.ggml, batch_input));
        auto * columns = ggml_mul_mat(ctx.ggml, weight_perm, transposed_input);
        auto * batch_output = ggml_col2im_1d(
            ctx.ggml,
            columns,
            config.stride,
            static_cast<int>(config.out_channels),
            config.padding);
        if (bias_matrix != nullptr) {
            batch_output = ggml_add(ctx.ggml, batch_output, bias_matrix);
        }
        auto batch_value = core::wrap_tensor(
            ggml_reshape_3d(ctx.ggml, batch_output, batch_output->ne[0], batch_output->ne[1], 1),
            core::TensorShape::from_dims({1, config.out_channels, batch_output->ne[0]}),
            GGML_TYPE_F32);
        if (batch_value.shape.dims[2] != output_shape.dims[2]) {
            throw std::runtime_error("ConvTranspose1dModule col2im fast path produced unexpected frame count");
        }
        output = output.valid() ? ConcatModule({0}).build(ctx, output, batch_value) : batch_value;
    }
    return output;
}

core::TensorValue view_batch_matrix(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t batch_index,
    int64_t channels,
    int64_t frames) {
    auto * view = ggml_view_2d(
        ctx.ggml,
        input.tensor,
        frames,
        channels,
        input.tensor->nb[1],
        static_cast<size_t>(batch_index) * input.tensor->nb[2]);
    return core::wrap_tensor(view, core::TensorShape::from_dims({channels, frames}), input.type);
}

}  // namespace

bool is_conv_transpose1d_col2im_fast_path_eligible(
    const core::ModuleBuildContext & ctx,
    const ConvTranspose1dConfig & config) noexcept {
    return ctx.backend_type == core::BackendType::Cuda && config.dilation == 1;
}

Conv1dModule::Conv1dModule(Conv1dConfig config) : config_(config) {
    if (config_.in_channels <= 0 || config_.out_channels <= 0 || config_.kernel_size <= 0) {
        throw std::runtime_error("Conv1dConfig dimensions must be positive");
    }
    if (config_.stride <= 0 || config_.dilation <= 0) {
        throw std::runtime_error("Conv1d stride and dilation must be positive");
    }
}

const Conv1dConfig & Conv1dModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & Conv1dModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue Conv1dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Conv1dWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 3, 3, "input");
    core::validate_shape(
        input,
        core::TensorShape::from_dims({input.shape.dims[0], config_.in_channels, input.shape.dims[2]}),
        "input");
    core::validate_shape(
        weights.weight,
        core::TensorShape::from_dims({config_.out_channels, config_.in_channels, config_.kernel_size}),
        "weight");

    const auto output_shape = core::TensorShape::from_dims(
        {input.shape.dims[0], config_.out_channels, conv1d_output_frames(config_, input.shape.dims[2])});
    const auto input_contiguous = ensure_f32(ctx, tensor_layout::ensure_contiguous_layout_if_needed(ctx, input));
    const auto weight_contiguous = regular_conv_weight(ctx, weights.weight, "Conv1dModule");
    core::TensorValue output;
    if (input.shape.dims[0] == 1) {
        output = core::wrap_tensor(
            ggml_conv_1d(
                ctx.ggml,
                weight_contiguous.tensor,
                input_contiguous.tensor,
                config_.stride,
                config_.padding,
                config_.dilation),
            output_shape,
            GGML_TYPE_F32);
    } else {
        for (int64_t batch_index = 0; batch_index < input.shape.dims[0]; ++batch_index) {
            const auto matrix_input = view_batch_matrix(
                ctx,
                input_contiguous,
                batch_index,
                config_.in_channels,
                input.shape.dims[2]);
            auto batch_output = core::wrap_tensor(
                ggml_conv_1d(
                    ctx.ggml,
                    weight_contiguous.tensor,
                    matrix_input.tensor,
                    config_.stride,
                    config_.padding,
                    config_.dilation),
                core::TensorShape::from_dims({1, config_.out_channels, output_shape.dims[2]}),
                GGML_TYPE_F32);
            output = output.valid() ? ConcatModule({0}).build(ctx, output, batch_output) : batch_output;
        }
    }
    if (config_.use_bias) {
        output = add_bias_if_needed(ctx, output, config_.out_channels, weights.bias);
    }
    return output;
}

const core::ModuleSchema & Conv1dModule::static_schema() noexcept {
    return kConv1dSchema;
}

Conv2dModule::Conv2dModule(Conv2dConfig config) : config_(config) {
    if (config_.in_channels <= 0 || config_.out_channels <= 0 || config_.kernel_height <= 0 || config_.kernel_width <= 0) {
        throw std::runtime_error("Conv2dConfig dimensions must be positive");
    }
    if (config_.stride_height <= 0 || config_.stride_width <= 0 || config_.dilation_height <= 0 || config_.dilation_width <= 0) {
        throw std::runtime_error("Conv2d stride and dilation must be positive");
    }
}

const Conv2dConfig & Conv2dModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & Conv2dModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue Conv2dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Conv2dWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 4, 4, "input");
    core::validate_shape(
        input,
        core::TensorShape::from_dims({input.shape.dims[0], config_.in_channels, input.shape.dims[2], input.shape.dims[3]}),
        "input");
    core::validate_shape(
        weights.weight,
        core::TensorShape::from_dims({config_.out_channels, config_.in_channels, config_.kernel_height, config_.kernel_width}),
        "weight");

    const auto output_shape = core::TensorShape::from_dims({
        input.shape.dims[0],
        config_.out_channels,
        conv2d_output_dim(
            input.shape.dims[2],
            static_cast<int>(config_.kernel_height),
            config_.stride_height,
            config_.padding_height,
            config_.dilation_height),
        conv2d_output_dim(
            input.shape.dims[3],
            static_cast<int>(config_.kernel_width),
            config_.stride_width,
            config_.padding_width,
            config_.dilation_width),
    });
    const auto input_contiguous = ensure_f32(ctx, tensor_layout::ensure_contiguous_layout_if_needed(ctx, input));
    const auto weight_contiguous = regular_conv_weight(ctx, weights.weight, "Conv2dModule");
    auto output = core::wrap_tensor(
        ggml_conv_2d(
            ctx.ggml,
            weight_contiguous.tensor,
            input_contiguous.tensor,
            config_.stride_width,
            config_.stride_height,
            config_.padding_width,
            config_.padding_height,
            config_.dilation_width,
            config_.dilation_height),
        output_shape,
        GGML_TYPE_F32);
    if (config_.use_bias) {
        if (!weights.bias.has_value()) {
            throw std::runtime_error("bias is required when Conv2dConfig.use_bias is true");
        }
        const auto output_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, output);
        core::validate_shape(*weights.bias, core::TensorShape::from_dims({config_.out_channels}), "bias");
        const auto bias_view = core::reshape_tensor(ctx, *weights.bias, core::TensorShape::from_dims({1, config_.out_channels, 1, 1}));
        const auto bias_expanded =
            core::wrap_tensor(ggml_repeat(ctx.ggml, bias_view.tensor, output_contiguous.tensor), output.shape, GGML_TYPE_F32);
        output = core::wrap_tensor(ggml_add(ctx.ggml, output_contiguous.tensor, bias_expanded.tensor), output.shape, GGML_TYPE_F32);
    }
    return output;
}

const core::ModuleSchema & Conv2dModule::static_schema() noexcept {
    return kConv2dSchema;
}

DepthwiseConv2dModule::DepthwiseConv2dModule(DepthwiseConv2dConfig config) : config_(config) {
    if (config_.channels <= 0 || config_.kernel_height <= 0 || config_.kernel_width <= 0) {
        throw std::runtime_error("DepthwiseConv2dConfig dimensions must be positive");
    }
    if (config_.stride_height <= 0 || config_.stride_width <= 0 || config_.dilation_height <= 0 || config_.dilation_width <= 0) {
        throw std::runtime_error("DepthwiseConv2d stride and dilation must be positive");
    }
}

const DepthwiseConv2dConfig & DepthwiseConv2dModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & DepthwiseConv2dModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue DepthwiseConv2dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Conv2dWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 4, 4, "input");
    core::validate_shape(
        input,
        core::TensorShape::from_dims({input.shape.dims[0], config_.channels, input.shape.dims[2], input.shape.dims[3]}),
        "input");
    core::validate_shape(
        weights.weight,
        core::TensorShape::from_dims({config_.channels, 1, config_.kernel_height, config_.kernel_width}),
        "weight");

    const auto output_shape = core::TensorShape::from_dims({
        input.shape.dims[0],
        config_.channels,
        conv2d_output_dim(
            input.shape.dims[2],
            static_cast<int>(config_.kernel_height),
            config_.stride_height,
            config_.padding_height,
            config_.dilation_height),
        conv2d_output_dim(
            input.shape.dims[3],
            static_cast<int>(config_.kernel_width),
            config_.stride_width,
            config_.padding_width,
            config_.dilation_width),
    });
    const auto input_contiguous = ensure_f32(ctx, tensor_layout::ensure_contiguous_layout_if_needed(ctx, input));
    const auto weight_contiguous = depthwise_conv2d_weight(ctx, weights.weight);
    auto output = core::wrap_tensor(
        ggml_conv_2d_dw_direct(
            ctx.ggml,
            weight_contiguous.tensor,
            input_contiguous.tensor,
            config_.stride_width,
            config_.stride_height,
            config_.padding_width,
            config_.padding_height,
            config_.dilation_width,
            config_.dilation_height),
        output_shape,
        GGML_TYPE_F32);
    if (config_.use_bias) {
        if (!weights.bias.has_value()) {
            throw std::runtime_error("bias is required when DepthwiseConv2dConfig.use_bias is true");
        }
        output = add_4d_channel_bias_if_needed(ctx, output, config_.channels, weights.bias);
    }
    return output;
}

const core::ModuleSchema & DepthwiseConv2dModule::static_schema() noexcept {
    return kDepthwiseConv2dSchema;
}

ConvTranspose1dModule::ConvTranspose1dModule(ConvTranspose1dConfig config) : config_(config) {
    if (config_.in_channels <= 0 || config_.out_channels <= 0 || config_.kernel_size <= 0) {
        throw std::runtime_error("ConvTranspose1dConfig dimensions must be positive");
    }
    if (config_.stride <= 0 || config_.dilation <= 0) {
        throw std::runtime_error("ConvTranspose1d stride and dilation must be positive");
    }
}

const ConvTranspose1dConfig & ConvTranspose1dModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & ConvTranspose1dModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue ConvTranspose1dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ConvTranspose1dWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 3, 3, "input");
    core::validate_shape(
        input,
        core::TensorShape::from_dims({input.shape.dims[0], config_.in_channels, input.shape.dims[2]}),
        "input");
    core::validate_shape(
        weights.weight,
        core::TensorShape::from_dims({config_.in_channels, config_.out_channels, config_.kernel_size}),
        "weight");
    const auto output_shape = core::TensorShape::from_dims(
        {input.shape.dims[0], config_.out_channels, conv_transpose1d_output_frames(config_, input.shape.dims[2])});
    if (is_conv_transpose1d_col2im_fast_path_eligible(ctx, config_)) {
        return build_conv_transpose1d_cuda_col2im_path(ctx, config_, input, weights, output_shape);
    }
    const auto input_contiguous = ensure_f32(ctx, tensor_layout::ensure_contiguous_layout_if_needed(ctx, input));
    const auto weight_contiguous = conv_transpose1d_weight(ctx, weights.weight);
    core::TensorValue output;
    for (int64_t batch_index = 0; batch_index < input.shape.dims[0]; ++batch_index) {
        const auto matrix_input = view_batch_matrix(
            ctx,
            input_contiguous,
            batch_index,
            config_.in_channels,
            input.shape.dims[2]);
        auto batch_output = core::wrap_tensor(
            ggml_conv_transpose_1d(
                ctx.ggml,
                weight_contiguous.tensor,
                matrix_input.tensor,
                config_.stride,
                config_.padding,
                config_.dilation),
            core::TensorShape::from_dims({1, config_.out_channels, output_shape.dims[2]}),
            GGML_TYPE_F32);
        if (!output.valid()) {
            output = batch_output;
        } else {
            output = ConcatModule({0}).build(ctx, output, batch_output);
        }
    }
    if (config_.use_bias) {
        output = add_bias_if_needed(ctx, output, config_.out_channels, weights.bias);
    }
    return output;
}

const core::ModuleSchema & ConvTranspose1dModule::static_schema() noexcept {
    return kConvTranspose1dSchema;
}

}  // namespace engine::modules
