#include "engine/framework/modules/streaming_conv_modules.h"

#include "tensor_layout_utils.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/structural_modules.h"

#include <stdexcept>
#include <string>

namespace engine::modules {

namespace {

core::TensorValue ensure_f32(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
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

int64_t depthwise_conv1d_output_frames(const DepthwiseConv1dConfig & config, int64_t input_frames) {
    return (input_frames + 2 * config.padding - config.dilation * (config.kernel_size - 1) - 1) / config.stride + 1;
}

core::TensorValue add_bias_bct(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & output,
    int64_t channels,
    const std::optional<core::TensorValue> & bias) {
    if (!bias.has_value()) {
        return output;
    }
    auto output_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, output);
    auto bias_view = core::reshape_tensor(ctx, *bias, core::TensorShape::from_dims({1, channels, 1}));
    auto repeated = core::wrap_tensor(ggml_repeat(ctx.ggml, bias_view.tensor, output_contiguous.tensor), output.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_add(ctx.ggml, output_contiguous.tensor, repeated.tensor), output.shape, GGML_TYPE_F32);
}

core::TensorValue zeros_like_prefix(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t prefix_frames) {
    auto prefix = RepeatModule({core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], prefix_frames})})
                      .build(ctx, SliceModule({2, 0, 1}).build(ctx, input));
    auto prefix_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, prefix);
    return core::wrap_tensor(ggml_scale(ctx.ggml, prefix_contiguous.tensor, 0.0f), prefix.shape, GGML_TYPE_F32);
}

core::TensorValue repeat_first_frame(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t prefix_frames) {
    auto first = SliceModule({2, 0, 1}).build(ctx, input);
    return RepeatModule({core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], prefix_frames})}).build(ctx, first);
}

}

DepthwiseConv1dModule::DepthwiseConv1dModule(DepthwiseConv1dConfig config) : config_(config) {
    if (config_.channels <= 0 || config_.kernel_size <= 0) {
        throw std::runtime_error("DepthwiseConv1dConfig dimensions must be positive");
    }
    if (config_.stride <= 0 || config_.dilation <= 0) {
        throw std::runtime_error("DepthwiseConv1d stride and dilation must be positive");
    }
}

core::TensorValue DepthwiseConv1dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const DepthwiseConv1dWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 3, 3, "input");
    core::validate_shape(
        input,
        core::TensorShape::from_dims({input.shape.dims[0], config_.channels, input.shape.dims[2]}),
        "input");
    core::validate_shape(
        weights.weight,
        core::TensorShape::from_dims({config_.channels, 1, config_.kernel_size}),
        "weight");
    const auto input_contiguous = ensure_f32(ctx, tensor_layout::ensure_contiguous_layout_if_needed(ctx, input));
    const auto weight_contiguous = regular_conv_weight(ctx, weights.weight, "DepthwiseConv1dModule");
    auto input_4d = core::reshape_tensor(
        ctx,
        input_contiguous,
        core::TensorShape::from_dims({input.shape.dims[0], config_.channels, 1, input.shape.dims[2]}));
    auto weight_4d = core::reshape_tensor(
        ctx,
        weight_contiguous,
        core::TensorShape::from_dims({config_.channels, 1, 1, config_.kernel_size}));
    // This 2D depthwise lowering greatly improves performance but may affect parity.
    auto output_4d = DepthwiseConv2dModule({
        config_.channels,
        1,
        config_.kernel_size,
        1,
        config_.stride,
        0,
        config_.padding,
        1,
        config_.dilation,
        config_.use_bias,
    }).build(ctx, input_4d, {weight_4d, weights.bias});
    return core::reshape_tensor(
        ctx,
        output_4d,
        core::TensorShape::from_dims({
            input.shape.dims[0],
            config_.channels,
            depthwise_conv1d_output_frames(config_, input.shape.dims[2]),
        }));
}

PointwiseConv1dModule::PointwiseConv1dModule(PointwiseConv1dConfig config) : config_(config) {}

core::TensorValue PointwiseConv1dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const PointwiseConv1dWeights & weights) const {
    if (config_.quant) {
        core::validate_rank_between(input, 3, 3, "input");
        core::validate_shape(
            input,
            core::TensorShape::from_dims({input.shape.dims[0], config_.in_channels, input.shape.dims[2]}),
            "input");
        core::TensorValue weight = weights.weight;
        if (weight.shape.rank == 3) {
            core::validate_shape(
                weight,
                core::TensorShape::from_dims({config_.out_channels, config_.in_channels, 1}),
                "weight");
            weight = core::reshape_tensor(
                ctx,
                weight,
                core::TensorShape::from_dims({config_.out_channels, config_.in_channels}));
        } else {
            core::validate_shape(
                weight,
                core::TensorShape::from_dims({config_.out_channels, config_.in_channels}),
                "weight");
        }
        auto x = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
        x = LinearModule({config_.in_channels, config_.out_channels, config_.use_bias}).build(ctx, x, {weight, weights.bias});
        return TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    }
    return Conv1dModule({config_.in_channels, config_.out_channels, 1, 1, 0, 1, config_.use_bias}).build(ctx, input, weights);
}

StreamingConv1dModule::StreamingConv1dModule(StreamingConv1dConfig config) : config_(config) {
    if (config_.in_channels <= 0 || config_.out_channels <= 0 || config_.kernel_size <= 0) {
        throw std::runtime_error("StreamingConv1dConfig dimensions must be positive");
    }
    if (config_.stride <= 0 || config_.dilation <= 0) {
        throw std::runtime_error("StreamingConv1d stride and dilation must be positive");
    }
}

core::TensorValue StreamingConv1dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const StreamingConv1dWeights & weights) const {
    const int64_t effective_kernel = (config_.kernel_size - 1) * config_.dilation + 1;
    const int64_t left_pad = effective_kernel - config_.stride;
    if (left_pad < 0) {
        throw std::runtime_error("StreamingConv1dModule requires effective_kernel >= stride");
    }
    if (input.shape.dims[2] <= 0) {
        throw std::runtime_error("StreamingConv1dModule input must have frames");
    }

    auto padded = input;
    if (left_pad > 0) {
        core::TensorValue prefix = config_.pad_mode == StreamingPadMode::Replicate
            ? repeat_first_frame(ctx, input, left_pad)
            : zeros_like_prefix(ctx, input, left_pad);
        padded = ConcatModule({2}).build(ctx, prefix, input);
    }
    return Conv1dModule({
        config_.in_channels,
        config_.out_channels,
        config_.kernel_size,
        config_.stride,
        0,
        config_.dilation,
        config_.use_bias,
    }).build(ctx, padded, weights);
}

DepthwiseConvTranspose1dModule::DepthwiseConvTranspose1dModule(DepthwiseConvTranspose1dConfig config) : config_(config) {
    if (config_.channels <= 0 || config_.kernel_size <= 0) {
        throw std::runtime_error("DepthwiseConvTranspose1dConfig dimensions must be positive");
    }
    if (config_.stride <= 0) {
        throw std::runtime_error("DepthwiseConvTranspose1d stride must be positive");
    }
}

core::TensorValue DepthwiseConvTranspose1dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const DepthwiseConvTranspose1dWeights & weights) const {
    core::validate_shape(weights.weight, core::TensorShape::from_dims({config_.channels, 1, config_.kernel_size}), "weight");

    core::TensorValue output;
    for (int64_t channel = 0; channel < config_.channels; ++channel) {
        auto input_channel = SliceModule({1, channel, 1}).build(ctx, input);
        auto weight_channel = SliceModule({0, channel, 1}).build(ctx, weights.weight);
        auto channel_out = ConvTranspose1dModule({
            1,
            1,
            config_.kernel_size,
            config_.stride,
            0,
            1,
            false,
        }).build(ctx, input_channel, ConvTranspose1dWeights{weight_channel, std::nullopt});
        output = output.valid() ? ConcatModule({1}).build(ctx, output, channel_out) : channel_out;
    }
    return add_bias_bct(ctx, output, config_.channels, weights.bias);
}

}  // namespace engine::modules
