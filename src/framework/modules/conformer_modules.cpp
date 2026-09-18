#include "engine/framework/modules/conformer_modules.h"

#include "tensor_layout_utils.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <stdexcept>

namespace engine::modules {

namespace {

int64_t calc_subsampled_dim(int64_t input_dim, int64_t total_padding, int64_t kernel_size, int layers, int stride) {
    int64_t value = input_dim;
    for (int i = 0; i < layers; ++i) {
        value = (value + total_padding - kernel_size) / stride + 1;
    }
    return value;
}

core::TensorValue calc_subsampled_lengths(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lengths,
    int64_t total_padding,
    int64_t kernel_size,
    int layers,
    int stride) {
    auto value = core::wrap_tensor(ggml_cast(ctx.ggml, lengths.tensor, GGML_TYPE_F32), lengths.shape, GGML_TYPE_F32);
    for (int i = 0; i < layers; ++i) {
        auto ones = core::wrap_tensor(ggml_div(ctx.ggml, value.tensor, value.tensor), value.shape, GGML_TYPE_F32);
        auto add_pad = core::wrap_tensor(
            ggml_scale(ctx.ggml, ones.tensor, static_cast<float>(total_padding - kernel_size)),
            value.shape,
            GGML_TYPE_F32);
        auto one = core::wrap_tensor(ggml_scale(ctx.ggml, ones.tensor, 1.0f), value.shape, GGML_TYPE_F32);
        auto stride_value = core::wrap_tensor(
            ggml_scale(ctx.ggml, ones.tensor, static_cast<float>(stride)),
            value.shape,
            GGML_TYPE_F32);
        value = core::wrap_tensor(ggml_add(ctx.ggml, value.tensor, add_pad.tensor), value.shape, GGML_TYPE_F32);
        value = core::wrap_tensor(ggml_div(ctx.ggml, value.tensor, stride_value.tensor), value.shape, GGML_TYPE_F32);
        value = core::wrap_tensor(ggml_add(ctx.ggml, value.tensor, one.tensor), value.shape, GGML_TYPE_F32);
        value = core::wrap_tensor(ggml_floor(ctx.ggml, value.tensor), value.shape, GGML_TYPE_F32);
    }
    return core::wrap_tensor(ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_I32), lengths.shape, GGML_TYPE_I32);
}

core::TensorValue build_time_mask_4d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & lengths) {
    core::validate_rank_between(input, 4, 4, "time_mask.input");
    core::validate_shape(lengths, core::TensorShape::from_dims({input.shape.dims[0]}), "time_mask.lengths");

    const int64_t batch = input.shape.dims[0];
    const int64_t channels = input.shape.dims[1];
    const int64_t frames = input.shape.dims[2];
    const int64_t features = input.shape.dims[3];

    auto lengths_f32 = core::wrap_tensor(ggml_cast(ctx.ggml, lengths.tensor, GGML_TYPE_F32), lengths.shape, GGML_TYPE_F32);
    auto lengths_4d = core::reshape_tensor(ctx, lengths_f32, core::TensorShape::from_dims({batch, 1, 1, 1}));
    lengths_4d = RepeatModule(RepeatConfig{core::TensorShape::from_dims({batch, 1, frames, 1})}).build(ctx, lengths_4d);

    auto positions = core::wrap_tensor(ggml_arange(ctx.ggml, 0.5f, static_cast<float>(frames) + 0.5f, 1.0f), core::TensorShape::from_dims({frames}), GGML_TYPE_F32);
    positions = core::reshape_tensor(ctx, positions, core::TensorShape::from_dims({1, 1, frames, 1}));
    positions = RepeatModule(RepeatConfig{core::TensorShape::from_dims({batch, 1, frames, 1})}).build(ctx, positions);

    auto diff = core::wrap_tensor(ggml_add(ctx.ggml, lengths_4d.tensor, ggml_scale(ctx.ggml, positions.tensor, -1.0f)), lengths_4d.shape, GGML_TYPE_F32);
    auto mask = core::wrap_tensor(ggml_step(ctx.ggml, diff.tensor), diff.shape, GGML_TYPE_F32);
    return RepeatModule(RepeatConfig{core::TensorShape::from_dims({batch, channels, frames, features})}).build(ctx, mask);
}

core::TensorValue apply_lengths_time_mask_4d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & lengths) {
    auto mask = build_time_mask_4d(ctx, input, lengths);
    return core::wrap_tensor(ggml_mul(ctx.ggml, input.tensor, mask.tensor), input.shape, GGML_TYPE_F32);
}

core::TensorValue apply_channel_affine_btc(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const decltype(ConformerConvModuleWeights::depthwise_norm) & weights) {
    auto contiguous_input = tensor_layout::ensure_contiguous_nontransposed_layout_if_needed(ctx, input);
    core::validate_rank_between(contiguous_input, 3, 3, "channel_affine.input");
    const auto scale_view = core::reshape_tensor(
        ctx,
        weights.scale,
        core::TensorShape::from_dims({1, 1, contiguous_input.shape.dims[2]}));
    const auto bias_view = core::reshape_tensor(
        ctx,
        weights.bias,
        core::TensorShape::from_dims({1, 1, contiguous_input.shape.dims[2]}));
    const auto scale = core::wrap_tensor(
        ggml_repeat(ctx.ggml, scale_view.tensor, contiguous_input.tensor),
        contiguous_input.shape,
        GGML_TYPE_F32);
    const auto bias = core::wrap_tensor(
        ggml_repeat(ctx.ggml, bias_view.tensor, contiguous_input.tensor),
        contiguous_input.shape,
        GGML_TYPE_F32);
    const auto scaled = core::wrap_tensor(
        ggml_mul(ctx.ggml, contiguous_input.tensor, scale.tensor),
        contiguous_input.shape,
        GGML_TYPE_F32);
    return core::wrap_tensor(ggml_add(ctx.ggml, scaled.tensor, bias.tensor), contiguous_input.shape, GGML_TYPE_F32);
}

}

ConvSubsamplingModule::ConvSubsamplingModule(ConvSubsamplingConfig config) : config_(config) {
    if (config_.input_features <= 0 || config_.output_features <= 0 || config_.conv_channels <= 0) {
        throw std::runtime_error("ConvSubsamplingConfig dimensions must be positive");
    }
    if (config_.subsampling_factor != 4) {
        throw std::runtime_error("ConvSubsamplingModule currently supports subsampling_factor == 4");
    }
}

ConvSubsamplingOutputs ConvSubsamplingModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & lengths,
    const ConvSubsamplingWeights & weights) const {
    core::validate_rank_between(input, 3, 3, "input");
    core::validate_last_dim(input, config_.input_features, "input");

    const int64_t batch = input.shape.dims[0];
    const int64_t frames = input.shape.dims[1];
    const int64_t total_padding = config_.kernel_size - 1;
    const int64_t feat_after = calc_subsampled_dim(config_.input_features, total_padding, config_.kernel_size, 2, config_.stride);
    const int64_t frames_after = calc_subsampled_dim(frames, total_padding, config_.kernel_size, 2, config_.stride);
    auto current_lengths = lengths;

    auto x4 = core::reshape_tensor(ctx, input, core::TensorShape::from_dims({batch, 1, frames, config_.input_features}));
    x4 = apply_lengths_time_mask_4d(ctx, x4, current_lengths);
    x4 = Conv2dModule({
        1,
        config_.conv_channels,
        config_.kernel_size,
        config_.kernel_size,
        config_.stride,
        config_.stride,
        static_cast<int>(total_padding / 2),
        static_cast<int>(total_padding / 2),
        1,
        1,
        config_.use_bias,
    }).build(ctx, x4, weights.conv0);
    x4 = ReluModule().build(ctx, x4);
    current_lengths = calc_subsampled_lengths(ctx, current_lengths, total_padding, config_.kernel_size, 1, config_.stride);
    x4 = apply_lengths_time_mask_4d(ctx, x4, current_lengths);
    x4 = Conv2dModule({
        config_.conv_channels,
        config_.conv_channels,
        config_.kernel_size,
        config_.kernel_size,
        config_.stride,
        config_.stride,
        static_cast<int>(total_padding / 2),
        static_cast<int>(total_padding / 2),
        1,
        1,
        config_.use_bias,
    }).build(ctx, x4, weights.conv1);
    x4 = ReluModule().build(ctx, x4);
    current_lengths = calc_subsampled_lengths(ctx, current_lengths, total_padding, config_.kernel_size, 1, config_.stride);
    x4 = apply_lengths_time_mask_4d(ctx, x4, current_lengths);
    x4 = tensor_layout::swap_channel_time_axes_4d(ctx, x4);
    x4 = core::wrap_tensor(ggml_cont(ctx.ggml, x4.tensor), x4.shape, x4.type);
    auto flat = core::reshape_tensor(ctx, x4, core::TensorShape::from_dims({batch, frames_after, config_.conv_channels * feat_after}));
    auto output = LinearModule({config_.conv_channels * feat_after, config_.output_features, config_.use_bias}).build(ctx, flat, weights.linear);
    return {
        output,
        current_lengths,
    };
}

DepthwiseConvSubsamplingModule::DepthwiseConvSubsamplingModule(DepthwiseConvSubsamplingConfig config)
    : config_(config) {
    if (config.input_features <= 0 || config.output_features <= 0 || config.conv_channels <= 0 ||
        config.kernel_size <= 0 || config.stride <= 0 || config.padding < 0) {
        throw std::runtime_error("DepthwiseConvSubsampling requires positive dimensions and nonnegative padding");
    }
}

core::TensorValue DepthwiseConvSubsamplingModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const DepthwiseConvSubsamplingWeights & weights,
    const std::vector<core::TensorValue> & stage_keep_masks) const {
    core::validate_rank_between(input, 3, 3, "subsampling.input");
    core::validate_last_dim(input, config_.input_features, "subsampling.input");
    if (!stage_keep_masks.empty() && stage_keep_masks.size() != weights.stages.size() + 1) {
        throw std::runtime_error("DepthwiseConvSubsampling requires one keep mask per downsampling stage");
    }
    const auto channels = config_.conv_channels;
    const int k = config_.kernel_size, s = config_.stride, p = config_.padding;
    auto x = core::reshape_tensor(ctx, input,
        core::TensorShape::from_dims({input.shape.dims[0], 1, input.shape.dims[1], input.shape.dims[2]}));
    x = Conv2dModule({1, channels, k, k, s, s, p, p, 1, 1, config_.use_bias}).build(ctx, x, weights.input_conv);
    x = ReluModule().build(ctx, x);
    if (!stage_keep_masks.empty()) {
        x = TimeMask4dModule().build(ctx, x, stage_keep_masks[0]);
    }
    for (size_t i = 0; i < weights.stages.size(); ++i) {
        x = DepthwiseConv2dModule({channels, k, k, s, s, p, p, 1, 1, config_.use_bias})
            .build(ctx, x, weights.stages[i].depthwise);
        if (!stage_keep_masks.empty()) {
            x = TimeMask4dModule().build(ctx, x, stage_keep_masks[i + 1]);
        }
        x = Conv2dModule({channels, channels, 1, 1, 1, 1, 0, 0, 1, 1, config_.use_bias})
            .build(ctx, x, weights.stages[i].pointwise);
        x = ReluModule().build(ctx, x);
        if (!stage_keep_masks.empty()) {
            x = TimeMask4dModule().build(ctx, x, stage_keep_masks[i + 1]);
        }
    }
    x = tensor_layout::swap_channel_time_axes_4d(ctx, x);
    x = core::wrap_tensor(ggml_cont(ctx.ggml, x.tensor), x.shape, x.type);
    const int64_t flat_features = x.shape.dims[2] * x.shape.dims[3];
    x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({x.shape.dims[0], x.shape.dims[1], flat_features}));
    return LinearModule({flat_features, config_.output_features, config_.use_bias}).build(ctx, x, weights.projection);
}

ConformerConvModule::ConformerConvModule(ConformerConvModuleConfig config) : config_(config) {}

core::TensorValue ConformerConvModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ConformerConvModuleWeights & weights,
    const std::optional<core::TensorValue> & keep_mask) const {
    auto x = LayerNormModule({config_.hidden_size, config_.eps, true, true}).build(ctx, input, weights.norm);
    x = LinearModule({config_.hidden_size, config_.hidden_size * 2, config_.use_bias}).build(ctx, x, weights.pointwise_in);
    x = GLUModule({config_.contiguous_glu_gate}).build(ctx, x);
    if (keep_mask.has_value()) {
        x = MaskingModule().build(ctx, x, *keep_mask);
    }
    x = tensor_layout::swap_channel_time_axes_3d(ctx, x);
    x = DepthwiseConv1dModule({config_.hidden_size, config_.kernel_size, 1, static_cast<int>(config_.kernel_size / 2), 1, config_.use_bias})
            .build(ctx, x, weights.depthwise);
    x = tensor_layout::swap_channel_time_axes_3d(ctx, x);
    x = apply_channel_affine_btc(ctx, x, weights.depthwise_norm);
    x = SiluModule().build(ctx, x);
    return LinearModule({config_.hidden_size, config_.hidden_size, config_.use_bias}).build(ctx, x, weights.pointwise_out);
}

StreamingConformerConvModule::StreamingConformerConvModule(ConformerConvModuleConfig config) : config_(config) {}

StreamingConformerConvOutputs StreamingConformerConvModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ConformerConvModuleWeights & weights,
    const std::optional<core::TensorValue> & prefix_cache,
    const std::optional<core::TensorValue> & keep_mask) const {
    auto x = LayerNormModule({config_.hidden_size, config_.eps, true, true}).build(ctx, input, weights.norm);
    x = LinearModule({config_.hidden_size, config_.hidden_size * 2, config_.use_bias}).build(ctx, x, weights.pointwise_in);
    x = GLUModule({config_.contiguous_glu_gate}).build(ctx, x);
    if (keep_mask.has_value()) {
        x = MaskingModule().build(ctx, x, *keep_mask);
    }
    x = tensor_layout::swap_channel_time_axes_3d(ctx, x);

    const int64_t left_context = config_.kernel_size / 2;
    const int64_t right_context = config_.kernel_size - 1 - left_context;

    core::TensorValue conv_input = x;
    if (right_context > 0) {
        auto last = SliceModule({2, x.shape.dims[2] - 1, 1}).build(ctx, x);
        auto right_zeros = RepeatModule({core::TensorShape::from_dims({x.shape.dims[0], x.shape.dims[1], right_context})}).build(ctx, last);
        right_zeros = core::wrap_tensor(ggml_scale(ctx.ggml, right_zeros.tensor, 0.0f), right_zeros.shape, GGML_TYPE_F32);
        conv_input = ConcatModule({2}).build(ctx, conv_input, right_zeros);
    }
    if (prefix_cache.has_value()) {
        conv_input = ConcatModule({2}).build(ctx, *prefix_cache, conv_input);
    } else if (left_context > 0) {
        auto first = SliceModule({2, 0, 1}).build(ctx, x);
        auto left_zeros = RepeatModule({core::TensorShape::from_dims({x.shape.dims[0], x.shape.dims[1], left_context})}).build(ctx, first);
        left_zeros = core::wrap_tensor(ggml_scale(ctx.ggml, left_zeros.tensor, 0.0f), left_zeros.shape, GGML_TYPE_F32);
        conv_input = ConcatModule({2}).build(ctx, left_zeros, conv_input);
    }

    core::TensorValue next_cache;
    if (prefix_cache.has_value()) {
        auto cache_source = conv_input;
        if (config_.cache_drop_size > 0) {
            const int64_t keep_frames = conv_input.shape.dims[2] - config_.cache_drop_size;
            cache_source = SliceModule({2, 0, keep_frames}).build(ctx, conv_input);
        }
        next_cache = SliceModule({2, cache_source.shape.dims[2] - prefix_cache->shape.dims[2], prefix_cache->shape.dims[2]}).build(ctx, cache_source);
    } else if (left_context > 0) {
        next_cache = SliceModule({2, conv_input.shape.dims[2] - left_context, left_context}).build(ctx, conv_input);
    }

    auto y = DepthwiseConv1dModule({config_.hidden_size, config_.kernel_size, 1, 0, 1, config_.use_bias}).build(ctx, conv_input, weights.depthwise);
    y = tensor_layout::swap_channel_time_axes_3d(ctx, y);
    y = apply_channel_affine_btc(ctx, y, weights.depthwise_norm);
    y = SiluModule().build(ctx, y);
    y = LinearModule({config_.hidden_size, config_.hidden_size, config_.use_bias}).build(ctx, y, weights.pointwise_out);
    return {y, next_cache};
}

ConformerBlockModule::ConformerBlockModule(ConformerBlockConfig config) : config_(config) {}

core::TensorValue ConformerBlockModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ConformerBlockWeights & weights) const {
    auto x = input;
    x = LayerNormModule({config_.hidden_size, config_.eps, true, true}).build(ctx, x, weights.ffn1_norm);
    auto ff1 = LinearModule({config_.hidden_size, config_.intermediate_size, config_.use_bias}).build(ctx, x, weights.ffn1_fc1);
    ff1 = SiluModule().build(ctx, ff1);
    ff1 = LinearModule({config_.intermediate_size, config_.hidden_size, config_.use_bias}).build(ctx, ff1, weights.ffn1_fc2);
    auto ff1_half = core::wrap_tensor(ggml_scale(ctx.ggml, ff1.tensor, 0.5f), ff1.shape, GGML_TYPE_F32);
    x = core::wrap_tensor(ggml_add(ctx.ggml, input.tensor, ff1_half.tensor), input.shape, GGML_TYPE_F32);

    auto y = LayerNormModule({config_.hidden_size, config_.eps, true, true}).build(ctx, x, weights.norm1);
    y = SelfAttentionModule({config_.hidden_size, config_.num_heads, config_.use_bias}).build(ctx, y, weights.self_attention);
    x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, y.tensor), x.shape, GGML_TYPE_F32);

    y = ConformerConvModule({config_.hidden_size, config_.kernel_size, config_.use_bias, config_.eps, 0, config_.contiguous_glu_gate}).build(ctx, x, weights.conv);
    x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, y.tensor), x.shape, GGML_TYPE_F32);

    y = LayerNormModule({config_.hidden_size, config_.eps, true, true}).build(ctx, x, weights.norm2);
    y = LinearModule({config_.hidden_size, config_.intermediate_size, config_.use_bias}).build(ctx, y, weights.ffn2_fc1);
    y = SiluModule().build(ctx, y);
    y = LinearModule({config_.intermediate_size, config_.hidden_size, config_.use_bias}).build(ctx, y, weights.ffn2_fc2);
    auto y_half = core::wrap_tensor(ggml_scale(ctx.ggml, y.tensor, 0.5f), y.shape, GGML_TYPE_F32);
    x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, y_half.tensor), x.shape, GGML_TYPE_F32);

    return LayerNormModule({config_.hidden_size, config_.eps, true, true}).build(ctx, x, weights.final_norm);
}

RelativeConformerBlockModule::RelativeConformerBlockModule(ConformerBlockConfig config) : config_(config) {}

core::TensorValue RelativeConformerBlockModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::optional<core::TensorValue> & pos_emb,
    const RelativeConformerBlockWeights & weights,
    const std::optional<core::TensorValue> & attention_mask,
    const std::optional<core::TensorValue> & keep_mask,
    const std::optional<core::TensorValue> & query_keep_mask,
    const std::optional<core::TensorValue> & projected_pos_emb) const {
    auto x = input;
    x = LayerNormModule({config_.hidden_size, config_.eps, true, true}).build(ctx, x, weights.ffn1_norm);
    auto ff1 = LinearModule({config_.hidden_size, config_.intermediate_size, config_.use_bias}).build(ctx, x, weights.ffn1_fc1);
    ff1 = SiluModule().build(ctx, ff1);
    ff1 = LinearModule({config_.intermediate_size, config_.hidden_size, config_.use_bias}).build(ctx, ff1, weights.ffn1_fc2);
    auto ff1_half = core::wrap_tensor(ggml_scale(ctx.ggml, ff1.tensor, 0.5f), ff1.shape, GGML_TYPE_F32);
    x = core::wrap_tensor(ggml_add(ctx.ggml, input.tensor, ff1_half.tensor), input.shape, GGML_TYPE_F32);

    auto y = LayerNormModule({config_.hidden_size, config_.eps, true, true}).build(ctx, x, weights.norm1);
    y = RelativeSelfAttentionModule({
        config_.hidden_size,
        config_.num_heads,
        config_.use_bias,
        config_.left_context,
        config_.right_context,
        config_.cache_drop_size,
    }).build(ctx, y, pos_emb, weights.self_attention, attention_mask, query_keep_mask, projected_pos_emb);
    x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, y.tensor), x.shape, GGML_TYPE_F32);

    y = ConformerConvModule({config_.hidden_size, config_.kernel_size, config_.use_bias, config_.eps, 0, config_.contiguous_glu_gate}).build(ctx, x, weights.conv, keep_mask);
    x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, y.tensor), x.shape, GGML_TYPE_F32);

    y = LayerNormModule({config_.hidden_size, config_.eps, true, true}).build(ctx, x, weights.norm2);
    y = LinearModule({config_.hidden_size, config_.intermediate_size, config_.use_bias}).build(ctx, y, weights.ffn2_fc1);
    y = SiluModule().build(ctx, y);
    y = LinearModule({config_.intermediate_size, config_.hidden_size, config_.use_bias}).build(ctx, y, weights.ffn2_fc2);
    auto y_half = core::wrap_tensor(ggml_scale(ctx.ggml, y.tensor, 0.5f), y.shape, GGML_TYPE_F32);
    x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, y_half.tensor), x.shape, GGML_TYPE_F32);

    return LayerNormModule({config_.hidden_size, config_.eps, true, true}).build(ctx, x, weights.final_norm);
}

StreamingConformerBlockModule::StreamingConformerBlockModule(ConformerBlockConfig config) : config_(config) {}

StreamingConformerBlockOutputs StreamingConformerBlockModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue &,
    const ConformerBlockWeights & weights,
    const std::optional<core::TensorValue> &,
    const std::optional<core::TensorValue> & prefix_time_cache,
    const std::optional<core::TensorValue> &) const {
    auto x = input;
    x = LayerNormModule({config_.hidden_size, config_.eps, true, true}).build(ctx, x, weights.ffn1_norm);
    auto ff1 = LinearModule({config_.hidden_size, config_.intermediate_size, config_.use_bias}).build(ctx, x, weights.ffn1_fc1);
    ff1 = SiluModule().build(ctx, ff1);
    ff1 = LinearModule({config_.intermediate_size, config_.hidden_size, config_.use_bias}).build(ctx, ff1, weights.ffn1_fc2);
    auto ff1_half = core::wrap_tensor(ggml_scale(ctx.ggml, ff1.tensor, 0.5f), ff1.shape, GGML_TYPE_F32);
    x = core::wrap_tensor(ggml_add(ctx.ggml, input.tensor, ff1_half.tensor), input.shape, GGML_TYPE_F32);

    auto y = LayerNormModule({config_.hidden_size, config_.eps, true, true}).build(ctx, x, weights.norm1);
    y = SelfAttentionModule({config_.hidden_size, config_.num_heads, config_.use_bias}).build(ctx, y, weights.self_attention);
    x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, y.tensor), x.shape, GGML_TYPE_F32);

    auto conv = StreamingConformerConvModule({config_.hidden_size, config_.kernel_size, config_.use_bias, config_.eps, config_.cache_drop_size, config_.contiguous_glu_gate}).build(
        ctx,
        x,
        weights.conv,
        prefix_time_cache);
    x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, conv.output.tensor), x.shape, GGML_TYPE_F32);

    y = LayerNormModule({config_.hidden_size, config_.eps, true, true}).build(ctx, x, weights.norm2);
    y = LinearModule({config_.hidden_size, config_.intermediate_size, config_.use_bias}).build(ctx, y, weights.ffn2_fc1);
    y = SiluModule().build(ctx, y);
    y = LinearModule({config_.intermediate_size, config_.hidden_size, config_.use_bias}).build(ctx, y, weights.ffn2_fc2);
    auto y_half = core::wrap_tensor(ggml_scale(ctx.ggml, y.tensor, 0.5f), y.shape, GGML_TYPE_F32);
    x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, y_half.tensor), x.shape, GGML_TYPE_F32);
    x = LayerNormModule({config_.hidden_size, config_.eps, true, true}).build(ctx, x, weights.final_norm);
    return {x, core::TensorValue{}, conv.next_cache};
}

StreamingRelativeConformerBlockModule::StreamingRelativeConformerBlockModule(ConformerBlockConfig config) : config_(config) {}

StreamingConformerBlockOutputs StreamingRelativeConformerBlockModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & pos_emb,
    const RelativeConformerBlockWeights & weights,
    const std::optional<core::TensorValue> & prefix_channel_cache,
    const std::optional<core::TensorValue> & prefix_time_cache,
    const std::optional<core::TensorValue> & attention_mask) const {
    auto x = input;
    x = LayerNormModule({config_.hidden_size, config_.eps, true, true}).build(ctx, x, weights.ffn1_norm);
    auto ff1 = LinearModule({config_.hidden_size, config_.intermediate_size, config_.use_bias}).build(ctx, x, weights.ffn1_fc1);
    ff1 = SiluModule().build(ctx, ff1);
    ff1 = LinearModule({config_.intermediate_size, config_.hidden_size, config_.use_bias}).build(ctx, ff1, weights.ffn1_fc2);
    auto ff1_half = core::wrap_tensor(ggml_scale(ctx.ggml, ff1.tensor, 0.5f), ff1.shape, GGML_TYPE_F32);
    x = core::wrap_tensor(ggml_add(ctx.ggml, input.tensor, ff1_half.tensor), input.shape, GGML_TYPE_F32);

    auto y = LayerNormModule({config_.hidden_size, config_.eps, true, true}).build(ctx, x, weights.norm1);
    auto attn = StreamingRelativeSelfAttentionModule({
        config_.hidden_size,
        config_.num_heads,
        config_.use_bias,
        config_.left_context,
        config_.right_context,
        config_.cache_drop_size,
    }).build(
        ctx,
        y,
        pos_emb,
        weights.self_attention,
        prefix_channel_cache,
        prefix_channel_cache,
        attention_mask);
    x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, attn.output.tensor), x.shape, GGML_TYPE_F32);

    auto conv = StreamingConformerConvModule({config_.hidden_size, config_.kernel_size, config_.use_bias, config_.eps, config_.cache_drop_size, config_.contiguous_glu_gate}).build(
        ctx,
        x,
        weights.conv,
        prefix_time_cache);
    x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, conv.output.tensor), x.shape, GGML_TYPE_F32);

    y = LayerNormModule({config_.hidden_size, config_.eps, true, true}).build(ctx, x, weights.norm2);
    y = LinearModule({config_.hidden_size, config_.intermediate_size, config_.use_bias}).build(ctx, y, weights.ffn2_fc1);
    y = SiluModule().build(ctx, y);
    y = LinearModule({config_.intermediate_size, config_.hidden_size, config_.use_bias}).build(ctx, y, weights.ffn2_fc2);
    auto y_half = core::wrap_tensor(ggml_scale(ctx.ggml, y.tensor, 0.5f), y.shape, GGML_TYPE_F32);
    x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, y_half.tensor), x.shape, GGML_TYPE_F32);
    x = LayerNormModule({config_.hidden_size, config_.eps, true, true}).build(ctx, x, weights.final_norm);
    return {x, attn.key, conv.next_cache};
}

CacheAwareStreamingConfig make_cache_aware_streaming_config(
    int64_t subsampling_factor,
    int64_t left_context,
    int64_t right_context,
    int64_t chunk_size,
    int64_t shift_size) {
    if (subsampling_factor <= 0 || chunk_size <= 0 || shift_size <= 0) {
        throw std::runtime_error("Streaming config inputs must be positive");
    }
    CacheAwareStreamingConfig config;
    config.chunk_size = chunk_size;
    config.shift_size = shift_size;
    config.cache_drop_size = chunk_size - shift_size;
    config.last_channel_cache_size = std::max<int64_t>(left_context, 0);
    config.valid_out_len = shift_size / subsampling_factor;
    config.pre_encode_cache_size = 0;
    config.drop_extra_pre_encoded = config.pre_encode_cache_size / subsampling_factor;
    if (right_context > 0 && config.cache_drop_size < right_context) {
        config.cache_drop_size = right_context;
    }
    return config;
}

}  // namespace engine::modules
