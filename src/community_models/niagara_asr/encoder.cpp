#include "engine/community_models/niagara_asr/encoder.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <array>
#include <cmath>
#include <stdexcept>

namespace engine::community_models::niagara_asr {
namespace {

namespace core = engine::core;
namespace modules = engine::modules;

constexpr float kL1NormEpsilon = 1.0e-6f;
constexpr int64_t kNumAttentionHeads = 4;
constexpr int64_t kRelativePositionCenter = 3000;

core::TensorShape make_last_dim_broadcast_shape(const core::TensorShape & input) {
    core::TensorShape shape = {};
    shape.rank = input.rank;
    for (size_t i = 0; i < shape.rank; ++i) {
        shape.dims[i] = 1;
    }
    shape.dims[shape.rank - 1] = input.last_dim();
    return shape;
}

core::TensorShape make_scalar_broadcast_shape(const core::TensorShape & input) {
    core::TensorShape shape = {};
    shape.rank = input.rank;
    for (size_t i = 0; i < shape.rank; ++i) {
        shape.dims[i] = 1;
    }
    return shape;
}

core::TensorValue contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    return core::wrap_tensor(ggml_cont(ctx.ggml, input.tensor), input.shape, input.type);
}

core::TensorValue transpose(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    std::array<int, core::kMaxTensorRank> axes,
    size_t rank) {
    if (input.shape.rank != rank) {
        throw std::runtime_error("Niagara transpose rank mismatch");
    }
    core::TensorShape output_shape = {};
    output_shape.rank = rank;
    std::array<int, core::kMaxTensorRank> ggml_axes = {0, 1, 2, 3};
    for (size_t out_logical_axis = 0; out_logical_axis < rank; ++out_logical_axis) {
        const int in_logical_axis = axes[out_logical_axis];
        output_shape.dims[out_logical_axis] = input.shape.dims[static_cast<size_t>(in_logical_axis)];
        const int out_ggml_axis = static_cast<int>(rank) - 1 - static_cast<int>(out_logical_axis);
        const int in_ggml_axis = core::logical_axis_to_ggml_axis(rank, in_logical_axis);
        ggml_axes[static_cast<size_t>(in_ggml_axis)] = out_ggml_axis;
    }
    return core::wrap_tensor(
        ggml_permute(ctx.ggml, input.tensor, ggml_axes[0], ggml_axes[1], ggml_axes[2], ggml_axes[3]),
        output_shape,
        input.type);
}

core::TensorValue add_last_dim_bias(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & bias) {
    core::validate_shape(bias, core::TensorShape::from_dims({input.shape.last_dim()}), "bias");
    const auto bias_view = core::reshape_tensor(ctx, bias, make_last_dim_broadcast_shape(input.shape));
    const auto repeated = core::wrap_tensor(ggml_repeat(ctx.ggml, bias_view.tensor, input.tensor), input.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_add(ctx.ggml, input.tensor, repeated.tensor), input.shape, GGML_TYPE_F32);
}

core::TensorValue add_scaled_residual(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & residual,
    const core::TensorValue & scale_logits) {
    core::validate_shape(residual, input.shape, "residual");
    core::validate_shape(scale_logits, core::TensorShape::from_dims({1}), "scale_logits");
    const auto scale = core::wrap_tensor(ggml_sigmoid(ctx.ggml, scale_logits.tensor), scale_logits.shape, GGML_TYPE_F32);
    const auto scale_view = core::reshape_tensor(ctx, scale, make_scalar_broadcast_shape(input.shape));
    const auto repeated_scale = core::wrap_tensor(ggml_repeat(ctx.ggml, scale_view.tensor, residual.tensor), input.shape, GGML_TYPE_F32);
    const auto scaled = core::wrap_tensor(ggml_mul(ctx.ggml, residual.tensor, repeated_scale.tensor), input.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_add(ctx.ggml, input.tensor, scaled.tensor), input.shape, GGML_TYPE_F32);
}

core::TensorValue reshape_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    const int64_t hidden = input.shape.last_dim();
    if (hidden % kNumAttentionHeads != 0) {
        throw std::runtime_error("Niagara attention hidden size must divide evenly by heads");
    }
    auto x = core::reshape_tensor(
        ctx,
        input,
        core::TensorShape::from_dims({
            input.shape.dims[0],
            input.shape.dims[1],
            kNumAttentionHeads,
            hidden / kNumAttentionHeads,
        }));
    return contiguous(ctx, transpose(ctx, x, {0, 2, 1, 3}, 4));
}

core::TensorValue relative_shift(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t frames) {
    auto x = modules::Pad2dModule({0, 1, 0, 0}).build(ctx, input);
    x = contiguous(ctx, x);
    x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({
        input.shape.dims[0],
        input.shape.dims[1],
        frames * frames * 2,
    }));
    auto * padded = ggml_pad_ext(
        ctx.ggml,
        x.tensor,
        0,
        static_cast<int>(frames - 1),
        0,
        0,
        0,
        0,
        0,
        0);
    x = core::wrap_tensor(
        padded,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], frames * frames * 2 + frames - 1}),
        GGML_TYPE_F32);
    x = contiguous(ctx, x);
    x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({
        input.shape.dims[0],
        input.shape.dims[1],
        frames + 1,
        frames * 2 - 1,
    }));
    x = modules::SliceModule({2, 0, frames}).build(ctx, x);
    return modules::SliceModule({3, frames - 1, frames}).build(ctx, x);
}

core::TensorValue affine_last_dim(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::NormWeights & weights,
    int64_t hidden_size) {
    if (!weights.weight.has_value() || !weights.bias.has_value()) {
        throw std::runtime_error("Niagara L1 norm requires gamma and beta");
    }
    core::validate_shape(*weights.weight, core::TensorShape::from_dims({hidden_size}), "l1norm gamma");
    core::validate_shape(*weights.bias, core::TensorShape::from_dims({hidden_size}), "l1norm beta");

    const auto weight = core::reshape_tensor(ctx, *weights.weight, make_last_dim_broadcast_shape(input.shape));
    const auto bias = core::reshape_tensor(ctx, *weights.bias, make_last_dim_broadcast_shape(input.shape));
    const auto scaled = core::wrap_tensor(ggml_mul(ctx.ggml, input.tensor, weight.tensor), input.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_add(ctx.ggml, scaled.tensor, bias.tensor), input.shape, GGML_TYPE_F32);
}

}  // namespace

core::TensorValue build_niagara_l1_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::NormWeights & weights,
    const NiagaraEncoderConfig & config) {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 2, core::kMaxTensorRank, "input");
    core::validate_last_dim(input, config.hidden_size, "input");

    const auto x = core::ensure_backend_addressable_layout(ctx, input);
    const auto mean = modules::ReduceMeanModule({-1}).build(ctx, x);
    const auto centered = core::wrap_tensor(ggml_sub(ctx.ggml, x.tensor, mean.tensor), x.shape, GGML_TYPE_F32);
    const auto abs_centered = core::wrap_tensor(ggml_abs(ctx.ggml, centered.tensor), x.shape, GGML_TYPE_F32);
    const auto l1_mean = modules::ReduceMeanModule({-1}).build(ctx, abs_centered);
    const auto denom = core::wrap_tensor(
        ggml_scale_bias(ctx.ggml, l1_mean.tensor, 1.0f, kL1NormEpsilon),
        l1_mean.shape,
        GGML_TYPE_F32);
    const auto normalized = core::wrap_tensor(ggml_div(ctx.ggml, centered.tensor, denom.tensor), x.shape, GGML_TYPE_F32);
    ggml_set_output(normalized.tensor);
    return affine_last_dim(ctx, normalized, weights, config.hidden_size);
}

core::TensorValue build_niagara_feed_forward(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NiagaraFeedForwardWeights & weights,
    const NiagaraEncoderConfig & config) {
    auto x = build_niagara_l1_norm(ctx, input, weights.norm, config);
    x = modules::LinearModule({config.hidden_size, config.intermediate_size, true}).build(ctx, x, weights.dense1);
    x = modules::SiluModule{}.build(ctx, x);
    return modules::LinearModule({config.intermediate_size, config.hidden_size, true}).build(ctx, x, weights.dense2);
}

core::TensorValue build_niagara_subsampling(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NiagaraSubsamplingWeights & weights,
    const NiagaraEncoderConfig & config) {
    core::validate_shape(input, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], 80}), "input");
    auto x = core::reshape_tensor(
        ctx,
        input,
        core::TensorShape::from_dims({input.shape.dims[0], 1, input.shape.dims[1], 80}));
    x = modules::Conv2dModule({1, 4, 1, 1, 1, 1, 0, 0, 1, 1, false}).build(ctx, x, weights.input_projection);
    x = modules::Conv2dModule({4, config.hidden_size, 4, 3, 2, 2, 0, 0, 1, 1, true}).build(ctx, x, weights.conv1);
    x = modules::BatchNorm2dEvalModule({config.hidden_size}).build(ctx, x, weights.bn0);
    x = modules::SiluModule{}.build(ctx, x);
    x = modules::Conv2dModule({config.hidden_size, config.hidden_size, 4, 3, 2, 2, 0, 0, 1, 1, true})
            .build(ctx, x, weights.conv2);
    x = modules::BatchNorm2dEvalModule({config.hidden_size}).build(ctx, x, weights.bn1);
    x = modules::SiluModule{}.build(ctx, x);
    x = contiguous(ctx, transpose(ctx, x, {0, 2, 3, 1}, 4));
    x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({
        input.shape.dims[0],
        x.shape.dims[1],
        x.shape.dims[2] * x.shape.dims[3],
    }));
    return modules::LinearModule({config.dense_input_features * config.hidden_size, config.hidden_size, true})
        .build(ctx, x, weights.dense);
}

core::TensorValue build_niagara_self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NiagaraSelfAttentionWeights & weights,
    const NiagaraEncoderConfig & config) {
    const int64_t frames = input.shape.dims[1];
    const int64_t head_dim = config.hidden_size / kNumAttentionHeads;
    auto x = build_niagara_l1_norm(ctx, input, weights.norm, config);
    auto q = modules::LinearModule({config.hidden_size, config.hidden_size, true}).build(ctx, x, weights.query);
    auto k = modules::LinearModule({config.hidden_size, config.hidden_size, true}).build(ctx, x, weights.key);
    auto v = modules::LinearModule({config.hidden_size, config.hidden_size, true}).build(ctx, x, weights.value);
    auto q_content = reshape_heads(ctx, add_last_dim_bias(ctx, q, weights.u_bias));
    auto q_position = reshape_heads(ctx, add_last_dim_bias(ctx, q, weights.v_bias));
    k = reshape_heads(ctx, k);
    v = reshape_heads(ctx, v);

    auto scores = modules::MatMulModule{}.build(ctx, q_content, transpose(ctx, k, {0, 1, 3, 2}, 4));

    const int64_t rel_start = kRelativePositionCenter - frames;
    auto relative = modules::SliceModule({0, rel_start, frames * 2 - 1}).build(ctx, weights.relative_position);
    relative = modules::LinearModule({config.hidden_size, config.hidden_size, true}).build(ctx, relative, weights.global_attention);
    relative = core::reshape_tensor(
        ctx,
        relative,
        core::TensorShape::from_dims({1, frames * 2 - 1, kNumAttentionHeads, head_dim}));
    relative = contiguous(ctx, transpose(ctx, relative, {0, 2, 3, 1}, 4));
    auto relative_scores = modules::MatMulModule{}.build(ctx, q_position, relative);
    relative_scores = relative_shift(ctx, relative_scores, frames);
    scores = modules::AddModule{}.build(ctx, scores, relative_scores);
    scores = core::wrap_tensor(
        ggml_scale(ctx.ggml, core::ensure_backend_addressable_layout(ctx, scores).tensor, 1.0f / std::sqrt(static_cast<float>(head_dim))),
        scores.shape,
        GGML_TYPE_F32);
    auto attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, scores.tensor), scores.shape, GGML_TYPE_F32);
    auto context = modules::MatMulModule{}.build(ctx, attn, v);
    context = contiguous(ctx, transpose(ctx, context, {0, 2, 1, 3}, 4));
    context = core::reshape_tensor(ctx, context, input.shape);
    return modules::LinearModule({config.hidden_size, config.hidden_size, true}).build(ctx, context, weights.out);
}

core::TensorValue build_niagara_state_space(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NiagaraStateSpaceWeights & weights,
    const NiagaraEncoderConfig & config) {
    constexpr int64_t kernel_size = 128;
    auto x = build_niagara_l1_norm(ctx, input, weights.norm, config);
    x = modules::LinearModule({config.hidden_size, config.state_channels, true}).build(ctx, x, weights.dense1);
    x = core::reshape_tensor(
        ctx,
        x,
        core::TensorShape::from_dims({x.shape.dims[0], x.shape.dims[1], 1, config.state_channels}));
    x = transpose(ctx, x, {0, 3, 1, 2}, 4);
    x = core::ensure_backend_addressable_layout(ctx, x);
    x = modules::Pad2dModule({0, 0, kernel_size - 1, 0}).build(ctx, x);
    x = modules::DepthwiseConv2dModule({config.state_channels, kernel_size, 1, 1, 1, 0, 0, 1, 1, false})
            .build(ctx, x, {weights.conv_kernel, std::nullopt});
    x = transpose(ctx, x, {0, 2, 3, 1}, 4);
    x = core::wrap_tensor(ggml_cont(ctx.ggml, x.tensor), x.shape, x.type);
    x = core::reshape_tensor(
        ctx,
        x,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.state_channels}));

    if (config.state_channels == config.hidden_size * 2) {
        const int channel_axis = static_cast<int>(x.shape.rank - 1);
        const auto gate = modules::SliceModule({channel_axis, config.hidden_size, config.hidden_size}).build(ctx, x);
        const auto value = modules::SliceModule({channel_axis, 0, config.hidden_size}).build(ctx, x);
        x = core::wrap_tensor(ggml_mul(ctx.ggml, value.tensor, modules::SigmoidModule{}.build(ctx, gate).tensor), value.shape, GGML_TYPE_F32);
    } else if (config.state_channels == config.hidden_size) {
        x = modules::SiluModule{}.build(ctx, x);
    } else {
        throw std::runtime_error("Niagara state-space channel count must be hidden or 2 * hidden");
    }

    return modules::LinearModule({config.hidden_size, config.hidden_size, true}).build(ctx, x, weights.dense2);
}

core::TensorValue build_niagara_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NiagaraLayerWeights & weights,
    const NiagaraEncoderConfig & config) {
    auto x = add_scaled_residual(ctx, input, build_niagara_feed_forward(ctx, input, weights.ffn, config), weights.ffn.residual_scale);
    auto attention = build_niagara_self_attention(ctx, x, weights.self_attention, config);
    x = modules::ResidualAddModule{}.build(ctx, x, attention);
    auto state = build_niagara_state_space(ctx, x, weights.state_space, config);
    x = modules::ResidualAddModule{}.build(ctx, x, state);
    x = add_scaled_residual(ctx, x, build_niagara_feed_forward(ctx, x, weights.ffn1, config), weights.ffn1.residual_scale);
    return build_niagara_l1_norm(ctx, x, weights.out_norm, config);
}

core::TensorValue build_niagara_encoder_logits(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NiagaraWeights & weights,
    const NiagaraAsrConfig & config) {
    auto x = build_niagara_subsampling(ctx, input, weights.subsampling, config.encoder);
    for (const auto & layer : weights.layers) {
        x = build_niagara_layer(ctx, x, layer, config.encoder);
    }
    return modules::LinearModule({config.encoder.hidden_size, config.vocab_size + 1, true}).build(ctx, x, weights.ctc);
}

}  // namespace engine::community_models::niagara_asr
