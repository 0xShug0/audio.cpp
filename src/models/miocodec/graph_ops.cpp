#include "graph_ops.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/core/constant_tensor_cache.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::miocodec::graphs {

namespace {

int64_t conv_transpose1d_output_frames(
    const engine::modules::ConvTranspose1dConfig & config,
    int64_t input_frames) {
    return (input_frames - 1) * config.stride - 2 * config.padding + config.dilation * (config.kernel_size - 1) + 1;
}

core::TensorValue conv_transpose1d_with_padding(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const MioCodecConvTranspose1dWeights & weights) {
    auto config = weights.config;
    config.padding = 0;
    auto output = engine::modules::ConvTranspose1dModule(config).build(ctx, input, weights.weights);
    if (weights.config.padding > 0) {
        output = engine::modules::SliceModule({2, weights.config.padding, conv_transpose1d_output_frames(weights.config, input.shape.dims[2])})
                     .build(ctx, output);
    }
    return output;
}

}  // namespace

core::TensorValue group_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const MioCodecNormWeights & weights,
    int64_t groups,
    float eps = 1.0e-6F) {
    auto x = input.type == GGML_TYPE_F32
        ? input
        : core::wrap_tensor(ggml_cast(ctx.ggml, input.tensor, GGML_TYPE_F32), input.shape, GGML_TYPE_F32);
    if (x.shape.rank != 3 || groups <= 0 || x.shape.dims[1] % groups != 0) {
        throw std::runtime_error("MioCodec GroupNorm requires [batch, channels, frames] with channels divisible by groups");
    }
    const int64_t channels = x.shape.dims[1];
    const int64_t channels_per_group = channels / groups;
    auto contiguous = core::ensure_backend_addressable_layout(ctx, x);
    const auto grouped_shape = core::TensorShape::from_dims({
        x.shape.dims[0],
        groups,
        channels_per_group * x.shape.dims[2],
    });
    auto grouped = core::reshape_tensor(ctx, contiguous, grouped_shape);
    auto mean = engine::modules::ReduceMeanModule({2}).build(ctx, grouped);
    auto mean_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, mean.tensor, grouped.tensor), grouped.shape, GGML_TYPE_F32);
    auto centered = core::wrap_tensor(ggml_sub(ctx.ggml, grouped.tensor, mean_rep.tensor), grouped.shape, GGML_TYPE_F32);
    auto squared = core::wrap_tensor(ggml_sqr(ctx.ggml, centered.tensor), grouped.shape, GGML_TYPE_F32);
    auto variance = engine::modules::ReduceMeanModule({2}).build(ctx, squared);
    auto stddev = core::wrap_tensor(
        ggml_sqrt(ctx.ggml, ggml_scale_bias(ctx.ggml, variance.tensor, 1.0F, eps)),
        variance.shape,
        GGML_TYPE_F32);
    auto stddev_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, stddev.tensor, centered.tensor), grouped.shape, GGML_TYPE_F32);
    auto normalized = core::wrap_tensor(ggml_div(ctx.ggml, centered.tensor, stddev_rep.tensor), grouped.shape, GGML_TYPE_F32);
    x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, normalized), x.shape);
    auto weight = core::reshape_tensor(ctx, *weights.weight, core::TensorShape::from_dims({1, input.shape.dims[1], 1}));
    auto bias = core::reshape_tensor(ctx, *weights.bias, core::TensorShape::from_dims({1, input.shape.dims[1], 1}));
    auto weight_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, weight.tensor, x.tensor), x.shape, GGML_TYPE_F32);
    auto bias_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, bias.tensor, x.tensor), x.shape, GGML_TYPE_F32);
    x = core::wrap_tensor(ggml_mul(ctx.ggml, x.tensor, weight_rep.tensor), x.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, bias_rep.tensor), x.shape, GGML_TYPE_F32);
}

core::TensorValue mask_bct(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & time_mask) {
    core::validate_shape(
        time_mask,
        core::TensorShape::from_dims({input.shape.dims[0], 1, input.shape.dims[2]}),
        "time mask");
    const auto mask_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, time_mask.tensor, input.tensor), input.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_mul(ctx.ggml, input.tensor, mask_rep.tensor), input.shape, GGML_TYPE_F32);
}

core::TensorValue masked_group_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const MioCodecNormWeights & weights,
    int64_t groups,
    const core::TensorValue & time_mask,
    const core::TensorValue & inv_valid_group_values,
    float eps = 1.0e-6F) {
    auto x = input.type == GGML_TYPE_F32
        ? input
        : core::wrap_tensor(ggml_cast(ctx.ggml, input.tensor, GGML_TYPE_F32), input.shape, GGML_TYPE_F32);
    if (x.shape.rank != 3 || groups <= 0 || x.shape.dims[1] % groups != 0) {
        throw std::runtime_error("MioCodec masked GroupNorm requires [batch, channels, frames] with channels divisible by groups");
    }
    core::validate_shape(
        time_mask,
        core::TensorShape::from_dims({x.shape.dims[0], 1, x.shape.dims[2]}),
        "masked GroupNorm time mask");
    core::validate_shape(
        inv_valid_group_values,
        core::TensorShape::from_dims({x.shape.dims[0], 1, 1}),
        "masked GroupNorm inverse count");
    const int64_t channels = x.shape.dims[1];
    const int64_t channels_per_group = channels / groups;
    auto mask_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, time_mask.tensor, x.tensor), x.shape, GGML_TYPE_F32);
    auto masked = core::wrap_tensor(ggml_mul(ctx.ggml, x.tensor, mask_rep.tensor), x.shape, GGML_TYPE_F32);
    auto grouped = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, masked),
        core::TensorShape::from_dims({x.shape.dims[0], groups, channels_per_group * x.shape.dims[2]}));
    auto grouped_mask = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, mask_rep),
        grouped.shape);
    auto sum = engine::modules::ReduceSumModule({2}).build(ctx, grouped);
    auto inv_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, inv_valid_group_values.tensor, sum.tensor), sum.shape, GGML_TYPE_F32);
    auto mean = core::wrap_tensor(ggml_mul(ctx.ggml, sum.tensor, inv_rep.tensor), sum.shape, GGML_TYPE_F32);
    auto mean_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, mean.tensor, grouped.tensor), grouped.shape, GGML_TYPE_F32);
    auto centered = core::wrap_tensor(ggml_sub(ctx.ggml, grouped.tensor, mean_rep.tensor), grouped.shape, GGML_TYPE_F32);
    auto centered_masked = core::wrap_tensor(ggml_mul(ctx.ggml, centered.tensor, grouped_mask.tensor), grouped.shape, GGML_TYPE_F32);
    auto squared = core::wrap_tensor(ggml_sqr(ctx.ggml, centered_masked.tensor), grouped.shape, GGML_TYPE_F32);
    auto variance_sum = engine::modules::ReduceSumModule({2}).build(ctx, squared);
    auto variance = core::wrap_tensor(ggml_mul(ctx.ggml, variance_sum.tensor, inv_rep.tensor), variance_sum.shape, GGML_TYPE_F32);
    auto stddev = core::wrap_tensor(
        ggml_sqrt(ctx.ggml, ggml_scale_bias(ctx.ggml, variance.tensor, 1.0F, eps)),
        variance.shape,
        GGML_TYPE_F32);
    auto stddev_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, stddev.tensor, grouped.tensor), grouped.shape, GGML_TYPE_F32);
    auto normalized = core::wrap_tensor(ggml_div(ctx.ggml, centered.tensor, stddev_rep.tensor), grouped.shape, GGML_TYPE_F32);
    normalized = core::wrap_tensor(ggml_mul(ctx.ggml, normalized.tensor, grouped_mask.tensor), grouped.shape, GGML_TYPE_F32);
    x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, normalized), x.shape);
    auto weight = core::reshape_tensor(ctx, *weights.weight, core::TensorShape::from_dims({1, input.shape.dims[1], 1}));
    auto bias = core::reshape_tensor(ctx, *weights.bias, core::TensorShape::from_dims({1, input.shape.dims[1], 1}));
    auto weight_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, weight.tensor, x.tensor), x.shape, GGML_TYPE_F32);
    auto bias_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, bias.tensor, x.tensor), x.shape, GGML_TYPE_F32);
    x = core::wrap_tensor(ggml_mul(ctx.ggml, x.tensor, weight_rep.tensor), x.shape, GGML_TYPE_F32);
    x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, bias_rep.tensor), x.shape, GGML_TYPE_F32);
    return mask_bct(ctx, x, time_mask);
}

core::TensorValue repeat_like(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    const core::TensorValue & like,
    const core::TensorShape & broadcast_shape) {
    const auto reshaped = core::reshape_tensor(ctx, value, broadcast_shape);
    return core::wrap_tensor(ggml_repeat(ctx.ggml, reshaped.tensor, like.tensor), like.shape, GGML_TYPE_F32);
}

core::TensorValue apply_gamma(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & gamma,
    int64_t hidden) {
    auto repeated = repeat_like(ctx, gamma, input, core::TensorShape::from_dims({1, hidden, 1}));
    return core::wrap_tensor(ggml_mul(ctx.ggml, input.tensor, repeated.tensor), input.shape, GGML_TYPE_F32);
}

core::TensorValue swiglu(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const MioCodecTransformerLayerWeights & weights,
    int64_t dim,
    int64_t intermediate_dim) {
    auto gate = engine::modules::LinearModule({dim, intermediate_dim, weights.feed_forward_w1.bias.has_value()}).build(
        ctx,
        input,
        weights.feed_forward_w1);
    gate = core::wrap_tensor(ggml_silu(ctx.ggml, gate.tensor), gate.shape, GGML_TYPE_F32);
    auto up = engine::modules::LinearModule({dim, intermediate_dim, weights.feed_forward_w3.bias.has_value()}).build(
        ctx,
        input,
        weights.feed_forward_w3);
    auto hidden = core::wrap_tensor(ggml_mul(ctx.ggml, gate.tensor, up.tensor), gate.shape, GGML_TYPE_F32);
    return engine::modules::LinearModule({intermediate_dim, dim, weights.feed_forward_w2.bias.has_value()}).build(
        ctx,
        hidden,
        weights.feed_forward_w2);
}

core::TensorValue apply_rope_fast(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    int64_t head_dim) {
    if (input.shape.rank != 4 || input.shape.dims[3] != head_dim || head_dim % 2 != 0) {
        throw std::runtime_error("MioCodec RoPE requires [batch, frames, heads, even_head_dim]");
    }
    core::validate_shape(positions, core::TensorShape::from_dims({input.shape.dims[1]}), "MioCodec RoPE positions");
    return core::wrap_tensor(
        ggml_rope_ext(
            ctx.ggml,
            input.tensor,
            positions.tensor,
            nullptr,
            static_cast<int>(head_dim),
            GGML_ROPE_TYPE_NORMAL,
            0,
            10000.0F,
            1.0F,
            0.0F,
            1.0F,
            0.0F,
            0.0F),
        input.shape,
        input.type);
}

core::TensorValue apply_rope_exact(
    core::ModuleBuildContext & ctx,
    core::ConstantTensorCache & constants,
    const core::TensorValue & input,
    int64_t head_dim) {
    if (input.shape.rank != 4 || input.shape.dims[3] != head_dim || head_dim % 2 != 0) {
        throw std::runtime_error("MioCodec RoPE requires [batch, frames, heads, even_head_dim]");
    }
    const int64_t frames = input.shape.dims[1];
    const auto trig_shape = core::TensorShape::from_dims({1, frames, 1, 1});
    core::TensorValue output;
    for (int64_t d = 0; d < head_dim; d += 2) {
        const double theta = std::pow(10000.0, -static_cast<double>(d) / static_cast<double>(head_dim));
        std::vector<float> cos_values(static_cast<size_t>(frames), 0.0F);
        std::vector<float> sin_values(static_cast<size_t>(frames), 0.0F);
        for (int64_t t = 0; t < frames; ++t) {
            const double angle = static_cast<double>(t) * theta;
            cos_values[static_cast<size_t>(t)] = static_cast<float>(std::cos(angle));
            sin_values[static_cast<size_t>(t)] = static_cast<float>(std::sin(angle));
        }
        auto x0 = engine::modules::SliceModule({3, d, 1}).build(ctx, input);
        auto x1 = engine::modules::SliceModule({3, d + 1, 1}).build(ctx, input);
        auto cos_t = repeat_like(ctx, constants.make_f32(trig_shape, cos_values), x0, trig_shape);
        auto sin_t = repeat_like(ctx, constants.make_f32(trig_shape, sin_values), x0, trig_shape);
        auto y0 = core::wrap_tensor(
            ggml_sub(
                ctx.ggml,
                ggml_mul(ctx.ggml, x0.tensor, cos_t.tensor),
                ggml_mul(ctx.ggml, x1.tensor, sin_t.tensor)),
            x0.shape,
            GGML_TYPE_F32);
        auto y1 = core::wrap_tensor(
            ggml_add(
                ctx.ggml,
                ggml_mul(ctx.ggml, x0.tensor, sin_t.tensor),
                ggml_mul(ctx.ggml, x1.tensor, cos_t.tensor)),
            x1.shape,
            GGML_TYPE_F32);
        auto pair = engine::modules::ConcatModule({3}).build(ctx, y0, y1);
        output = output.valid()
            ? engine::modules::ConcatModule({3}).build(ctx, output, pair)
            : pair;
    }
    return output;
}

std::vector<float> local_attention_mask(int64_t heads, int64_t frames, int64_t window_size, int64_t valid_frames) {
    const int64_t side = window_size / 2;
    const float neg_inf = -std::numeric_limits<float>::infinity();
    std::vector<float> values(static_cast<size_t>(heads * frames * frames), neg_inf);
    for (int64_t head = 0; head < heads; ++head) {
        for (int64_t query = 0; query < frames; ++query) {
            auto * row = values.data() + static_cast<size_t>((head * frames + query) * frames);
            if (query >= valid_frames) {
                row[0] = 0.0F;
            } else {
                const int64_t begin = std::max<int64_t>(0, query - side);
                const int64_t end = std::min<int64_t>(valid_frames, query + side + 1);
                std::fill(row + begin, row + end, 0.0F);
            }
        }
    }
    return values;
}

core::TensorValue make_window_mask(
    core::ConstantTensorCache & constants,
    int64_t heads,
    int64_t frames,
    int64_t window_size) {
    return constants.make_f32(
        core::TensorShape::from_dims({1, heads, frames, frames}),
        local_attention_mask(heads, frames, window_size, frames));
}

core::TensorValue make_i32_constant(
    core::ConstantTensorCache & constants,
    const core::TensorShape & shape,
    const std::vector<int32_t> & values) {
    return constants.make_tensor(
        shape,
        GGML_TYPE_I32,
        values.data(),
        values.size() * sizeof(int32_t));
}

core::TensorValue make_ones_i32(
    core::ConstantTensorCache & constants,
    const core::TensorShape & shape) {
    return make_i32_constant(
        constants,
        shape,
        std::vector<int32_t>(static_cast<size_t>(shape.num_elements()), 1));
}

core::TensorValue make_ones_f32(
    core::ConstantTensorCache & constants,
    const core::TensorShape & shape) {
    return constants.make_f32(
        shape,
        std::vector<float>(static_cast<size_t>(shape.num_elements()), 1.0F));
}

TimeMaskConstants make_full_time_mask(
    core::ConstantTensorCache & constants,
    int64_t frames,
    int64_t channels_per_group) {
    return {
        make_ones_f32(constants, core::TensorShape::from_dims({1, 1, frames})),
        constants.make_f32(
            core::TensorShape::from_dims({1, 1, 1}),
            std::vector<float>{1.0F / static_cast<float>(frames * channels_per_group)}),
    };
}

InterpolationConstants make_interpolation_constants(
    core::ConstantTensorCache & constants,
    int64_t content_frames,
    int64_t stft_frames,
    const MioCodecConvTranspose1dWeights & upsample) {
    const int64_t source_frames = conv_transpose1d_output_frames(upsample.config, content_frames);
    std::vector<int32_t> left(static_cast<size_t>(stft_frames), 0);
    std::vector<int32_t> right(static_cast<size_t>(stft_frames), 0);
    std::vector<float> weight(static_cast<size_t>(stft_frames), 0.0F);
    const double scale = static_cast<double>(source_frames) / static_cast<double>(stft_frames);
    for (int64_t t = 0; t < stft_frames; ++t) {
        double source = (static_cast<double>(t) + 0.5) * scale - 0.5;
        source = std::max(0.0, std::min(source, static_cast<double>(source_frames - 1)));
        const int64_t l = static_cast<int64_t>(std::floor(source));
        const int64_t r = std::min<int64_t>(l + 1, source_frames - 1);
        left[static_cast<size_t>(t)] = static_cast<int32_t>(l);
        right[static_cast<size_t>(t)] = static_cast<int32_t>(r);
        weight[static_cast<size_t>(t)] = static_cast<float>(source - static_cast<double>(l));
    }
    return {
        make_i32_constant(constants, core::TensorShape::from_dims({stft_frames}), left),
        make_i32_constant(constants, core::TensorShape::from_dims({stft_frames}), right),
        constants.make_f32(core::TensorShape::from_dims({1, stft_frames, 1}), weight),
    };
}

core::TensorValue attention(
    core::ModuleBuildContext & ctx,
    core::ConstantTensorCache & constants,
    const core::TensorValue & mask,
    const core::TensorValue & positions,
    const core::TensorValue & input,
    const MioCodecTransformerLayerWeights & weights,
    const MioCodecTransformerWeights & config,
    bool exact_rope) {
    const int64_t head_dim = config.head_dim;
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    auto qkv = engine::modules::LinearModule({config.dim, 3 * config.dim, weights.qkv_proj.bias.has_value()}).build(
        ctx,
        input,
        weights.qkv_proj);
    const auto qkv_axis = static_cast<int>(qkv.shape.rank - 1);
    auto q = engine::modules::SliceModule({qkv_axis, 0, config.dim}).build(ctx, qkv);
    auto k = engine::modules::SliceModule({qkv_axis, config.dim, config.dim}).build(ctx, qkv);
    auto v = engine::modules::SliceModule({qkv_axis, 2 * config.dim, config.dim}).build(ctx, qkv);
    q = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, q), core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.heads, head_dim}));
    k = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, k), core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.heads, head_dim}));
    v = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, v), core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.heads, head_dim}));
    if (exact_rope) {
        q = apply_rope_exact(ctx, constants, q, head_dim);
        k = apply_rope_exact(ctx, constants, k, head_dim);
    } else {
        q = apply_rope_fast(ctx, q, positions, head_dim);
        k = apply_rope_fast(ctx, k, positions, head_dim);
    }
    auto qh = engine::modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto kh = engine::modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto vh = engine::modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto scores = engine::modules::MatMulModule{}.build(
        ctx,
        qh,
        engine::modules::TransposeModule({{0, 1, 3, 2}, kh.shape.rank}).build(ctx, kh));
    auto attn = core::wrap_tensor(
        ggml_soft_max_ext(ctx.ggml, core::ensure_backend_addressable_layout(ctx, scores).tensor, mask.tensor, scale, 0.0F),
        scores.shape,
        GGML_TYPE_F32);
    auto context = engine::modules::MatMulModule{}.build(ctx, attn, vh);
    context = core::ensure_backend_addressable_layout(
        ctx,
        engine::modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context));
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.dim}));
    return engine::modules::LinearModule({config.dim, config.dim, weights.out_proj.bias.has_value()}).build(
        ctx,
        context,
        weights.out_proj);
}

std::pair<core::TensorValue, std::optional<core::TensorValue>> adaln(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & condition,
    const MioCodecAdaLayerNormWeights & weights,
    int64_t dim,
    bool return_gate) {
    auto normalized = engine::modules::LayerNormModule({dim, 1.0e-5F, false, false}).build(ctx, input, {});
    const int64_t out_dim = return_gate ? 3 * dim : 2 * dim;
    auto params = engine::modules::LinearModule(
        {condition.shape.last_dim(), out_dim, weights.condition_projection.bias.has_value()}).build(
        ctx,
        engine::modules::SiluModule().build(ctx, condition),
        weights.condition_projection);
    auto shift = engine::modules::SliceModule({static_cast<int>(params.shape.rank - 1), 0, dim}).build(ctx, params);
    auto scale = engine::modules::SliceModule({static_cast<int>(params.shape.rank - 1), dim, dim}).build(ctx, params);
    auto one_plus_scale = core::wrap_tensor(ggml_scale_bias(ctx.ggml, scale.tensor, 1.0F, 1.0F), scale.shape, GGML_TYPE_F32);
    auto scale_full = core::wrap_tensor(ggml_repeat(ctx.ggml, one_plus_scale.tensor, normalized.tensor), normalized.shape, GGML_TYPE_F32);
    auto shift_full = core::wrap_tensor(ggml_repeat(ctx.ggml, shift.tensor, normalized.tensor), normalized.shape, GGML_TYPE_F32);
    auto out = core::wrap_tensor(ggml_add(ctx.ggml, ggml_mul(ctx.ggml, normalized.tensor, scale_full.tensor), shift_full.tensor), normalized.shape, GGML_TYPE_F32);
    if (!return_gate) {
        return {out, std::nullopt};
    }
    auto gate = engine::modules::SliceModule({static_cast<int>(params.shape.rank - 1), 2 * dim, dim}).build(ctx, params);
    return {out, core::wrap_tensor(ggml_repeat(ctx.ggml, gate.tensor, out.tensor), out.shape, GGML_TYPE_F32)};
}

core::TensorValue transformer(
    core::ModuleBuildContext & ctx,
    core::ConstantTensorCache & constants,
    const core::TensorValue & input,
    const MioCodecTransformerWeights & weights,
    const std::optional<core::TensorValue> & condition,
    const std::optional<core::TensorValue> & attention_mask,
    bool exact_rope) {
    auto x = input;
    if (weights.output_projection.has_value() && input.shape.last_dim() != weights.dim) {
        throw std::runtime_error("MioCodec transformer input projection is not configured for this component");
    }
    const auto mask = attention_mask.has_value()
        ? *attention_mask
        : make_window_mask(constants, weights.heads, input.shape.dims[1], weights.window_size);
    std::vector<int32_t> position_values(static_cast<size_t>(input.shape.dims[1]), 0);
    for (int64_t i = 0; i < input.shape.dims[1]; ++i) {
        position_values[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    const auto positions = constants.make_tensor(
        core::TensorShape::from_dims({input.shape.dims[1]}),
        GGML_TYPE_I32,
        position_values.data(),
        position_values.size() * sizeof(int32_t));
    for (const auto & layer : weights.layers) {
        core::TensorValue normed;
        std::optional<core::TensorValue> attn_gate;
        if (weights.use_adaln) {
            if (!condition.has_value() || !layer.attention_adaln.has_value()) {
                throw std::runtime_error("MioCodec AdaLN transformer requires condition");
            }
            auto out = adaln(ctx, x, *condition, *layer.attention_adaln, weights.dim, true);
            normed = out.first;
            attn_gate = out.second;
        } else {
            normed = engine::modules::LayerNormModule({weights.dim, 1.0e-5F, true, true}).build(
                ctx,
                x,
                *layer.attention_norm);
        }
        auto attn_out = attention(ctx, constants, mask, positions, normed, layer, weights, exact_rope);
        if (attn_gate.has_value()) {
            attn_out = core::wrap_tensor(ggml_mul(ctx.ggml, attn_out.tensor, attn_gate->tensor), attn_out.shape, GGML_TYPE_F32);
        }
        x = engine::modules::ResidualAddModule{}.build(ctx, x, attn_out);

        core::TensorValue ffn_normed;
        std::optional<core::TensorValue> ffn_gate;
        if (weights.use_adaln) {
            auto out = adaln(ctx, x, *condition, *layer.feed_forward_adaln, weights.dim, true);
            ffn_normed = out.first;
            ffn_gate = out.second;
        } else {
            ffn_normed = engine::modules::LayerNormModule({weights.dim, 1.0e-5F, true, true}).build(
                ctx,
                x,
                *layer.feed_forward_norm);
        }
        auto ffn_out = swiglu(ctx, ffn_normed, layer, weights.dim, weights.intermediate_dim);
        if (ffn_gate.has_value()) {
            ffn_out = core::wrap_tensor(ggml_mul(ctx.ggml, ffn_out.tensor, ffn_gate->tensor), ffn_out.shape, GGML_TYPE_F32);
        }
        x = engine::modules::ResidualAddModule{}.build(ctx, x, ffn_out);
    }
    if (weights.use_adaln) {
        x = adaln(ctx, x, *condition, *weights.adaln_norm, weights.dim, false).first;
    } else {
        x = engine::modules::LayerNormModule({weights.dim, 1.0e-5F, true, true}).build(
            ctx,
            x,
            *weights.norm);
    }
    if (weights.output_projection.has_value()) {
        x = engine::modules::LinearModule(
            {weights.dim, weights.output_projection->weight.shape.dims[0], weights.output_projection->bias.has_value()}).build(
            ctx,
            x,
            *weights.output_projection);
    }
    return x;
}

core::TensorValue fsq_quantized(
    core::ModuleBuildContext & ctx,
    core::ConstantTensorCache & constants,
    const core::TensorValue & input,
    const MioCodecQuantizerWeights & weights) {
    auto latent = engine::modules::LinearModule({768, 5, weights.input_projection.bias.has_value()}).build(
        ctx,
        input,
        weights.input_projection);
    constexpr float eps = 1.0e-3F;
    const float half_l_values[] = {3.5F * (1.0F - eps), 3.5F * (1.0F - eps), 3.5F * (1.0F - eps), 2.0F * (1.0F - eps), 2.0F * (1.0F - eps)};
    const float offset_values[] = {0.5F, 0.5F, 0.5F, 0.0F, 0.0F};
    const float shift_values[] = {
        std::tan(offset_values[0] / half_l_values[0]),
        std::tan(offset_values[1] / half_l_values[1]),
        std::tan(offset_values[2] / half_l_values[2]),
        0.0F,
        0.0F};
    const float half_width_values[] = {4.0F, 4.0F, 4.0F, 2.0F, 2.0F};
    const auto constant_shape = core::TensorShape::from_dims({1, 1, 5});
    const auto half_l = repeat_like(ctx, constants.make_f32(constant_shape, std::vector<float>(std::begin(half_l_values), std::end(half_l_values))), latent, constant_shape);
    const auto offset = repeat_like(ctx, constants.make_f32(constant_shape, std::vector<float>(std::begin(offset_values), std::end(offset_values))), latent, constant_shape);
    const auto shift = repeat_like(ctx, constants.make_f32(constant_shape, std::vector<float>(std::begin(shift_values), std::end(shift_values))), latent, constant_shape);
    const auto half_width = repeat_like(ctx, constants.make_f32(constant_shape, std::vector<float>(std::begin(half_width_values), std::end(half_width_values))), latent, constant_shape);
    auto shifted = core::wrap_tensor(ggml_add(ctx.ggml, latent.tensor, shift.tensor), latent.shape, GGML_TYPE_F32);
    auto bounded = core::wrap_tensor(ggml_tanh(ctx.ggml, shifted.tensor), latent.shape, GGML_TYPE_F32);
    bounded = core::wrap_tensor(ggml_sub(ctx.ggml, ggml_mul(ctx.ggml, bounded.tensor, half_l.tensor), offset.tensor), latent.shape, GGML_TYPE_F32);
    auto rounded = core::wrap_tensor(ggml_round(ctx.ggml, bounded.tensor), latent.shape, GGML_TYPE_F32);
    auto normalized = core::wrap_tensor(ggml_div(ctx.ggml, rounded.tensor, half_width.tensor), latent.shape, GGML_TYPE_F32);
    return engine::modules::LinearModule({5, 768, weights.output_projection.bias.has_value()}).build(
        ctx,
        normalized,
        weights.output_projection);
}

core::TensorValue convnext_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const MioCodecConvNeXtBlockWeights & weights) {
    auto x = engine::modules::DepthwiseConv1dModule(weights.depthwise_conv_config).build(ctx, input_bct, weights.depthwise_conv);
    x = engine::modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
    x = engine::modules::LayerNormModule({384, 1.0e-6F, true, true}).build(ctx, x, weights.norm);
    x = engine::modules::LinearModule({384, 1152, weights.pointwise_conv1.bias.has_value()}).build(
        ctx,
        x,
        weights.pointwise_conv1);
    x = engine::modules::GeluModule({engine::modules::GeluApproximation::ExactErf}).build(ctx, x);
    x = engine::modules::LinearModule({1152, 384, weights.pointwise_conv2.bias.has_value()}).build(
        ctx,
        x,
        weights.pointwise_conv2);
    x = engine::modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
    x = apply_gamma(ctx, x, weights.gamma, 384);
    return engine::modules::ResidualAddModule{}.build(ctx, input_bct, x);
}

core::TensorValue global_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_btc,
    const MioCodecGlobalEncoderWeights & weights) {
    auto x = engine::modules::TransposeModule({{0, 2, 1, 3}, input_btc.shape.rank}).build(ctx, input_btc);
    x = engine::modules::Conv1dModule(weights.embedding.config).build(ctx, x, weights.embedding.weights);
    x = engine::modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
    x = engine::modules::LayerNormModule({384, 1.0e-6F, true, true}).build(ctx, x, weights.embedding_norm);
    x = engine::modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
    for (const auto & block : weights.blocks) {
        x = convnext_block(ctx, x, block);
    }
    auto features_btc = engine::modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
    features_btc = engine::modules::LayerNormModule({384, 1.0e-6F, true, true}).build(ctx, features_btc, weights.final_norm);
    auto features_bct = engine::modules::TransposeModule({{0, 2, 1, 3}, features_btc.shape.rank}).build(ctx, features_btc);
    auto alpha = engine::modules::Conv1dModule(weights.attention_conv1.config).build(ctx, features_bct, weights.attention_conv1.weights);
    alpha = core::wrap_tensor(ggml_tanh(ctx.ggml, alpha.tensor), alpha.shape, GGML_TYPE_F32);
    alpha = engine::modules::Conv1dModule(weights.attention_conv2.config).build(ctx, alpha, weights.attention_conv2.weights);
    alpha = core::wrap_tensor(ggml_soft_max(ctx.ggml, core::ensure_backend_addressable_layout(ctx, alpha).tensor), alpha.shape, GGML_TYPE_F32);
    auto weighted = core::wrap_tensor(ggml_mul(ctx.ggml, alpha.tensor, features_bct.tensor), features_bct.shape, GGML_TYPE_F32);
    auto mean = engine::modules::ReduceSumModule({2}).build(ctx, weighted);
    auto features_contiguous = core::ensure_backend_addressable_layout(ctx, features_bct);
    auto squared = core::wrap_tensor(ggml_sqr(ctx.ggml, features_contiguous.tensor), features_contiguous.shape, GGML_TYPE_F32);
    auto second = engine::modules::ReduceSumModule({2}).build(ctx, core::wrap_tensor(ggml_mul(ctx.ggml, alpha.tensor, squared.tensor), squared.shape, GGML_TYPE_F32));
    auto mean_sq = core::wrap_tensor(ggml_sqr(ctx.ggml, mean.tensor), mean.shape, GGML_TYPE_F32);
    auto var = core::wrap_tensor(ggml_sub(ctx.ggml, second.tensor, mean_sq.tensor), mean.shape, GGML_TYPE_F32);
    var = core::wrap_tensor(ggml_clamp(ctx.ggml, var.tensor, 1.0e-4F, 1.0e4F), var.shape, GGML_TYPE_F32);
    auto std = core::wrap_tensor(ggml_sqrt(ctx.ggml, var.tensor), var.shape, GGML_TYPE_F32);
    auto stats = engine::modules::ConcatModule({1}).build(ctx, mean, std);
    stats = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, stats), core::TensorShape::from_dims({stats.shape.dims[0], 768}));
    auto out = engine::modules::LinearModule({768, 128, weights.pooling_projection.bias.has_value()}).build(
        ctx,
        stats,
        weights.pooling_projection);
    return engine::modules::LayerNormModule({128, 1.0e-5F, true, true}).build(ctx, out, weights.pooling_norm);
}

core::TensorValue dynamic_linear_interpolate_bct(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const core::TensorValue & left_indices,
    const core::TensorValue & right_indices,
    const core::TensorValue & right_weight) {
    const int64_t source_frames = input_bct.shape.dims[2];
    const int64_t channels = input_bct.shape.dims[1];
    const int64_t target_frames = left_indices.shape.dims[0];
    core::validate_shape(right_indices, left_indices.shape, "interpolate right indices");
    core::validate_shape(
        right_weight,
        core::TensorShape::from_dims({1, target_frames, 1}),
        "interpolate right weight");
    auto input_btc = engine::modules::TransposeModule({{0, 2, 1, 3}, input_bct.shape.rank}).build(ctx, input_bct);
    auto table = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, input_btc),
        core::TensorShape::from_dims({source_frames, channels}));
    auto left = engine::modules::EmbeddingModule({source_frames, channels}).build(ctx, left_indices, table);
    auto right = engine::modules::EmbeddingModule({source_frames, channels}).build(ctx, right_indices, table);
    left = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, left), core::TensorShape::from_dims({1, target_frames, channels}));
    right = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, right), core::TensorShape::from_dims({1, target_frames, channels}));
    auto weight = core::wrap_tensor(ggml_repeat(ctx.ggml, right_weight.tensor, left.tensor), left.shape, GGML_TYPE_F32);
    auto left_weight = core::wrap_tensor(ggml_scale_bias(ctx.ggml, weight.tensor, -1.0F, 1.0F), weight.shape, GGML_TYPE_F32);
    auto left_scaled = core::wrap_tensor(ggml_mul(ctx.ggml, left.tensor, left_weight.tensor), left.shape, GGML_TYPE_F32);
    auto right_scaled = core::wrap_tensor(ggml_mul(ctx.ggml, right.tensor, weight.tensor), right.shape, GGML_TYPE_F32);
    auto out_btc = core::wrap_tensor(ggml_add(ctx.ggml, left_scaled.tensor, right_scaled.tensor), left.shape, GGML_TYPE_F32);
    return engine::modules::TransposeModule({{0, 2, 1, 3}, out_btc.shape.rank}).build(ctx, out_btc);
}

core::TensorValue resnet_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const MioCodecResNetBlockWeights & weights,
    int64_t groups,
    const std::optional<core::TensorValue> & time_mask = std::nullopt,
    const std::optional<core::TensorValue> & inv_valid_group_values = std::nullopt) {
    auto x = time_mask.has_value()
        ? masked_group_norm(ctx, input_bct, weights.norm1, groups, *time_mask, *inv_valid_group_values)
        : group_norm(ctx, input_bct, weights.norm1, groups);
    x = engine::modules::SiluModule().build(ctx, x);
    x = engine::modules::Conv1dModule(weights.conv1.config).build(ctx, x, weights.conv1.weights);
    x = time_mask.has_value() ? mask_bct(ctx, x, *time_mask) : x;
    x = time_mask.has_value()
        ? masked_group_norm(ctx, x, weights.norm2, groups, *time_mask, *inv_valid_group_values)
        : group_norm(ctx, x, weights.norm2, groups);
    x = engine::modules::SiluModule().build(ctx, x);
    x = engine::modules::Conv1dModule(weights.conv2.config).build(ctx, x, weights.conv2.weights);
    x = engine::modules::ResidualAddModule{}.build(ctx, input_bct, x);
    return time_mask.has_value() ? mask_bct(ctx, x, *time_mask) : x;
}

core::TensorValue resnet_stack(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const MioCodecResNetStackWeights & weights,
    int64_t groups,
    const std::optional<core::TensorValue> & time_mask,
    const std::optional<core::TensorValue> & inv_valid_group_values) {
    auto x = input_bct;
    for (const auto & block : weights.blocks) {
        x = resnet_block(ctx, x, block, std::min<int64_t>(groups, x.shape.dims[1]), time_mask, inv_valid_group_values);
    }
    return x;
}

core::TensorValue snake_beta(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const MioCodecSnakeBetaWeights & weights) {
    auto input = core::ensure_backend_addressable_layout(ctx, input_bct);
    auto alpha = core::reshape_tensor(ctx, weights.alpha, core::TensorShape::from_dims({1, input_bct.shape.dims[1], 1}));
    auto inv_beta = core::reshape_tensor(ctx, weights.inv_beta, core::TensorShape::from_dims({1, input_bct.shape.dims[1], 1}));
    auto alpha_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, alpha.tensor, input.tensor), input.shape, GGML_TYPE_F32);
    auto inv_beta_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, inv_beta.tensor, input.tensor), input.shape, GGML_TYPE_F32);
    auto ax = core::wrap_tensor(ggml_mul(ctx.ggml, input.tensor, alpha_rep.tensor), input.shape, GGML_TYPE_F32);
    ax = core::ensure_backend_addressable_layout(ctx, ax);
    auto periodic = core::wrap_tensor(ggml_sqr(ctx.ggml, ggml_sin(ctx.ggml, ax.tensor)), input.shape, GGML_TYPE_F32);
    periodic = core::wrap_tensor(ggml_mul(ctx.ggml, periodic.tensor, inv_beta_rep.tensor), input.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_add(ctx.ggml, input.tensor, periodic.tensor), input.shape, GGML_TYPE_F32);
}

core::TensorValue upsampler(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const MioCodecUpsamplerWeights & weights,
    int64_t groups,
    const std::vector<core::TensorValue> * stage_time_masks,
    const std::vector<core::TensorValue> * stage_inv_valid_group_values) {
    auto x = input_bct;
    for (size_t index = 0; index < weights.stages.size(); ++index) {
        const auto & stage = weights.stages[index];
        x = conv_transpose1d_with_padding(ctx, x, stage.upsample);
        if (stage_time_masks != nullptr) {
            x = mask_bct(ctx, x, (*stage_time_masks)[index]);
        }
        x = snake_beta(ctx, x, stage.snake);
        if (stage_time_masks != nullptr) {
            x = mask_bct(ctx, x, (*stage_time_masks)[index]);
        }
        x = resnet_block(
            ctx,
            x,
            stage.resnet,
            std::min<int64_t>(groups, x.shape.dims[1]),
            stage_time_masks != nullptr ? std::optional<core::TensorValue>((*stage_time_masks)[index]) : std::nullopt,
            stage_inv_valid_group_values != nullptr ? std::optional<core::TensorValue>((*stage_inv_valid_group_values)[index]) : std::nullopt);
    }
    x = engine::modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
    x = engine::modules::LinearModule(
        {weights.output_projection.weight.shape.dims[1],
         weights.output_projection.weight.shape.dims[0],
         weights.output_projection.bias.has_value()}).build(
        ctx,
        x,
        weights.output_projection);
    x = snake_beta(
        ctx,
        engine::modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x),
        weights.output_snake);
    if (stage_time_masks != nullptr && !stage_time_masks->empty()) {
        x = mask_bct(ctx, x, stage_time_masks->back());
    }
    return engine::modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
}

}  // namespace engine::models::miocodec::graphs
