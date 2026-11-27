#include "engine/models/gigaam_asr/zapformer.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <stdexcept>

namespace engine::models::gigaam_asr {
namespace {

using core::TensorShape;
using core::TensorValue;

TensorValue wrap(ggml_tensor * tensor, std::initializer_list<int64_t> dims) {
    return core::wrap_tensor(tensor, TensorShape::from_dims(dims), GGML_TYPE_F32);
}

ggml_tensor * contiguous(core::ModuleBuildContext & ctx, ggml_tensor * tensor) {
    return ggml_is_contiguous(tensor) ? tensor : ggml_cont(ctx.ggml, tensor);
}

// Columns [start, start + width) of a [1, frames, features] tensor, made contiguous.
TensorValue columns(core::ModuleBuildContext & ctx, const TensorValue & x, int64_t start, int64_t width) {
    auto * t = contiguous(ctx, x.tensor);
    const int64_t frames = x.shape.dims[1];
    auto * view = ggml_view_2d(ctx.ggml, t, width, frames, t->nb[1], static_cast<size_t>(start) * t->nb[0]);
    return wrap(ggml_cont(ctx.ggml, view), {1, frames, width});
}

TensorValue linear(core::ModuleBuildContext & ctx, const TensorValue & x, const modules::LinearWeights & w,
    int64_t in, int64_t out) {
    return modules::LinearModule({in, out, w.bias.has_value()}).build(ctx, x, w);
}

TensorValue scale(core::ModuleBuildContext & ctx, const TensorValue & x, float s) {
    return core::wrap_tensor(ggml_scale(ctx.ggml, contiguous(ctx, x.tensor), s), x.shape, GGML_TYPE_F32);
}

TensorValue add(core::ModuleBuildContext & ctx, const TensorValue & a, const TensorValue & b) {
    return core::wrap_tensor(ggml_add(ctx.ggml, a.tensor, b.tensor), a.shape, GGML_TYPE_F32);
}

TensorValue mul(core::ModuleBuildContext & ctx, const TensorValue & a, const TensorValue & b) {
    return core::wrap_tensor(ggml_mul(ctx.ggml, a.tensor, b.tensor), a.shape, GGML_TYPE_F32);
}

TensorValue sigmoid(core::ModuleBuildContext & ctx, const TensorValue & x) {
    return core::wrap_tensor(ggml_sigmoid(ctx.ggml, contiguous(ctx, x.tensor)), x.shape, GGML_TYPE_F32);
}

TensorValue silu(core::ModuleBuildContext & ctx, const TensorValue & x) {
    return core::wrap_tensor(ggml_silu(ctx.ggml, contiguous(ctx, x.tensor)), x.shape, GGML_TYPE_F32);
}

// RMSNormTorch: x * rsqrt(mean(x^2) + eps) * weight.
TensorValue rms_norm(core::ModuleBuildContext & ctx, const TensorValue & x, const modules::NormWeights & w, float eps) {
    auto * y = ggml_rms_norm(ctx.ggml, contiguous(ctx, x.tensor), eps);
    if (w.weight) {
        y = ggml_mul(ctx.ggml, y, w.weight->tensor);
    }
    return core::wrap_tensor(y, x.shape, GGML_TYPE_F32);
}

TensorValue feed_forward(core::ModuleBuildContext & ctx, const TensorValue & x, const modules::LinearWeights & in,
    const modules::LinearWeights & out, int64_t dim, int64_t hidden) {
    auto y = linear(ctx, x, in, dim, hidden);
    y = silu(ctx, y);
    return linear(ctx, y, out, hidden, dim);
}

// First `count` frames of a [1, frames, heads, dim] tensor (a view; heads and dim stay contiguous).
TensorValue first_frames_4d(core::ModuleBuildContext & ctx, const TensorValue & x, int64_t count) {
    auto * t = contiguous(ctx, x.tensor);
    auto * view = ggml_view_3d(ctx.ggml, t, t->ne[0], t->ne[1], count, t->nb[1], t->nb[2], 0);
    return wrap(view, {1, count, x.shape.dims[2], x.shape.dims[3]});
}

struct LayerContext {
    int64_t frames = 0;        // frames of this stack (padded)
    int64_t valid_frames = 0;  // keys/values and averaged frames
    TensorValue positions;     // I32 [frames]
    const TensorValue * valid = nullptr;  // [1, frames, 1] when valid_frames < frames
};

TensorValue build_layer(core::ModuleBuildContext & ctx, const TensorValue & x0, const LayerContext & lc,
    const ZapformerStackConfig & c, const ZapformerLayerWeights & w, float rope_theta) {
    using namespace modules;
    const int64_t d = c.dim, heads = c.heads, qd = c.query_head_dim, vd = c.value_head_dim;
    const int64_t hidden = d * c.feedforward_multiple;
    const int64_t n = lc.frames;

    auto x = add(ctx, x0, feed_forward(ctx, x0, w.ff1_in, w.ff1_out, d, hidden));

    // Attention: queries and keys come from the layer input, values and gates from the feed-forward output.
    auto qk = linear(ctx, rms_norm(ctx, x0, w.attention_norm, 1e-6f), w.qk, d, 2 * heads * qd);
    auto q = core::reshape_tensor(ctx, columns(ctx, qk, 0, heads * qd), TensorShape::from_dims({1, n, heads, qd}));
    auto k = core::reshape_tensor(ctx, columns(ctx, qk, heads * qd, heads * qd), TensorShape::from_dims({1, n, heads, qd}));
    const RoPEModule rope({qd, GGML_ROPE_TYPE_NORMAL, rope_theta});
    q = rope.build(ctx, q, lc.positions);
    k = rope.build(ctx, k, lc.positions);
    auto vg = linear(ctx, x, w.vg, d, 3 * heads * vd);
    auto v = mul(ctx, columns(ctx, vg, 0, heads * vd), sigmoid(ctx, columns(ctx, vg, heads * vd, heads * vd)));
    const auto gate_out = sigmoid(ctx, columns(ctx, vg, 2 * heads * vd, heads * vd));
    v = core::reshape_tensor(ctx, v, TensorShape::from_dims({1, n, heads, vd}));
    if (lc.valid_frames < n) {
        k = first_frames_4d(ctx, k, lc.valid_frames);
        v = first_frames_4d(ctx, v, lc.valid_frames);
    }
    q = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    k = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
    v = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
    auto a = ScaledDotProductAttentionModule({qd, ScaledDotProductAttentionLowering::Flash}).build(ctx, q, k, v);
    a = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, a), TensorShape::from_dims({1, n, heads * vd}));
    a = linear(ctx, mul(ctx, a, gate_out), w.attention_out, heads * vd, d);
    x = add(ctx, x, a);

    if (w.has_conv) {
        auto ug = linear(ctx, x, w.conv_in, d, 2 * d);
        auto u = columns(ctx, ug, 0, d);
        const auto gate = sigmoid(ctx, columns(ctx, ug, d, d));
        if (lc.valid != nullptr) {
            u = mul(ctx, u, *lc.valid);
        }
        // Full-context weighted mean over the valid frames (the padded frame is zero).
        auto * ut = ggml_cont(ctx.ggml, ggml_transpose(ctx.ggml, ggml_reshape_2d(ctx.ggml, u.tensor, d, n)));
        auto * mean = ggml_scale(ctx.ggml, ggml_sum_rows(ctx.ggml, ut), 1.0f / static_cast<float>(lc.valid_frames));
        mean = ggml_mul(ctx.ggml, ggml_reshape_1d(ctx.ggml, mean, d), w.mean_weights.tensor);
        auto y = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, u);
        y = DepthwiseConv1dModule({d, c.conv_kernel, 1, static_cast<int>((c.conv_kernel - 1) / 2), 1, true})
            .build(ctx, y, w.depthwise);
        y = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, y);
        y = core::ensure_backend_addressable_layout(ctx, y);
        y = core::wrap_tensor(ggml_add(ctx.ggml, y.tensor, mean), y.shape, GGML_TYPE_F32);
        y = silu(ctx, mul(ctx, y, gate));
        x = add(ctx, x, linear(ctx, y, w.conv_out, d, d));
        x = add(ctx, x, feed_forward(ctx, x, w.ff2_in, w.ff2_out, d, hidden));
    }
    // norm(residual + 0.25 * (x - residual))
    auto * delta = ggml_scale(ctx.ggml, ggml_sub(ctx.ggml, x.tensor, x0.tensor), 0.25f);
    return rms_norm(ctx, core::wrap_tensor(ggml_add(ctx.ggml, x0.tensor, delta), x0.shape, GGML_TYPE_F32),
        w.output_norm, 1e-6f);
}

TensorValue conv2d(core::ModuleBuildContext & ctx, const TensorValue & x, const modules::Conv2dWeights & w,
    int64_t in, int64_t out, int64_t kernel, int stride, int pad_time, int pad_freq) {
    modules::Conv2dConfig c;
    c.in_channels = in;
    c.out_channels = out;
    c.kernel_height = kernel;
    c.kernel_width = kernel;
    c.stride_height = stride;
    c.stride_width = stride;
    c.padding_height = pad_time;
    c.padding_width = pad_freq;
    c.use_bias = true;
    return modules::Conv2dModule(c).build(ctx, x, w);
}

}  // namespace

int64_t ZapformerConfig::max_downsampling() const {
    int64_t out = 1;
    for (const auto & s : stacks) out = std::max(out, s.downsampling);
    return out;
}

int64_t ZapformerConfig::frontend_bins() const {
    return ((features - 1) / 2 - 1) / 2;
}

int64_t zapformer_encoder_frames(int64_t mel_frames) {
    const int64_t half = (mel_frames + 1) / 2;
    return (half + 1) / 2;
}

int64_t zapformer_padded_frames(const ZapformerConfig & config, int64_t mel_frames) {
    const int64_t frames = zapformer_encoder_frames(mel_frames);
    const int64_t m = config.max_downsampling();
    return (frames + m - 1) / m * m;
}

core::TensorValue build_zapformer_encoder(
    core::ModuleBuildContext & ctx,
    const ZapformerInputs & inputs,
    int64_t mel_frames,
    const ZapformerConfig & config,
    const ZapformerWeights & weights) {
    using namespace modules;
    const int64_t mels = config.features;
    const int64_t c1 = config.frontend_channels[0], c2 = config.frontend_channels[1], c3 = config.frontend_channels[2];

    // Frontend: [1, mels, T] -> [1, 1, T, mels] (time is height, mel is width).
    auto x = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, inputs.features);
    x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x),
        TensorShape::from_dims({1, 1, mel_frames, mels}));
    x = Pad2dModule({1, 1, 2, 0}).build(ctx, x);  // mel +-1, two causal frames on the left
    x = silu(ctx, conv2d(ctx, x, weights.conv1, 1, c1, 3, 1, 0, 0));
    x = silu(ctx, conv2d(ctx, x, weights.conv2, c1, c2, 3, 2, 1, 0));
    x = silu(ctx, conv2d(ctx, x, weights.conv3, c2, c3, 3, 2, 1, 0));
    {
        auto y = Pad2dModule({1, 1, 2, 0}).build(ctx, x);
        DepthwiseConv2dConfig dw;
        dw.channels = c3;
        dw.kernel_height = 3;
        dw.kernel_width = 3;
        dw.use_bias = true;
        y = DepthwiseConv2dModule(dw).build(ctx, y, weights.convnext_depthwise);
        y = silu(ctx, conv2d(ctx, y, weights.convnext_pointwise1, c3, 3 * c3, 1, 1, 0, 0));
        y = conv2d(ctx, y, weights.convnext_pointwise2, 3 * c3, c3, 1, 1, 0, 0);
        x = add(ctx, x, y);
    }
    const int64_t frames = x.shape.dims[2];
    const int64_t bins = x.shape.dims[3];
    if (frames != zapformer_encoder_frames(mel_frames) || bins != config.frontend_bins()) {
        throw std::runtime_error("GigaAM-v4 frontend produced an unexpected shape");
    }
    x = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x),
        TensorShape::from_dims({1, frames, c3 * bins}));
    x = linear(ctx, x, weights.out, c3 * bins, config.embed_dim);

    const int64_t padded = zapformer_padded_frames(config, mel_frames);
    if (padded > frames) {
        x = wrap(ggml_pad(ctx.ggml, contiguous(ctx, x.tensor), 0, static_cast<int>(padded - frames), 0, 0),
            {1, padded, config.embed_dim});
    }
    const bool has_padding = padded > frames;
    const int64_t embed = config.embed_dim;

    for (size_t s = 0; s < config.stacks.size(); ++s) {
        const auto & c = config.stacks[s];
        const auto & w = weights.stacks[s];
        const int64_t ds = c.downsampling;
        const int64_t n = padded / ds;
        if (ds > 1 && has_padding) {
            x = mul(ctx, x, inputs.valid);
        }
        x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x),
            TensorShape::from_dims({1, n, embed * ds}));
        const auto input = x;
        auto h = scale(ctx, linear(ctx, x, w.proj, embed * ds, c.dim), w.proj_scale);
        auto bypass = scale(ctx, h, w.residual_scales.at(0));
        x = scale(ctx, h, w.input_scale);
        LayerContext lc;
        lc.frames = n;
        // valid[::ds]: with ds > 1 every kept frame is real, because the padding is shorter than ds.
        lc.valid_frames = ds > 1 ? n : frames;
        lc.positions = core::wrap_tensor(ggml_view_1d(ctx.ggml, inputs.positions.tensor, n, 0),
            TensorShape::from_dims({n}), GGML_TYPE_I32);
        lc.valid = ds == 1 && has_padding ? &inputs.valid : nullptr;
        for (size_t i = 0; i < w.layers.size(); ++i) {
            x = build_layer(ctx, x, lc, c, w.layers[i], config.rope_theta);
            bypass = add(ctx, bypass, scale(ctx, x, w.residual_scales.at(i + 1)));
        }
        x = add(ctx, input, scale(ctx, linear(ctx, bypass, w.proj_t, c.dim, embed * ds), w.proj_scale));
        x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x),
            TensorShape::from_dims({1, padded, embed}));
    }
    // RmsNorm without weight: x * scale / sqrt(mean(x^2) + eps^2).
    auto * y = ggml_scale(ctx.ggml, ggml_rms_norm(ctx.ggml, contiguous(ctx, x.tensor), weights.out_eps * weights.out_eps),
        weights.out_scale);
    // The even padding stays: callers read only the first zapformer_encoder_frames() frames.
    return wrap(y, {1, padded, embed});
}

}  // namespace engine::models::gigaam_asr
