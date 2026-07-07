#include "engine/models/index_tts2/s2mel.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/torch_random.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::index_tts2 {
namespace {

namespace binding = engine::modules::binding;
namespace core = engine::core;
namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

constexpr int64_t kMelChannels = 80;
constexpr int64_t kContentDim = 1024;
constexpr int64_t kGptDim = 1280;
constexpr int64_t kHidden = 512;
constexpr int64_t kStyleDim = 192;
constexpr int64_t kDitLayers = 13;
constexpr int64_t kWavenetLayers = 8;
constexpr int64_t kWavenetKernel = 5;
constexpr int64_t kTimeFreqDim = 128;
constexpr int64_t kTimeEmbeddingDim = 256;
constexpr int64_t kDitFfnDim = 1536;
constexpr int64_t kDitHeads = 8;
constexpr int64_t kDitHeadDim = kHidden / kDitHeads;
constexpr float kLayerNormEps = 1.0e-6F;
constexpr float kRmsNormEps = 1.0e-5F;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

core::TensorValue sub(core::ModuleBuildContext & ctx, const core::TensorValue & lhs, const core::TensorValue & rhs) {
    core::validate_shape(rhs, lhs.shape, "Sub rhs");
    return core::wrap_tensor(ggml_sub(ctx.ggml, lhs.tensor, rhs.tensor), lhs.shape, GGML_TYPE_F32);
}

core::TensorValue scale(core::ModuleBuildContext & ctx, const core::TensorValue & input, float value) {
    return core::wrap_tensor(ggml_scale(ctx.ggml, input.tensor, value), input.shape, GGML_TYPE_F32);
}

core::TensorValue add_one(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    return core::wrap_tensor(ggml_scale_bias(ctx.ggml, input.tensor, 1.0F, 1.0F), input.shape, GGML_TYPE_F32);
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    return core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, input),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

core::TensorValue slice_last(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t start,
    int64_t length) {
    return modules::SliceModule({static_cast<int>(input.shape.rank - 1), start, length}).build(ctx, input);
}

core::TensorValue apply_channel_affine(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & weight,
    const core::TensorValue & bias,
    int64_t channels) {
    core::TensorShape broadcast_shape = {};
    broadcast_shape.rank = input.shape.rank;
    for (size_t axis = 0; axis < broadcast_shape.rank; ++axis) {
        broadcast_shape.dims[axis] = 1;
    }
    broadcast_shape.dims[1] = channels;
    auto weight_view = core::reshape_tensor(ctx, weight, broadcast_shape);
    auto bias_view = core::reshape_tensor(ctx, bias, broadcast_shape);
    auto weight_rep = modules::RepeatModule({input.shape}).build(ctx, weight_view);
    auto bias_rep = modules::RepeatModule({input.shape}).build(ctx, bias_view);
    return modules::AddModule{}.build(ctx, modules::MulModule{}.build(ctx, input, weight_rep), bias_rep);
}

core::TensorValue broadcast_batch_time(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t batch,
    int64_t frames,
    int64_t dims) {
    auto shaped = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, input), core::TensorShape::from_dims({batch, 1, dims}));
    return modules::RepeatModule({core::TensorShape::from_dims({batch, frames, dims})}).build(ctx, shaped);
}

core::TensorValue group_norm_1_group(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::NormWeights & weights,
    int64_t channels) {
    if (!weights.weight.has_value() || !weights.bias.has_value()) {
        throw std::runtime_error("IndexTTS2 S2Mel length regulator group norm requires affine weights");
    }
    const auto input4 = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, input),
        core::TensorShape::from_dims({input.shape.dims[0], channels, 1, input.shape.dims[2]}));
    auto normalized = core::wrap_tensor(ggml_group_norm(ctx.ggml, input4.tensor, 1, 1.0e-5F), input4.shape, GGML_TYPE_F32);
    normalized = apply_channel_affine(ctx, normalized, *weights.weight, *weights.bias, channels);
    return core::reshape_tensor(ctx, normalized, input.shape);
}

core::TensorValue mish(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    const auto softplus = core::wrap_tensor(ggml_softplus(ctx.ggml, input.tensor), input.shape, GGML_TYPE_F32);
    const auto tanh = core::wrap_tensor(ggml_tanh(ctx.ggml, softplus.tensor), input.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_mul(ctx.ggml, input.tensor, tanh.tensor), input.shape, GGML_TYPE_F32);
}

core::TensorValue timestep_embedding(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & timestep,
    const core::TensorValue & freqs,
    const modules::LinearWeights & linear0,
    const modules::LinearWeights & linear2) {
    const int64_t batch = timestep.shape.dims[0];
    auto t = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, timestep), core::TensorShape::from_dims({batch, 1}));
    auto freqs_batched = modules::RepeatModule({core::TensorShape::from_dims({batch, kTimeFreqDim})})
                             .build(ctx, core::reshape_tensor(ctx, freqs, core::TensorShape::from_dims({1, kTimeFreqDim})));
    auto args = modules::MulModule{}.build(ctx, modules::RepeatModule({freqs_batched.shape}).build(ctx, t), freqs_batched);
    args = scale(ctx, args, 1000.0F);
    auto cos_part = core::wrap_tensor(ggml_cos(ctx.ggml, core::ensure_backend_addressable_layout(ctx, args).tensor), args.shape, GGML_TYPE_F32);
    auto sin_part = core::wrap_tensor(ggml_sin(ctx.ggml, args.tensor), args.shape, GGML_TYPE_F32);
    auto emb = modules::ConcatModule({1}).build(ctx, cos_part, sin_part);
    emb = modules::LinearModule({kTimeEmbeddingDim, kHidden, true, GGML_PREC_F32}).build(ctx, emb, linear0);
    emb = modules::SiluModule{}.build(ctx, emb);
    return modules::LinearModule({kHidden, kHidden, true, GGML_PREC_F32}).build(ctx, emb, linear2);
}

core::TensorValue adaptive_rms_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & embedding,
    const IndexTTS2AdaLayerNormWeights & weights) {
    auto projected = modules::LinearModule({kHidden, 2 * kHidden, true, GGML_PREC_F32}).build(ctx, embedding, weights.project);
    auto weight = broadcast_batch_time(ctx, slice_last(ctx, projected, 0, kHidden), input.shape.dims[0], input.shape.dims[1], kHidden);
    auto bias = broadcast_batch_time(ctx, slice_last(ctx, projected, kHidden, kHidden), input.shape.dims[0], input.shape.dims[1], kHidden);
    auto normed = modules::RMSNormModule({kHidden, kRmsNormEps, true, false}).build(ctx, input, {weights.norm_weight, std::nullopt});
    return modules::AddModule{}.build(ctx, modules::MulModule{}.build(ctx, normed, weight), bias);
}

core::TensorValue cfm_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const IndexTTS2DitLayerWeights & weights) {
    auto qkv = modules::LinearModule({kHidden, 3 * kHidden, false, GGML_PREC_F32}).build(ctx, input, weights.qkv);
    auto q = slice_last(ctx, qkv, 0, kHidden);
    auto k = slice_last(ctx, qkv, kHidden, kHidden);
    auto v = slice_last(ctx, qkv, 2 * kHidden, kHidden);
    q = modules::RoPEModule({kDitHeadDim, GGML_ROPE_TYPE_NORMAL, 10000.0F}).build(ctx, reshape_heads(ctx, q, kDitHeads, kDitHeadDim), positions);
    k = modules::RoPEModule({kDitHeadDim, GGML_ROPE_TYPE_NORMAL, 10000.0F}).build(ctx, reshape_heads(ctx, k, kDitHeads, kDitHeadDim), positions);
    v = reshape_heads(ctx, v, kDitHeads, kDitHeadDim);
    auto qh = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    auto kh = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
    auto vh = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        core::ensure_backend_addressable_layout(ctx, qh).tensor,
        core::ensure_backend_addressable_layout(ctx, kh).tensor,
        core::ensure_backend_addressable_layout(ctx, vh).tensor,
        nullptr,
        1.0F / std::sqrt(static_cast<float>(kDitHeadDim)),
        0.0F,
        0.0F);
    ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
    auto context = core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], kDitHeads, kDitHeadDim}),
        GGML_TYPE_F32);
    context = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, context), input.shape);
    return modules::LinearModule({kHidden, kHidden, false, GGML_PREC_F32}).build(ctx, context, weights.attention_out);
}

core::TensorValue cfm_ffn(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const IndexTTS2DitLayerWeights & weights) {
    auto gate = modules::LinearModule({kHidden, kDitFfnDim, false, GGML_PREC_F32}).build(ctx, input, weights.ffn_w1);
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule({kHidden, kDitFfnDim, false, GGML_PREC_F32}).build(ctx, input, weights.ffn_w3);
    auto hidden = modules::MulModule{}.build(ctx, gate, up);
    return modules::LinearModule({kDitFfnDim, kHidden, false, GGML_PREC_F32}).build(ctx, hidden, weights.ffn_w2);
}

core::TensorValue cfm_transformer_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & timestep,
    const core::TensorValue & positions,
    const IndexTTS2DitLayerWeights & weights,
    const core::TensorValue * skip) {
    auto x = input;
    if (skip != nullptr) {
        x = modules::LinearModule({2 * kHidden, kHidden, true, GGML_PREC_F32})
                .build(ctx, modules::ConcatModule({2}).build(ctx, x, *skip), weights.skip_in);
    }
    auto attn = cfm_attention(ctx, adaptive_rms_norm(ctx, x, timestep, weights.attention_norm), positions, weights);
    auto h = modules::AddModule{}.build(ctx, x, attn);
    auto ff = cfm_ffn(ctx, adaptive_rms_norm(ctx, h, timestep, weights.ffn_norm), weights);
    return modules::AddModule{}.build(ctx, h, ff);
}

core::TensorValue cfm_wavenet(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const core::TensorValue & timestep_b,
    const IndexTTS2S2MelCfmWeights & weights) {
    auto g = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, timestep_b), core::TensorShape::from_dims({timestep_b.shape.dims[0], kHidden, 1}));
    g = modules::Conv1dModule({kHidden, 2 * kHidden * kWavenetLayers, 1, 1, 0, 1, true}).build(ctx, g, weights.wavenet_cond);
    auto output = sub(ctx, input_bct, input_bct);
    auto x = input_bct;
    for (int64_t i = 0; i < kWavenetLayers; ++i) {
        const int64_t dilation = 1;
        const int64_t padding = (kWavenetKernel * dilation - dilation) / 2;
        auto x_padded = modules::ReflectPad1dModule({padding, padding}).build(ctx, core::ensure_backend_addressable_layout(ctx, x));
        auto x_in = modules::Conv1dModule(
                        {kHidden, 2 * kHidden, kWavenetKernel, 1, 0, static_cast<int>(dilation), true})
                        .build(ctx, x_padded, weights.wavenet_layers[static_cast<size_t>(i)].in_layer);
        auto g_l = modules::SliceModule({1, i * 2 * kHidden, 2 * kHidden}).build(ctx, g);
        g_l = modules::RepeatModule({x_in.shape}).build(ctx, g_l);
        auto acts = modules::AddModule{}.build(ctx, x_in, g_l);
        auto tanh_part = modules::SliceModule({1, 0, kHidden}).build(ctx, acts);
        tanh_part = modules::TanhModule{}.build(ctx, tanh_part);
        auto sigmoid_part = modules::SliceModule({1, kHidden, kHidden}).build(ctx, acts);
        sigmoid_part = modules::SigmoidModule{}.build(ctx, sigmoid_part);
        acts = modules::MulModule{}.build(ctx, tanh_part, sigmoid_part);
        const int64_t res_skip_channels = i < kWavenetLayers - 1 ? 2 * kHidden : kHidden;
        auto res_skip = modules::Conv1dModule({kHidden, res_skip_channels, 1, 1, 0, 1, true})
                            .build(ctx, acts, weights.wavenet_layers[static_cast<size_t>(i)].res_skip_layer);
        if (i < kWavenetLayers - 1) {
            auto res = modules::SliceModule({1, 0, kHidden}).build(ctx, res_skip);
            auto skip = modules::SliceModule({1, kHidden, kHidden}).build(ctx, res_skip);
            x = modules::AddModule{}.build(ctx, x, res);
            output = modules::AddModule{}.build(ctx, output, skip);
        } else {
            output = modules::AddModule{}.build(ctx, output, res_skip);
        }
    }
    return output;
}

core::TensorValue cfm_final_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & timestep,
    const IndexTTS2S2MelCfmWeights & weights) {
    auto mod = modules::SiluModule{}.build(ctx, timestep);
    mod = modules::LinearModule({kHidden, 2 * kHidden, true, GGML_PREC_F32}).build(ctx, mod, weights.final_modulation);
    auto shift = broadcast_batch_time(ctx, slice_last(ctx, mod, 0, kHidden), input.shape.dims[0], input.shape.dims[1], kHidden);
    auto scale_v = broadcast_batch_time(ctx, slice_last(ctx, mod, kHidden, kHidden), input.shape.dims[0], input.shape.dims[1], kHidden);
    auto normed = modules::LayerNormModule({kHidden, kLayerNormEps, false, false}).build(ctx, input, {std::nullopt, std::nullopt});
    normed = modules::AddModule{}.build(ctx, modules::MulModule{}.build(ctx, normed, add_one(ctx, scale_v)), shift);
    return modules::LinearModule({kHidden, kHidden, true, GGML_PREC_F32}).build(ctx, normed, weights.final_linear);
}

core::TensorValue build_cfm_estimator(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x_bct,
    const core::TensorValue & prompt_bct,
    const core::TensorValue & cond_btc,
    const core::TensorValue & style_bc,
    const core::TensorValue & timestep_b,
    const core::TensorValue & positions,
    const IndexTTS2S2MelCfmWeights & weights) {
    const int64_t batch = x_bct.shape.dims[0];
    const int64_t frames = x_bct.shape.dims[2];
    auto t1 = timestep_embedding(ctx, timestep_b, weights.time_freqs, weights.time_mlp0, weights.time_mlp2);
    auto cond = modules::LinearModule({kHidden, kHidden, true, GGML_PREC_F32}).build(ctx, cond_btc, weights.cond_projection);
    auto x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x_bct);
    auto prompt = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, prompt_bct);
    auto style = broadcast_batch_time(ctx, style_bc, batch, frames, kStyleDim);
    auto hidden = modules::ConcatModule({2}).build(ctx, x, prompt);
    hidden = modules::ConcatModule({2}).build(ctx, hidden, cond);
    hidden = modules::ConcatModule({2}).build(ctx, hidden, style);
    hidden = modules::LinearModule({kHidden + 2 * kMelChannels + kStyleDim, kHidden, true, GGML_PREC_F32})
                 .build(ctx, hidden, weights.cond_x_merge);

    std::vector<core::TensorValue> skips;
    skips.reserve(static_cast<size_t>(kDitLayers / 2));
    for (int64_t i = 0; i < kDitLayers; ++i) {
        const core::TensorValue * skip = nullptr;
        if (i > kDitLayers / 2) {
            skip = &skips.back();
        }
        hidden = cfm_transformer_layer(ctx, hidden, t1, positions, weights.dit_layers[static_cast<size_t>(i)], skip);
        if (i > kDitLayers / 2) {
            skips.pop_back();
        } else if (i < kDitLayers / 2) {
            skips.push_back(hidden);
        }
    }
    hidden = adaptive_rms_norm(ctx, hidden, t1, weights.dit_norm);
    hidden = modules::LinearModule({kHidden + kMelChannels, kHidden, true, GGML_PREC_F32})
                 .build(ctx, modules::ConcatModule({2}).build(ctx, hidden, x), weights.skip_linear);
    auto wavenet_x = modules::LinearModule({kHidden, kHidden, true, GGML_PREC_F32}).build(ctx, hidden, weights.conv1);
    wavenet_x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, wavenet_x);
    auto t2 = timestep_embedding(ctx, timestep_b, weights.time2_freqs, weights.time2_mlp0, weights.time2_mlp2);
    wavenet_x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, cfm_wavenet(ctx, wavenet_x, t2, weights));
    auto projected = modules::LinearModule({kHidden, kHidden, true, GGML_PREC_F32}).build(ctx, hidden, weights.res_projection);
    hidden = modules::AddModule{}.build(ctx, wavenet_x, projected);
    hidden = cfm_final_layer(ctx, hidden, t1, weights);
    hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, hidden);
    return modules::Conv1dModule({kHidden, kMelChannels, 1, 1, 0, 1, true}).build(ctx, hidden, weights.conv2);
}

std::vector<float> fuse_weight_norm_linear(
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_features,
    int64_t in_features) {
    const auto g = source.require_f32(prefix + ".weight_g", {out_features, 1});
    const auto v = source.require_f32(prefix + ".weight_v", {out_features, in_features});
    std::vector<float> weight(v.size(), 0.0F);
    for (int64_t out = 0; out < out_features; ++out) {
        double norm = 0.0;
        for (int64_t in = 0; in < in_features; ++in) {
            const float value = v[static_cast<size_t>(out * in_features + in)];
            norm += static_cast<double>(value) * static_cast<double>(value);
        }
        const float scale = g[static_cast<size_t>(out)] / static_cast<float>(std::sqrt(norm));
        for (int64_t in = 0; in < in_features; ++in) {
            const size_t index = static_cast<size_t>(out * in_features + in);
            weight[index] = v[index] * scale;
        }
    }
    return weight;
}

std::vector<float> fuse_weight_norm_conv1d(
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_size) {
    const auto g = source.require_f32(prefix + ".weight_g", {out_channels, 1, 1});
    const auto v = source.require_f32(prefix + ".weight_v", {out_channels, in_channels, kernel_size});
    std::vector<float> weight(v.size(), 0.0F);
    for (int64_t out = 0; out < out_channels; ++out) {
        double norm = 0.0;
        for (int64_t in = 0; in < in_channels; ++in) {
            for (int64_t k = 0; k < kernel_size; ++k) {
                const float value = v[static_cast<size_t>((out * in_channels + in) * kernel_size + k)];
                norm += static_cast<double>(value) * static_cast<double>(value);
            }
        }
        const float scale = g[static_cast<size_t>(out)] / static_cast<float>(std::sqrt(norm));
        for (int64_t in = 0; in < in_channels; ++in) {
            for (int64_t k = 0; k < kernel_size; ++k) {
                const size_t index = static_cast<size_t>((out * in_channels + in) * kernel_size + k);
                weight[index] = v[index] * scale;
            }
        }
    }
    return weight;
}

engine::modules::LinearWeights load_weight_norm_linear(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type,
    int64_t out_features,
    int64_t in_features) {
    engine::modules::LinearWeights weights;
    weights.weight = store.make_from_f32(
        engine::core::TensorShape::from_dims({out_features, in_features}),
        storage_type,
        fuse_weight_norm_linear(source, prefix, out_features, in_features));
    weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    return weights;
}

engine::modules::Conv1dWeights load_weight_norm_conv1d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_size) {
    engine::modules::Conv1dWeights weights;
    weights.weight = store.make_from_f32(
        engine::core::TensorShape::from_dims({out_channels, in_channels, kernel_size}),
        storage_type,
        fuse_weight_norm_conv1d(source, prefix, out_channels, in_channels, kernel_size));
    weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    return weights;
}

IndexTTS2AdaLayerNormWeights load_ada_norm(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type) {
    return {
        store.load_f32_tensor(source, prefix + ".norm.weight", {kHidden}),
        binding::linear_from_source(store, source, prefix + ".project_layer", storage_type, 2 * kHidden, kHidden, true),
    };
}

IndexTTS2DitLayerWeights load_dit_layer(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    int64_t layer_index,
    engine::assets::TensorStorageType storage_type) {
    const std::string prefix = "cfm.estimator.transformer.layers." + std::to_string(layer_index);
    IndexTTS2DitLayerWeights layer;
    layer.attention_norm = load_ada_norm(store, source, prefix + ".attention_norm", storage_type);
    layer.qkv = binding::linear_from_source(store, source, prefix + ".attention.wqkv", storage_type, 3 * kHidden, kHidden, false);
    layer.attention_out = binding::linear_from_source(store, source, prefix + ".attention.wo", storage_type, kHidden, kHidden, false);
    layer.ffn_norm = load_ada_norm(store, source, prefix + ".ffn_norm", storage_type);
    layer.ffn_w1 = binding::linear_from_source(store, source, prefix + ".feed_forward.w1", storage_type, kDitFfnDim, kHidden, false);
    layer.ffn_w2 = binding::linear_from_source(store, source, prefix + ".feed_forward.w2", storage_type, kHidden, kDitFfnDim, false);
    layer.ffn_w3 = binding::linear_from_source(store, source, prefix + ".feed_forward.w3", storage_type, kDitFfnDim, kHidden, false);
    layer.skip_in = binding::linear_from_source(store, source, prefix + ".skip_in_linear", storage_type, kHidden, 2 * kHidden, true);
    return layer;
}

IndexTTS2LengthRegulatorWeights load_length_regulator(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    IndexTTS2LengthRegulatorWeights weights;
    weights.content_projection = binding::linear_from_source(
        store,
        source,
        "length_regulator.content_in_proj",
        matmul_storage_type,
        kHidden,
        kContentDim,
        true);
    for (int64_t i : {0, 3, 6, 9}) {
        weights.convs.push_back(binding::conv1d_from_source(
            store,
            source,
            "length_regulator.model." + std::to_string(i),
            conv_storage_type,
            kHidden,
            kHidden,
            3,
            true));
    }
    for (int64_t i : {1, 4, 7, 10}) {
        weights.norms.push_back({
            store.load_f32_tensor(source, "length_regulator.model." + std::to_string(i) + ".weight", {kHidden}),
            store.load_f32_tensor(source, "length_regulator.model." + std::to_string(i) + ".bias", {kHidden}),
        });
    }
    weights.output = binding::conv1d_from_source(
        store,
        source,
        "length_regulator.model.12",
        conv_storage_type,
        kHidden,
        kHidden,
        1,
        true);
    return weights;
}

IndexTTS2S2MelCfmWeights load_cfm(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    IndexTTS2S2MelCfmWeights weights;
    weights.x_embedder = load_weight_norm_linear(store, source, "cfm.estimator.x_embedder", matmul_storage_type, kHidden, kMelChannels);
    weights.cond_projection = binding::linear_from_source(store, source, "cfm.estimator.cond_projection", matmul_storage_type, kHidden, kHidden, true);
    weights.cond_x_merge = binding::linear_from_source(
        store,
        source,
        "cfm.estimator.cond_x_merge_linear",
        matmul_storage_type,
        kHidden,
        kHidden + 2 * kMelChannels + kStyleDim,
        true);
    weights.skip_linear = binding::linear_from_source(
        store,
        source,
        "cfm.estimator.skip_linear",
        matmul_storage_type,
        kHidden,
        kHidden + kMelChannels,
        true);
    weights.time_freqs = store.load_f32_tensor(source, "cfm.estimator.t_embedder.freqs", {kTimeFreqDim});
    weights.time_mlp0 = binding::linear_from_source(store, source, "cfm.estimator.t_embedder.mlp.0", matmul_storage_type, kHidden, kTimeEmbeddingDim, true);
    weights.time_mlp2 = binding::linear_from_source(store, source, "cfm.estimator.t_embedder.mlp.2", matmul_storage_type, kHidden, kHidden, true);
    weights.time2_freqs = store.load_f32_tensor(source, "cfm.estimator.t_embedder2.freqs", {kTimeFreqDim});
    weights.time2_mlp0 = binding::linear_from_source(store, source, "cfm.estimator.t_embedder2.mlp.0", matmul_storage_type, kHidden, kTimeEmbeddingDim, true);
    weights.time2_mlp2 = binding::linear_from_source(store, source, "cfm.estimator.t_embedder2.mlp.2", matmul_storage_type, kHidden, kHidden, true);
    weights.dit_layers.reserve(static_cast<size_t>(kDitLayers));
    for (int64_t i = 0; i < kDitLayers; ++i) {
        weights.dit_layers.push_back(load_dit_layer(store, source, i, matmul_storage_type));
    }
    weights.dit_norm = load_ada_norm(store, source, "cfm.estimator.transformer.norm", matmul_storage_type);
    weights.conv1 = binding::linear_from_source(store, source, "cfm.estimator.conv1", matmul_storage_type, kHidden, kHidden, true);
    weights.res_projection = binding::linear_from_source(store, source, "cfm.estimator.res_projection", matmul_storage_type, kHidden, kHidden, true);
    weights.wavenet_cond = load_weight_norm_conv1d(
        store,
        source,
        "cfm.estimator.wavenet.cond_layer.conv.conv",
        conv_storage_type,
        2 * kHidden * kWavenetLayers,
        kHidden,
        1);
    weights.wavenet_layers.reserve(static_cast<size_t>(kWavenetLayers));
    for (int64_t i = 0; i < kWavenetLayers; ++i) {
        const int64_t res_skip_channels = i < kWavenetLayers - 1 ? 2 * kHidden : kHidden;
        weights.wavenet_layers.push_back({
            load_weight_norm_conv1d(
                store,
                source,
                "cfm.estimator.wavenet.in_layers." + std::to_string(i) + ".conv.conv",
                conv_storage_type,
                2 * kHidden,
                kHidden,
                kWavenetKernel),
            load_weight_norm_conv1d(
                store,
                source,
                "cfm.estimator.wavenet.res_skip_layers." + std::to_string(i) + ".conv.conv",
                conv_storage_type,
                res_skip_channels,
                kHidden,
                1),
        });
    }
    weights.final_modulation = binding::linear_from_source(
        store,
        source,
        "cfm.estimator.final_layer.adaLN_modulation.1",
        matmul_storage_type,
        2 * kHidden,
        kHidden,
        true);
    weights.final_linear = load_weight_norm_linear(
        store,
        source,
        "cfm.estimator.final_layer.linear",
        matmul_storage_type,
        kHidden,
        kHidden);
    weights.conv2 = binding::conv1d_from_source(store, source, "cfm.estimator.conv2", conv_storage_type, kMelChannels, kHidden, 1, true);
    return weights;
}

}  // namespace

std::shared_ptr<const IndexTTS2S2MelWeights> load_index_tts2_s2mel_weights(
    const IndexTTS2Assets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes) {
    if (assets.s2mel_weights == nullptr) {
        throw std::runtime_error("IndexTTS2 S2Mel requires tensor source");
    }
    auto weights = std::make_shared<IndexTTS2S2MelWeights>();
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        backend,
        backend_type,
        "index_tts2.s2mel.weights",
        weight_context_bytes);

    const auto & source = *assets.s2mel_weights;
    weights->gpt_layer.linear0 = binding::linear_from_source(*weights->store, source, "gpt_layer.0", matmul_storage_type, 256, 1280, true);
    weights->gpt_layer.linear1 = binding::linear_from_source(*weights->store, source, "gpt_layer.1", matmul_storage_type, 128, 256, true);
    weights->gpt_layer.linear2 = binding::linear_from_source(*weights->store, source, "gpt_layer.2", matmul_storage_type, kContentDim, 128, true);
    weights->length_regulator = load_length_regulator(*weights->store, source, matmul_storage_type, conv_storage_type);
    weights->cfm = load_cfm(*weights->store, source, matmul_storage_type, conv_storage_type);

    weights->store->upload();
    assets.s2mel_weights->release_storage();
    return weights;
}

class IndexTTS2S2MelRuntime::GptLayerGraph {
public:
    GptLayerGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const IndexTTS2S2MelWeights> weights,
        int64_t frames,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          frames_(frames) {
        if (frames_ <= 0) {
            throw std::runtime_error("IndexTTS2 S2Mel GPT layer graph requires positive frame count");
        }
        if (weights_ == nullptr) {
            throw std::runtime_error("IndexTTS2 S2Mel GPT layer graph requires weights");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 S2Mel GPT layer graph context");
        }
        ggml_init_params input_params{16ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 S2Mel GPT layer input context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "index_tts2.s2mel.gpt_layer", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{
            input_ctx_.get(),
            "index_tts2.s2mel.gpt_layer.inputs",
            execution_.backend_type()};
        input_ =
            core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, frames_, kGptDim})).tensor;
        ggml_set_input(input_);
        auto x = core::wrap_tensor(input_, core::TensorShape::from_dims({1, frames_, kGptDim}), GGML_TYPE_F32);
        x = modules::LinearModule({kGptDim, 256, true}).build(ctx, x, weights_->gpt_layer.linear0);
        x = modules::LinearModule({256, 128, true}).build(ctx, x, weights_->gpt_layer.linear1);
        x = modules::LinearModule({128, kContentDim, true}).build(ctx, x, weights_->gpt_layer.linear2);
        output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), static_cast<size_t>(std::max<int64_t>(8192, frames_ * 128)), false);
        ggml_build_forward_expand(graph_, output_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate IndexTTS2 S2Mel GPT layer input buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate IndexTTS2 S2Mel GPT layer graph");
        }
        debug::timing_log_scalar("index_tts2.s2mel.gpt_layer.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("index_tts2.s2mel.gpt_layer.frames", frames_);
    }

    ~GptLayerGraph() {
        clear_graph();
    }

    int64_t frames() const noexcept {
        return frames_;
    }

    IndexTTS2S2MelSequence run(const std::vector<float> & latent) {
        if (static_cast<int64_t>(latent.size()) != frames_ * kGptDim) {
            throw std::runtime_error("IndexTTS2 S2Mel GPT layer latent size mismatch");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(input_, latent.data(), 0, latent.size() * sizeof(float));
        debug::timing_log_scalar("index_tts2.s2mel.gpt_layer.input_upload_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        timing_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        debug::timing_log_scalar("index_tts2.s2mel.gpt_layer.graph.compute_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("IndexTTS2 S2Mel GPT layer graph compute failed");
        }
        IndexTTS2S2MelSequence output;
        output.frames = frames_;
        output.dims = kContentDim;
        timing_start = Clock::now();
        output.values = core::read_tensor_f32(output_);
        debug::timing_log_scalar("index_tts2.s2mel.gpt_layer.output_read_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        return output;
    }

private:
    void clear_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
            input_buffer_ = nullptr;
        }
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const IndexTTS2S2MelWeights> weights_;
    int64_t frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
};

class IndexTTS2S2MelRuntime::LengthRegulatorGraph {
public:
    LengthRegulatorGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const IndexTTS2S2MelWeights> weights,
        int64_t input_frames,
        int64_t output_frames,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          input_frames_(input_frames),
          output_frames_(output_frames) {
        if (input_frames_ <= 0 || output_frames_ <= 0) {
            throw std::runtime_error("IndexTTS2 S2Mel length regulator graph requires positive frame counts");
        }
        if (weights_ == nullptr) {
            throw std::runtime_error("IndexTTS2 S2Mel length regulator graph requires weights");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 S2Mel length regulator graph context");
        }
        ggml_init_params input_params{16ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 S2Mel length regulator input context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "index_tts2.s2mel.length_regulator", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{
            input_ctx_.get(),
            "index_tts2.s2mel.length_regulator.inputs",
            execution_.backend_type()};
        input_ = core::make_tensor(
                     input_ctx,
                     GGML_TYPE_F32,
                     core::TensorShape::from_dims({1, input_frames_, kContentDim}))
                     .tensor;
        mask_ =
            core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, output_frames_, kHidden})).tensor;
        ggml_set_input(input_);
        ggml_set_input(mask_);
        auto x = core::wrap_tensor(input_, core::TensorShape::from_dims({1, input_frames_, kContentDim}), GGML_TYPE_F32);
        x = modules::LinearModule({kContentDim, kHidden, true}).build(ctx, x, weights_->length_regulator.content_projection);
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        x = modules::Interpolate1dModule({output_frames_, modules::Interpolate1dMode::Nearest}).build(ctx, x);
        for (size_t layer = 0; layer < weights_->length_regulator.convs.size(); ++layer) {
            x = modules::Conv1dModule({kHidden, kHidden, 3, 1, 1, 1, true}).build(ctx, x, weights_->length_regulator.convs[layer]);
            x = group_norm_1_group(ctx, x, weights_->length_regulator.norms[layer], kHidden);
            x = mish(ctx, x);
        }
        x = modules::Conv1dModule({kHidden, kHidden, 1, 1, 0, 1, true}).build(ctx, x, weights_->length_regulator.output);
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        x = core::ensure_backend_addressable_layout(ctx, x);
        auto mask = core::wrap_tensor(mask_, core::TensorShape::from_dims({1, output_frames_, kHidden}), GGML_TYPE_F32);
        auto out = modules::MulModule{}.build(ctx, x, mask);
        output_ = core::ensure_backend_addressable_layout(ctx, out).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), static_cast<size_t>(std::max<int64_t>(32768, output_frames_ * 512)), false);
        ggml_build_forward_expand(graph_, output_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate IndexTTS2 S2Mel length regulator input buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate IndexTTS2 S2Mel length regulator graph");
        }
        mask_values_.assign(static_cast<size_t>(output_frames_ * kHidden), 1.0F);
        core::write_tensor_f32(
            core::wrap_tensor(mask_, core::TensorShape::from_dims({1, output_frames_, kHidden}), GGML_TYPE_F32),
            mask_values_);
        debug::timing_log_scalar("index_tts2.s2mel.length_regulator.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("index_tts2.s2mel.length_regulator.input_frames", input_frames_);
        debug::trace_log_scalar("index_tts2.s2mel.length_regulator.output_frames", output_frames_);
    }

    ~LengthRegulatorGraph() {
        clear_graph();
    }

    bool matches(int64_t input_frames, int64_t output_frames) const noexcept {
        return input_frames_ == input_frames && output_frames_ == output_frames;
    }

    IndexTTS2S2MelSequence run(const std::vector<float> & content) {
        if (static_cast<int64_t>(content.size()) != input_frames_ * kContentDim) {
            throw std::runtime_error("IndexTTS2 S2Mel length regulator content size mismatch");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(input_, content.data(), 0, content.size() * sizeof(float));
        debug::timing_log_scalar("index_tts2.s2mel.length_regulator.input_upload_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        timing_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        debug::timing_log_scalar("index_tts2.s2mel.length_regulator.graph.compute_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("IndexTTS2 S2Mel length regulator graph compute failed");
        }
        IndexTTS2S2MelSequence output;
        output.frames = output_frames_;
        output.dims = kHidden;
        timing_start = Clock::now();
        output.values = core::read_tensor_f32(output_);
        debug::timing_log_scalar("index_tts2.s2mel.length_regulator.output_read_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        return output;
    }

private:
    void clear_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
            input_buffer_ = nullptr;
        }
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const IndexTTS2S2MelWeights> weights_;
    int64_t input_frames_ = 0;
    int64_t output_frames_ = 0;
    std::vector<float> mask_values_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
};

class IndexTTS2S2MelRuntime::CfmGraph {
public:
    CfmGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const IndexTTS2S2MelWeights> weights,
        int64_t frames,
        bool use_cfg,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          frames_(frames),
          use_cfg_(use_cfg),
          batch_(use_cfg ? 2 : 1) {
        if (frames_ <= 0) {
            throw std::runtime_error("IndexTTS2 S2Mel CFM graph requires positive frame count");
        }
        if (weights_ == nullptr) {
            throw std::runtime_error("IndexTTS2 S2Mel CFM graph requires weights");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 S2Mel CFM graph context");
        }
        ggml_init_params input_params{16ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 S2Mel CFM input context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "index_tts2.s2mel.cfm", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "index_tts2.s2mel.cfm.inputs", execution_.backend_type()};
        x_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_, kMelChannels, frames_}))
                 .tensor;
        prompt_ =
            core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_, kMelChannels, frames_})).tensor;
        cond_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_, frames_, kHidden})).tensor;
        style_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_, kStyleDim})).tensor;
        timestep_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_})).tensor;
        positions_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({frames_})).tensor;
        ggml_set_input(x_);
        ggml_set_input(prompt_);
        ggml_set_input(cond_);
        ggml_set_input(style_);
        ggml_set_input(timestep_);
        ggml_set_input(positions_);
        auto output = build_cfm_estimator(
            ctx,
            core::wrap_tensor(x_, core::TensorShape::from_dims({batch_, kMelChannels, frames_}), GGML_TYPE_F32),
            core::wrap_tensor(prompt_, core::TensorShape::from_dims({batch_, kMelChannels, frames_}), GGML_TYPE_F32),
            core::wrap_tensor(cond_, core::TensorShape::from_dims({batch_, frames_, kHidden}), GGML_TYPE_F32),
            core::wrap_tensor(style_, core::TensorShape::from_dims({batch_, kStyleDim}), GGML_TYPE_F32),
            core::wrap_tensor(timestep_, core::TensorShape::from_dims({batch_}), GGML_TYPE_F32),
            core::wrap_tensor(positions_, core::TensorShape::from_dims({frames_}), GGML_TYPE_I32),
            weights_->cfm);
        output_ = core::ensure_backend_addressable_layout(ctx, output).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), static_cast<size_t>(std::max<int64_t>(131072, frames_ * 4096)), false);
        ggml_build_forward_expand(graph_, output_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate IndexTTS2 S2Mel CFM input buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate IndexTTS2 S2Mel CFM graph");
        }
        positions_values_.assign(static_cast<size_t>(frames_), 0);
        for (int64_t i = 0; i < frames_; ++i) {
            positions_values_[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        ggml_backend_tensor_set(positions_, positions_values_.data(), 0, positions_values_.size() * sizeof(int32_t));
        debug::timing_log_scalar("index_tts2.s2mel.cfm.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("index_tts2.s2mel.cfm.frames", frames_);
        debug::trace_log_scalar("index_tts2.s2mel.cfm.batch", batch_);
    }

    ~CfmGraph() {
        clear_graph();
    }

    bool matches(int64_t frames, bool use_cfg) const noexcept {
        return frames_ == frames && use_cfg_ == use_cfg;
    }

    std::vector<float> run(
        const std::vector<float> & x,
        const std::vector<float> & prompt,
        const std::vector<float> & cond,
        const std::vector<float> & style,
        const std::vector<float> & timestep) {
        const int64_t mel_values = batch_ * kMelChannels * frames_;
        if (static_cast<int64_t>(x.size()) != mel_values ||
            static_cast<int64_t>(prompt.size()) != mel_values ||
            static_cast<int64_t>(cond.size()) != batch_ * frames_ * kHidden ||
            static_cast<int64_t>(style.size()) != batch_ * kStyleDim ||
            static_cast<int64_t>(timestep.size()) != batch_) {
            throw std::runtime_error("IndexTTS2 S2Mel CFM graph input shape mismatch");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(x_, x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(prompt_, prompt.data(), 0, prompt.size() * sizeof(float));
        ggml_backend_tensor_set(cond_, cond.data(), 0, cond.size() * sizeof(float));
        ggml_backend_tensor_set(style_, style.data(), 0, style.size() * sizeof(float));
        ggml_backend_tensor_set(timestep_, timestep.data(), 0, timestep.size() * sizeof(float));
        debug::timing_log_scalar("index_tts2.s2mel.cfm.input_upload_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        timing_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        debug::timing_log_scalar("index_tts2.s2mel.cfm.graph.compute_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("IndexTTS2 S2Mel CFM graph compute failed");
        }
        std::vector<float> out(static_cast<size_t>(mel_values), 0.0F);
        timing_start = Clock::now();
        ggml_backend_tensor_get(output_, out.data(), 0, out.size() * sizeof(float));
        debug::timing_log_scalar("index_tts2.s2mel.cfm.output_read_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        return out;
    }

private:
    void clear_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
            input_buffer_ = nullptr;
        }
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const IndexTTS2S2MelWeights> weights_;
    int64_t frames_ = 0;
    bool use_cfg_ = false;
    int64_t batch_ = 1;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * x_ = nullptr;
    ggml_tensor * prompt_ = nullptr;
    ggml_tensor * cond_ = nullptr;
    ggml_tensor * style_ = nullptr;
    ggml_tensor * timestep_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    std::vector<int32_t> positions_values_;
};

void copy_row(
    const std::vector<float> & src,
    int64_t src_row,
    std::vector<float> & dst,
    int64_t dst_row,
    int64_t row_values) {
    std::copy_n(
        src.data() + static_cast<std::ptrdiff_t>(src_row * row_values),
        static_cast<size_t>(row_values),
        dst.data() + static_cast<std::ptrdiff_t>(dst_row * row_values));
}

std::vector<float> repeat_or_zero_rows(const std::vector<float> & values, int64_t row_values, bool use_cfg, bool zero_second) {
    if (!use_cfg) {
        return values;
    }
    std::vector<float> out(static_cast<size_t>(2 * row_values), 0.0F);
    copy_row(values, 0, out, 0, row_values);
    if (!zero_second) {
        copy_row(values, 0, out, 1, row_values);
    }
    return out;
}

void zero_prompt_region(std::vector<float> & values, int64_t channels, int64_t frames, int64_t prompt_frames) {
    if (prompt_frames < 0 || prompt_frames > frames) {
        throw std::runtime_error("IndexTTS2 S2Mel CFM prompt frame count is out of range");
    }
    for (int64_t c = 0; c < channels; ++c) {
        auto begin = values.begin() + static_cast<std::ptrdiff_t>(c * frames);
        std::fill(begin, begin + static_cast<std::ptrdiff_t>(prompt_frames), 0.0F);
    }
}

std::vector<float> make_prompt_x(
    const std::vector<float> & prompt,
    int64_t channels,
    int64_t frames,
    int64_t prompt_frames) {
    std::vector<float> out(static_cast<size_t>(channels * frames), 0.0F);
    if (static_cast<int64_t>(prompt.size()) != channels * prompt_frames) {
        throw std::runtime_error("IndexTTS2 S2Mel CFM reference mel shape mismatch");
    }
    for (int64_t c = 0; c < channels; ++c) {
        std::copy_n(
            prompt.data() + static_cast<std::ptrdiff_t>(c * prompt_frames),
            static_cast<size_t>(prompt_frames),
            out.data() + static_cast<std::ptrdiff_t>(c * frames));
    }
    return out;
}

std::vector<float> make_condition_with_prompt(
    const std::vector<float> & condition,
    int64_t frames) {
    if (static_cast<int64_t>(condition.size()) != frames * kHidden) {
        throw std::runtime_error("IndexTTS2 S2Mel CFM condition shape mismatch");
    }
    return condition;
}

IndexTTS2S2MelRuntime::IndexTTS2S2MelRuntime(
    std::shared_ptr<const IndexTTS2Assets> assets,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type)
    : assets_(std::move(assets)),
      execution_(&execution),
      graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr) {
        throw std::runtime_error("IndexTTS2 S2Mel runtime requires assets");
    }
    if (graph_arena_bytes_ == 0) {
        throw std::runtime_error("IndexTTS2 S2Mel graph arena must be non-zero");
    }
    weights_ = load_index_tts2_s2mel_weights(
        *assets_,
        execution.backend(),
        execution.backend_type(),
        matmul_storage_type,
        conv_storage_type,
        weight_context_bytes);
}

IndexTTS2S2MelRuntime::~IndexTTS2S2MelRuntime() = default;

void IndexTTS2S2MelRuntime::prepare_gpt_layer(int64_t frames) {
    if (execution_ == nullptr) {
        throw std::runtime_error("IndexTTS2 S2Mel runtime execution context is missing");
    }
    if (gpt_layer_graph_ != nullptr && gpt_layer_graph_->frames() == frames) {
        return;
    }
    gpt_layer_graph_.reset();
    gpt_layer_graph_ = std::make_unique<GptLayerGraph>(*execution_, weights_, frames, graph_arena_bytes_);
}

void IndexTTS2S2MelRuntime::prepare_length_regulator(int64_t input_frames, int64_t output_frames) {
    if (execution_ == nullptr) {
        throw std::runtime_error("IndexTTS2 S2Mel runtime execution context is missing");
    }
    if (length_regulator_graph_ != nullptr && length_regulator_graph_->matches(input_frames, output_frames)) {
        return;
    }
    length_regulator_graph_.reset();
    length_regulator_graph_ = std::make_unique<LengthRegulatorGraph>(
        *execution_,
        weights_,
        input_frames,
        output_frames,
        graph_arena_bytes_);
}

void IndexTTS2S2MelRuntime::prepare_cfm(int64_t total_frames, bool use_cfg) {
    if (execution_ == nullptr) {
        throw std::runtime_error("IndexTTS2 S2Mel runtime execution context is missing");
    }
    if (total_frames <= 0) {
        throw std::runtime_error("IndexTTS2 S2Mel CFM prepare requires positive frame count");
    }
    if (cfm_graph_ != nullptr && cfm_graph_->matches(total_frames, use_cfg)) {
        return;
    }
    cfm_graph_.reset();
    cfm_graph_ = std::make_unique<CfmGraph>(*execution_, weights_, total_frames, use_cfg, graph_arena_bytes_);
}

void IndexTTS2S2MelRuntime::release_pre_cfm_graphs() {
    gpt_layer_graph_.reset();
    length_regulator_graph_.reset();
}

void IndexTTS2S2MelRuntime::release_cfm_graph() {
    cfm_graph_.reset();
}

IndexTTS2S2MelSequence IndexTTS2S2MelRuntime::project_gpt_latent(const std::vector<float> & latent, int64_t frames) {
    if (gpt_layer_graph_ == nullptr || gpt_layer_graph_->frames() != frames) {
        throw std::runtime_error("IndexTTS2 S2Mel GPT layer graph was not prepared for this latent length");
    }
    return gpt_layer_graph_->run(latent);
}

IndexTTS2S2MelSequence IndexTTS2S2MelRuntime::regulate_length(
    const std::vector<float> & content,
    int64_t input_frames,
    int64_t output_frames) {
    if (length_regulator_graph_ == nullptr || !length_regulator_graph_->matches(input_frames, output_frames)) {
        throw std::runtime_error("IndexTTS2 S2Mel length regulator graph was not prepared for this shape");
    }
    return length_regulator_graph_->run(content);
}

IndexTTS2S2MelMel IndexTTS2S2MelRuntime::infer_mel(
    const std::vector<float> & condition,
    int64_t total_frames,
    const std::vector<float> & reference_mel,
    int64_t reference_frames,
    const std::vector<float> & style,
    int64_t diffusion_steps,
    float cfg_rate,
    uint32_t seed,
    uint64_t rng_offset_blocks) {
    if (condition.empty() || reference_mel.empty() || style.empty()) {
        throw std::runtime_error("IndexTTS2 S2Mel CFM requires non-empty condition, reference mel, and style");
    }
    if (total_frames <= 0 || reference_frames <= 0 || reference_frames > total_frames) {
        throw std::runtime_error("IndexTTS2 S2Mel CFM frame counts are invalid");
    }
    if (diffusion_steps <= 0) {
        throw std::runtime_error("IndexTTS2 S2Mel CFM diffusion steps must be positive");
    }
    if (static_cast<int64_t>(style.size()) != kStyleDim) {
        throw std::runtime_error("IndexTTS2 S2Mel CFM style shape mismatch");
    }
    const bool use_cfg = cfg_rate > 0.0F;
    if (cfm_graph_ == nullptr || !cfm_graph_->matches(total_frames, use_cfg)) {
        throw std::runtime_error("IndexTTS2 S2Mel CFM graph was not prepared for this shape");
    }
    const auto rng_policy = engine::sampling::resolve_torch_cuda_sampling_policy(
        execution_->backend_type(),
        execution_->config().device,
        "index_tts2.s2mel.cuda_sampling_policy",
        "IndexTTS2",
        engine::sampling::TorchCudaSamplingPolicyFailureMode::StrictCuda);
    std::vector<float> x = engine::sampling::generate_torch_cuda_tensor_iterator_randn(
        static_cast<size_t>(kMelChannels * total_frames),
        seed,
        rng_offset_blocks,
        rng_policy,
        engine::sampling::TorchRandnPrecision::Float32);
    auto prompt_x = make_prompt_x(reference_mel, kMelChannels, total_frames, reference_frames);
    zero_prompt_region(x, kMelChannels, total_frames, reference_frames);
    auto mu = make_condition_with_prompt(condition, total_frames);

    const auto cfm_start = Clock::now();
    double graph_ms = 0.0;
    float t = 0.0F;
    float dt = 1.0F / static_cast<float>(diffusion_steps);
    for (int64_t step = 1; step <= diffusion_steps; ++step) {
        const auto graph_start = Clock::now();
        auto x_batched = repeat_or_zero_rows(x, kMelChannels * total_frames, use_cfg, false);
        auto prompt_batched = repeat_or_zero_rows(prompt_x, kMelChannels * total_frames, use_cfg, true);
        auto cond_batched = repeat_or_zero_rows(mu, total_frames * kHidden, use_cfg, true);
        auto style_batched = repeat_or_zero_rows(style, kStyleDim, use_cfg, true);
        std::vector<float> timestep(static_cast<size_t>(use_cfg ? 2 : 1), t);
        const auto velocity = cfm_graph_->run(x_batched, prompt_batched, cond_batched, style_batched, timestep);
        graph_ms += engine::debug::elapsed_ms(graph_start);
        const int64_t row_values = kMelChannels * total_frames;
        for (int64_t i = 0; i < row_values; ++i) {
            float dphi = velocity[static_cast<size_t>(i)];
            if (use_cfg) {
                dphi = (1.0F + cfg_rate) * dphi - cfg_rate * velocity[static_cast<size_t>(row_values + i)];
            }
            x[static_cast<size_t>(i)] += dt * dphi;
        }
        t += dt;
        if (step < diffusion_steps) {
            dt = (static_cast<float>(step + 1) / static_cast<float>(diffusion_steps)) - t;
        }
        zero_prompt_region(x, kMelChannels, total_frames, reference_frames);
    }

    IndexTTS2S2MelMel out;
    out.frames = total_frames - reference_frames;
    out.channels = kMelChannels;
    out.values.resize(static_cast<size_t>(out.channels * out.frames));
    for (int64_t c = 0; c < kMelChannels; ++c) {
        std::copy_n(
            x.data() + static_cast<std::ptrdiff_t>(c * total_frames + reference_frames),
            static_cast<size_t>(out.frames),
            out.values.data() + static_cast<std::ptrdiff_t>(c * out.frames));
    }
    debug::timing_log_scalar("index_tts2.s2mel.cfm.euler_graph_ms", graph_ms);
    debug::timing_log_scalar("index_tts2.s2mel.cfm.euler_total_ms", engine::debug::elapsed_ms(cfm_start));
    return out;
}

}  // namespace engine::models::index_tts2
