#include "engine/models/confucius4_tts/s2a.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/lookup_modules.h"
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

namespace engine::models::confucius4_tts {
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
        throw std::runtime_error("Confucius S2Mel length regulator group norm requires affine weights");
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
    const ConfuciusS2AAdaLayerNormWeights & weights) {
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
    const ConfuciusS2ADitLayerWeights & weights) {
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
    const ConfuciusS2ADitLayerWeights & weights) {
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
    const ConfuciusS2ADitLayerWeights & weights,
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
    const ConfuciusS2ACfmWeights & weights) {
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
    const ConfuciusS2ACfmWeights & weights) {
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
    const ConfuciusS2ACfmWeights & weights) {
    const int64_t batch = x_bct.shape.dims[0];
    const int64_t frames = x_bct.shape.dims[2];
    auto t1 = timestep_embedding(ctx, timestep_b, weights.time_freqs, weights.time_mlp0, weights.time_mlp2);
    auto cond = modules::LinearModule({kHidden, kHidden, true, GGML_PREC_F32}).build(ctx, cond_btc, weights.input_mu_projection);
    auto x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x_bct);
    auto prompt = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, prompt_bct);
    auto style = broadcast_batch_time(ctx, style_bc, batch, frames, kStyleDim);
    auto hidden = modules::ConcatModule({2}).build(ctx, x, prompt);
    hidden = modules::ConcatModule({2}).build(ctx, hidden, cond);
    hidden = modules::ConcatModule({2}).build(ctx, hidden, style);
    hidden = modules::LinearModule({kHidden + 2 * kMelChannels + kStyleDim, kHidden, true, GGML_PREC_F32})
                 .build(ctx, hidden, weights.input_projection);

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

ConfuciusS2AAdaLayerNormWeights load_ada_norm(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type) {
    return {
        store.load_f32_tensor(source, prefix + ".norm.weight", {kHidden}),
        binding::linear_from_source(store, source, prefix + ".modulation", storage_type, 2 * kHidden, kHidden, true),
    };
}

ConfuciusS2ADitLayerWeights load_dit_layer(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    int64_t layer_index,
    engine::assets::TensorStorageType storage_type) {
    const std::string prefix = "decoder.estimator.transformer_blocks." + std::to_string(layer_index);
    ConfuciusS2ADitLayerWeights layer;
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

ConfuciusS2ALengthRegulatorWeights load_length_regulator(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    ConfuciusS2ALengthRegulatorWeights weights;
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
        weights.norms.push_back(binding::norm_from_source(
            store,
            source,
            "length_regulator.model." + std::to_string(i),
            kHidden));
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

ConfuciusS2ACfmWeights load_cfm(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    ConfuciusS2ACfmWeights weights;
    weights.input_mu_projection = binding::linear_from_source(
        store,
        source,
        "decoder.estimator.input_embed.mu_projection",
        matmul_storage_type,
        kHidden,
        kHidden,
        true);
    weights.input_projection = binding::linear_from_source(
        store,
        source,
        "decoder.estimator.input_embed.proj",
        matmul_storage_type,
        kHidden,
        kHidden + 2 * kMelChannels + kStyleDim,
        true);
    weights.skip_linear = binding::linear_from_source(
        store,
        source,
        "decoder.estimator.skip_linear",
        matmul_storage_type,
        kHidden,
        kHidden + kMelChannels,
        true);
    std::vector<float> freqs(static_cast<size_t>(kTimeFreqDim));
    for (int64_t i = 0; i < kTimeFreqDim; ++i) {
        freqs[static_cast<size_t>(i)] = std::exp(-std::log(10000.0F) * static_cast<float>(i) / static_cast<float>(kTimeFreqDim));
    }
    weights.time_freqs = store.make_f32(core::TensorShape::from_dims({kTimeFreqDim}), freqs);
    weights.time_mlp0 = binding::linear_from_source(store, source, "decoder.estimator.t_embedder.time_mlp.0", matmul_storage_type, kHidden, kTimeEmbeddingDim, true);
    weights.time_mlp2 = binding::linear_from_source(store, source, "decoder.estimator.t_embedder.time_mlp.2", matmul_storage_type, kHidden, kHidden, true);
    weights.time2_freqs = store.make_f32(core::TensorShape::from_dims({kTimeFreqDim}), std::move(freqs));
    weights.time2_mlp0 = binding::linear_from_source(store, source, "decoder.estimator.t_embedder2.time_mlp.0", matmul_storage_type, kHidden, kTimeEmbeddingDim, true);
    weights.time2_mlp2 = binding::linear_from_source(store, source, "decoder.estimator.t_embedder2.time_mlp.2", matmul_storage_type, kHidden, kHidden, true);
    weights.dit_layers.reserve(static_cast<size_t>(kDitLayers));
    for (int64_t i = 0; i < kDitLayers; ++i) {
        weights.dit_layers.push_back(load_dit_layer(store, source, i, matmul_storage_type));
    }
    weights.dit_norm = load_ada_norm(store, source, "decoder.estimator.transformer_norm", matmul_storage_type);
    weights.conv1 = binding::linear_from_source(store, source, "decoder.estimator.conv1", matmul_storage_type, kHidden, kHidden, true);
    weights.res_projection = binding::linear_from_source(store, source, "decoder.estimator.res_projection", matmul_storage_type, kHidden, kHidden, true);
    weights.wavenet_cond = load_weight_norm_conv1d(
        store,
        source,
        "decoder.estimator.wavenet.cond_layer.conv",
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
                "decoder.estimator.wavenet.in_layers." + std::to_string(i) + ".conv",
                conv_storage_type,
                2 * kHidden,
                kHidden,
                kWavenetKernel),
            load_weight_norm_conv1d(
                store,
                source,
                "decoder.estimator.wavenet.res_skip_layers." + std::to_string(i) + ".conv",
                conv_storage_type,
                res_skip_channels,
                kHidden,
                1),
        });
    }
    weights.final_modulation = binding::linear_from_source(
        store,
        source,
        "decoder.estimator.final_layer.adaLN_modulation.1",
        matmul_storage_type,
        2 * kHidden,
        kHidden,
        true);
    weights.final_linear = binding::linear_from_source(
        store,
        source,
        "decoder.estimator.final_layer.linear",
        matmul_storage_type,
        kHidden,
        kHidden,
        true);
    weights.conv2 = binding::conv1d_from_source(store, source, "decoder.estimator.conv2", conv_storage_type, kMelChannels, kHidden, 1, true);
    return weights;
}

}  // namespace

std::shared_ptr<const ConfuciusS2AWeights> load_confucius_s2a_weights(
    const ConfuciusAssets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes) {
    if (assets.s2a_weights == nullptr) {
        throw std::runtime_error("Confucius4-TTS S2A requires tensor source");
    }
    auto weights = std::make_shared<ConfuciusS2AWeights>();
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        backend,
        backend_type,
        "confucius4_tts.s2a.weights",
        weight_context_bytes);

    const auto & source = *assets.s2a_weights;
    weights->input_embedding.token_embedding = weights->store->load_tensor(
        source,
        "input_embedding.embedding.weight",
        matmul_storage_type,
        {8192, 8});
    weights->input_embedding.token_projection = binding::conv1d_from_source(
        *weights->store,
        source,
        "input_embedding.out_project",
        conv_storage_type,
        kContentDim,
        8,
        1,
        true);
    weights->input_embedding.encoder_projection = binding::linear_from_source(
        *weights->store,
        source,
        "encoder_proj",
        matmul_storage_type,
        kContentDim,
        kGptDim + kContentDim,
        true);
    weights->input_embedding.prompt_cond = weights->store->load_f32_tensor(source, "prompt_cond", {1, 1, kHidden});
    weights->length_regulator = load_length_regulator(*weights->store, source, matmul_storage_type, conv_storage_type);
    weights->cfm = load_cfm(*weights->store, source, matmul_storage_type, conv_storage_type);

    weights->store->upload();
    assets.s2a_weights->release_storage();
    return weights;
}

class ConfuciusS2ARuntime::ConditionGraph {
public:
    ConditionGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const ConfuciusS2AWeights> weights,
        int64_t input_frames,
        int64_t output_frames,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          input_frames_(input_frames),
          output_frames_(output_frames) {
        if (input_frames_ <= 0 || output_frames_ <= 0) {
            throw std::runtime_error("Confucius4-TTS S2A condition graph requires positive frame counts");
        }
        if (weights_ == nullptr) {
            throw std::runtime_error("Confucius4-TTS S2A condition graph requires weights");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Confucius4-TTS S2A condition graph context");
        }
        ggml_init_params input_params{16ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Confucius4-TTS S2A condition input context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "confucius4_tts.s2a.condition", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{
            input_ctx_.get(),
            "confucius4_tts.s2a.condition.inputs",
            execution_.backend_type()};
        semantic_ids_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({input_frames_})).tensor;
        latent_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, input_frames_, kGptDim})).tensor;
        mask_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, output_frames_, kHidden})).tensor;
        ggml_set_input(semantic_ids_);
        ggml_set_input(latent_);
        ggml_set_input(mask_);
        auto semantic_ids = core::wrap_tensor(semantic_ids_, core::TensorShape::from_dims({input_frames_}), GGML_TYPE_I32);
        auto semantic = modules::EmbeddingModule({8192, 8}).build(ctx, semantic_ids, weights_->input_embedding.token_embedding);
        semantic = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, semantic), core::TensorShape::from_dims({1, input_frames_, 8}));
        semantic = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, semantic);
        semantic = modules::Conv1dModule({8, kContentDim, 1, 1, 0, 1, true}).build(ctx, semantic, weights_->input_embedding.token_projection);
        semantic = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, semantic);
        auto latent = core::wrap_tensor(latent_, core::TensorShape::from_dims({1, input_frames_, kGptDim}), GGML_TYPE_F32);
        auto x = modules::ConcatModule({2}).build(ctx, latent, semantic);
        x = modules::LinearModule({kGptDim + kContentDim, kContentDim, true, GGML_PREC_F32})
                .build(ctx, x, weights_->input_embedding.encoder_projection);
        x = modules::LinearModule({kContentDim, kHidden, true, GGML_PREC_F32})
                .build(ctx, x, weights_->length_regulator.content_projection);
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        x = modules::Interpolate1dModule({output_frames_, modules::Interpolate1dMode::Nearest}).build(ctx, x);
        for (size_t layer = 0; layer < weights_->length_regulator.convs.size(); ++layer) {
            x = modules::Conv1dModule({kHidden, kHidden, 3, 1, 1, 1, true}).build(ctx, x, weights_->length_regulator.convs[layer]);
            x = group_norm_1_group(ctx, x, weights_->length_regulator.norms[layer], kHidden);
            x = mish(ctx, x);
        }
        x = modules::Conv1dModule({kHidden, kHidden, 1, 1, 0, 1, true}).build(ctx, x, weights_->length_regulator.output);
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        auto mask = core::wrap_tensor(mask_, core::TensorShape::from_dims({1, output_frames_, kHidden}), GGML_TYPE_F32);
        x = modules::MulModule{}.build(ctx, x, mask);
        output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        debug::trace_log_scalar("confucius4_tts.s2a.condition.graph_nodes", ggml_graph_n_nodes(graph_));
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Confucius4-TTS S2A condition input buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate Confucius4-TTS S2A condition graph");
        }
        mask_values_.assign(static_cast<size_t>(output_frames_ * kHidden), 1.0F);
        ggml_backend_tensor_set(mask_, mask_values_.data(), 0, mask_values_.size() * sizeof(float));
        debug::timing_log_scalar("confucius4_tts.s2a.condition.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("confucius4_tts.s2a.condition.input_frames", input_frames_);
        debug::trace_log_scalar("confucius4_tts.s2a.condition.output_frames", output_frames_);
    }

    ~ConditionGraph() {
        clear_graph();
    }

    bool matches(int64_t input_frames, int64_t output_frames) const noexcept {
        return input_frames_ == input_frames && output_frames_ == output_frames;
    }

    ConfuciusS2ASequence run(const std::vector<int32_t> & semantic_tokens, const std::vector<float> & latent) {
        if (static_cast<int64_t>(semantic_tokens.size()) != input_frames_ ||
            static_cast<int64_t>(latent.size()) != input_frames_ * kGptDim) {
            throw std::runtime_error("Confucius4-TTS S2A condition input shape mismatch");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(semantic_ids_, semantic_tokens.data(), 0, semantic_tokens.size() * sizeof(int32_t));
        ggml_backend_tensor_set(latent_, latent.data(), 0, latent.size() * sizeof(float));
        debug::timing_log_scalar("confucius4_tts.s2a.condition.input_upload_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        timing_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        debug::timing_log_scalar("confucius4_tts.s2a.condition.graph.compute_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Confucius4-TTS S2A condition graph compute failed");
        }
        ConfuciusS2ASequence output;
        output.frames = output_frames_;
        output.dims = kHidden;
        timing_start = Clock::now();
        output.values = core::read_tensor_f32(output_);
        debug::timing_log_scalar("confucius4_tts.s2a.condition.output_read_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
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
    std::shared_ptr<const ConfuciusS2AWeights> weights_;
    int64_t input_frames_ = 0;
    int64_t output_frames_ = 0;
    std::vector<float> mask_values_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * semantic_ids_ = nullptr;
    ggml_tensor * latent_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
};

class ConfuciusS2ARuntime::CfmGraph {
public:
    struct Output {
        std::vector<float> values;
        double input_upload_ms = 0.0;
        double graph_compute_ms = 0.0;
        double output_read_ms = 0.0;
    };

    CfmGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const ConfuciusS2AWeights> weights,
        int64_t frames,
        bool use_cfg,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          frames_(frames),
          use_cfg_(use_cfg),
          batch_(use_cfg ? 2 : 1) {
        if (frames_ <= 0) {
            throw std::runtime_error("Confucius S2Mel CFM graph requires positive frame count");
        }
        if (weights_ == nullptr) {
            throw std::runtime_error("Confucius S2Mel CFM graph requires weights");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Confucius S2Mel CFM graph context");
        }
        ggml_init_params input_params{16ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Confucius S2Mel CFM input context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "confucius4_tts.s2a.cfm", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "confucius4_tts.s2a.cfm.inputs", execution_.backend_type()};
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
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        ggml_build_forward_expand(graph_, output_);
        debug::trace_log_scalar("confucius4_tts.s2a.cfm.graph_nodes", ggml_graph_n_nodes(graph_));
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Confucius S2Mel CFM input buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate Confucius S2Mel CFM graph");
        }
        positions_values_.assign(static_cast<size_t>(frames_), 0);
        for (int64_t i = 0; i < frames_; ++i) {
            positions_values_[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        ggml_backend_tensor_set(positions_, positions_values_.data(), 0, positions_values_.size() * sizeof(int32_t));
        debug::timing_log_scalar("confucius4_tts.s2a.cfm.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("confucius4_tts.s2a.cfm.frames", frames_);
        debug::trace_log_scalar("confucius4_tts.s2a.cfm.batch", batch_);
    }

    ~CfmGraph() {
        clear_graph();
    }

    bool matches(int64_t frames, bool use_cfg) const noexcept {
        return frames_ == frames && use_cfg_ == use_cfg;
    }

    double upload_static_inputs(
        const std::vector<float> & prompt,
        const std::vector<float> & cond,
        const std::vector<float> & style) {
        const int64_t mel_values = batch_ * kMelChannels * frames_;
        if (static_cast<int64_t>(prompt.size()) != mel_values ||
            static_cast<int64_t>(cond.size()) != batch_ * frames_ * kHidden ||
            static_cast<int64_t>(style.size()) != batch_ * kStyleDim) {
            throw std::runtime_error("Confucius S2Mel CFM static input shape mismatch");
        }
        const auto timing_start = Clock::now();
        ggml_backend_tensor_set(prompt_, prompt.data(), 0, prompt.size() * sizeof(float));
        ggml_backend_tensor_set(cond_, cond.data(), 0, cond.size() * sizeof(float));
        ggml_backend_tensor_set(style_, style.data(), 0, style.size() * sizeof(float));
        return engine::debug::elapsed_ms(timing_start, Clock::now());
    }

    Output run(
        const std::vector<float> & x,
        const std::vector<float> & timestep) {
        const int64_t mel_values = batch_ * kMelChannels * frames_;
        if (static_cast<int64_t>(x.size()) != mel_values || static_cast<int64_t>(timestep.size()) != batch_) {
            throw std::runtime_error("Confucius S2Mel CFM graph input shape mismatch");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(x_, x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(timestep_, timestep.data(), 0, timestep.size() * sizeof(float));
        Output output;
        output.input_upload_ms = engine::debug::elapsed_ms(timing_start, Clock::now());
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        timing_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        output.graph_compute_ms = engine::debug::elapsed_ms(timing_start, Clock::now());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Confucius S2Mel CFM graph compute failed");
        }
        output.values.assign(static_cast<size_t>(mel_values), 0.0F);
        timing_start = Clock::now();
        ggml_backend_tensor_get(output_, output.values.data(), 0, output.values.size() * sizeof(float));
        output.output_read_ms = engine::debug::elapsed_ms(timing_start, Clock::now());
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
    std::shared_ptr<const ConfuciusS2AWeights> weights_;
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
        throw std::runtime_error("Confucius S2Mel CFM prompt frame count is out of range");
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
        throw std::runtime_error("Confucius S2Mel CFM reference mel shape mismatch");
    }
    for (int64_t c = 0; c < channels; ++c) {
        std::copy_n(
            prompt.data() + static_cast<std::ptrdiff_t>(c * prompt_frames),
            static_cast<size_t>(prompt_frames),
            out.data() + static_cast<std::ptrdiff_t>(c * frames));
    }
    return out;
}

ConfuciusS2ARuntime::ConfuciusS2ARuntime(
    std::shared_ptr<const ConfuciusAssets> assets,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type)
    : assets_(std::move(assets)),
      execution_(&execution),
      graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Confucius S2Mel runtime requires assets");
    }
    if (graph_arena_bytes_ == 0) {
        throw std::runtime_error("Confucius S2Mel graph arena must be non-zero");
    }
    weights_ = load_confucius_s2a_weights(
        *assets_,
        execution.backend(),
        execution.backend_type(),
        matmul_storage_type,
        conv_storage_type,
        weight_context_bytes);
}

ConfuciusS2ARuntime::~ConfuciusS2ARuntime() = default;

void ConfuciusS2ARuntime::prepare_condition(int64_t input_frames, int64_t output_frames) {
    if (execution_ == nullptr) {
        throw std::runtime_error("Confucius S2Mel runtime execution context is missing");
    }
    if (condition_graph_ != nullptr && condition_graph_->matches(input_frames, output_frames)) {
        return;
    }
    condition_graph_.reset();
    condition_graph_ = std::make_unique<ConditionGraph>(
        *execution_,
        weights_,
        input_frames,
        output_frames,
        graph_arena_bytes_);
}

void ConfuciusS2ARuntime::prepare_cfm(int64_t total_frames, bool use_cfg) {
    if (execution_ == nullptr) {
        throw std::runtime_error("Confucius S2Mel runtime execution context is missing");
    }
    if (total_frames <= 0) {
        throw std::runtime_error("Confucius S2Mel CFM prepare requires positive frame count");
    }
    if (cfm_graph_ != nullptr && cfm_graph_->matches(total_frames, use_cfg)) {
        return;
    }
    cfm_graph_.reset();
    cfm_graph_ = std::make_unique<CfmGraph>(*execution_, weights_, total_frames, use_cfg, graph_arena_bytes_);
}

void ConfuciusS2ARuntime::release_pre_cfm_graphs() {
    condition_graph_.reset();
}

void ConfuciusS2ARuntime::release_cfm_graph() {
    cfm_graph_.reset();
}

ConfuciusS2AMel ConfuciusS2ARuntime::infer_mel(
    const ConfuciusT2SSemanticGeneration & generation,
    const ConfuciusMelOutput & reference_mel,
    const ConfuciusStyleEmbedding & style,
    int64_t target_frames,
    int64_t num_inference_steps,
    float guidance_scale,
    uint32_t seed,
    uint64_t & rng_offset_blocks) {
    if (generation.semantic_codes.empty() || generation.latent.empty() ||
        reference_mel.values.empty() || style.values.empty()) {
        throw std::runtime_error("Confucius S2Mel CFM requires non-empty condition, reference mel, and style");
    }
    prepare_condition(static_cast<int64_t>(generation.semantic_codes.size()), target_frames);
    const auto condition = condition_graph_->run(generation.semantic_codes, generation.latent);
    const int64_t reference_frames = reference_mel.frames;
    const int64_t total_frames = reference_frames + target_frames;
    if (total_frames <= 0 || reference_frames <= 0 || reference_frames > total_frames) {
        throw std::runtime_error("Confucius S2Mel CFM frame counts are invalid");
    }
    if (num_inference_steps <= 0) {
        throw std::runtime_error("Confucius S2Mel CFM num_inference_steps must be positive");
    }
    if (static_cast<int64_t>(style.values.size()) != kStyleDim) {
        throw std::runtime_error("Confucius S2Mel CFM style shape mismatch");
    }
    const bool use_cfg = guidance_scale > 0.0F;
    auto timing_start = Clock::now();
    prepare_cfm(total_frames, use_cfg);
    debug::timing_log_scalar("confucius4_tts.s2a.cfm.prepare_ms", engine::debug::elapsed_ms(timing_start));
    if (cfm_graph_ == nullptr || !cfm_graph_->matches(total_frames, use_cfg)) {
        throw std::runtime_error("Confucius S2Mel CFM graph was not prepared for this shape");
    }
    const auto rng_policy = engine::sampling::resolve_torch_cuda_sampling_policy(
        execution_->backend_type(),
        execution_->config().device,
        "confucius4_tts.s2a.cuda_sampling_policy",
        "Confucius",
        engine::sampling::TorchCudaSamplingPolicyFailureMode::StrictCuda);
    const uint64_t noise_elements = static_cast<uint64_t>(kMelChannels * total_frames);
    timing_start = Clock::now();
    std::vector<float> x = engine::sampling::generate_torch_cuda_tensor_iterator_randn(
        static_cast<size_t>(noise_elements),
        seed,
        rng_offset_blocks,
        rng_policy,
        engine::sampling::TorchRandnPrecision::Float32);
    rng_offset_blocks += engine::sampling::torch_cuda_tensor_iterator_offset_blocks(noise_elements, rng_policy);
    debug::timing_log_scalar("confucius4_tts.s2a.noise_ms", engine::debug::elapsed_ms(timing_start));
    timing_start = Clock::now();
    auto prompt_x = make_prompt_x(reference_mel.values, kMelChannels, total_frames, reference_frames);
    zero_prompt_region(x, kMelChannels, total_frames, reference_frames);
    std::vector<float> mu(static_cast<size_t>(total_frames * kHidden), 0.0F);
    const auto & prompt = weights_->input_embedding.prompt_cond;
    std::vector<float> prompt_cond = core::read_tensor_f32(prompt.tensor);
    for (int64_t t = 0; t < reference_frames; ++t) {
        std::copy_n(
            prompt_cond.data(),
            static_cast<size_t>(kHidden),
            mu.data() + static_cast<std::ptrdiff_t>(t * kHidden));
    }
    std::copy_n(
        condition.values.data(),
        static_cast<size_t>(condition.frames * condition.dims),
        mu.data() + static_cast<std::ptrdiff_t>(reference_frames * kHidden));
    const auto prompt_batched = repeat_or_zero_rows(prompt_x, kMelChannels * total_frames, use_cfg, true);
    const auto cond_batched = repeat_or_zero_rows(mu, total_frames * kHidden, use_cfg, true);
    const auto style_batched = repeat_or_zero_rows(style.values, kStyleDim, use_cfg, true);
    debug::timing_log_scalar("confucius4_tts.s2a.cfm.host_static_inputs_ms", engine::debug::elapsed_ms(timing_start));
    debug::timing_log_scalar(
        "confucius4_tts.s2a.cfm.static_input_upload_ms",
        cfm_graph_->upload_static_inputs(prompt_batched, cond_batched, style_batched));

    const auto cfm_start = Clock::now();
    double graph_ms = 0.0;
    double input_upload_ms = 0.0;
    double graph_compute_ms = 0.0;
    double output_read_ms = 0.0;
    double x_batch_ms = 0.0;
    double update_ms = 0.0;
    float t = 0.0F;
    float dt = 1.0F / static_cast<float>(num_inference_steps);
    std::vector<float> x_batched(static_cast<size_t>((use_cfg ? 2 : 1) * kMelChannels * total_frames), 0.0F);
    for (int64_t step = 1; step <= num_inference_steps; ++step) {
        timing_start = Clock::now();
        if (use_cfg) {
            const size_t row_values = static_cast<size_t>(kMelChannels * total_frames);
            std::copy_n(x.data(), row_values, x_batched.data());
            std::copy_n(x.data(), row_values, x_batched.data() + static_cast<std::ptrdiff_t>(row_values));
        } else {
            std::copy(x.begin(), x.end(), x_batched.begin());
        }
        x_batch_ms += engine::debug::elapsed_ms(timing_start);
        const auto graph_start = Clock::now();
        std::vector<float> timestep(static_cast<size_t>(use_cfg ? 2 : 1), t);
        const auto velocity = cfm_graph_->run(x_batched, timestep);
        graph_ms += engine::debug::elapsed_ms(graph_start);
        input_upload_ms += velocity.input_upload_ms;
        graph_compute_ms += velocity.graph_compute_ms;
        output_read_ms += velocity.output_read_ms;
        timing_start = Clock::now();
        const int64_t row_values = kMelChannels * total_frames;
        for (int64_t i = 0; i < row_values; ++i) {
            float dphi = velocity.values[static_cast<size_t>(i)];
            if (use_cfg) {
                dphi = (1.0F + guidance_scale) * dphi - guidance_scale * velocity.values[static_cast<size_t>(row_values + i)];
            }
            x[static_cast<size_t>(i)] += dt * dphi;
        }
        t += dt;
        if (step < num_inference_steps) {
            dt = (static_cast<float>(step + 1) / static_cast<float>(num_inference_steps)) - t;
        }
        zero_prompt_region(x, kMelChannels, total_frames, reference_frames);
        update_ms += engine::debug::elapsed_ms(timing_start);
    }

    ConfuciusS2AMel out;
    out.frames = total_frames - reference_frames;
    out.channels = kMelChannels;
    out.values.resize(static_cast<size_t>(out.channels * out.frames));
    for (int64_t c = 0; c < kMelChannels; ++c) {
        std::copy_n(
            x.data() + static_cast<std::ptrdiff_t>(c * total_frames + reference_frames),
            static_cast<size_t>(out.frames),
            out.values.data() + static_cast<std::ptrdiff_t>(c * out.frames));
    }
    debug::timing_log_scalar("confucius4_tts.s2a.cfm.euler_graph_ms", graph_ms);
    debug::timing_log_scalar("confucius4_tts.s2a.cfm.input_upload_ms", input_upload_ms);
    debug::timing_log_scalar("confucius4_tts.s2a.cfm.graph.compute_ms", graph_compute_ms);
    debug::timing_log_scalar("confucius4_tts.s2a.cfm.output_read_ms", output_read_ms);
    debug::timing_log_scalar("confucius4_tts.s2a.cfm.euler_x_batch_ms", x_batch_ms);
    debug::timing_log_scalar("confucius4_tts.s2a.cfm.euler_update_ms", update_ms);
    debug::timing_log_scalar("confucius4_tts.s2a.cfm.euler_total_ms", engine::debug::elapsed_ms(cfm_start));
    return out;
}

}  // namespace engine::models::confucius4_tts
