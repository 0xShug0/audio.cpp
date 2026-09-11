#include "engine/models/seed_vc/v1_cfm.h"

#include "engine/models/seed_vc/assets.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::seed_vc {
namespace {

using engine::core::TensorShape;
using engine::core::TensorValue;

TensorValue require_tensor(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & name,
    engine::assets::TensorStorageType storage_type) {
    return engine::modules::binding::tensor_from_named_source(
        store,
        source,
        name,
        seed_vc_component_storage_type(source, name, storage_type));
}

engine::modules::LinearWeights linear_weights(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type) {
    return engine::modules::LinearWeights{
        require_tensor(store, source, prefix + ".weight", storage_type),
        require_tensor(store, source, prefix + ".bias", storage_type)};
}

engine::modules::LinearWeights linear_weights_no_bias(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type) {
    return engine::modules::LinearWeights{
        require_tensor(store, source, prefix + ".weight", storage_type),
        std::nullopt};
}

engine::modules::NormWeights rms_weight(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & name,
    engine::assets::TensorStorageType storage_type) {
    return engine::modules::NormWeights{require_tensor(store, source, name, storage_type), std::nullopt};
}

TensorValue contiguous(engine::core::ModuleBuildContext & ctx, const TensorValue & value) {
    return engine::core::ensure_backend_addressable_layout(ctx, value);
}

TensorValue add(engine::core::ModuleBuildContext & ctx, const TensorValue & lhs, const TensorValue & rhs) {
    return engine::modules::AddModule{}.build(ctx, lhs, rhs);
}

TensorValue mul(engine::core::ModuleBuildContext & ctx, const TensorValue & lhs, const TensorValue & rhs) {
    return engine::modules::MulModule{}.build(ctx, lhs, rhs);
}

TensorValue slice_last(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    int64_t start,
    int64_t length) {
    return engine::modules::SliceModule({static_cast<int>(input.shape.rank - 1), start, length}).build(ctx, input);
}

TensorValue slice_tokens(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    int64_t start,
    int64_t length) {
    return engine::modules::SliceModule({1, start, length}).build(ctx, input);
}

TensorValue slice_channels(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    int64_t start,
    int64_t length) {
    return engine::modules::SliceModule({1, start, length}).build(ctx, input);
}

TensorValue expand_batch_token(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & value,
    int64_t batch,
    int64_t tokens,
    int64_t channels) {
    auto reshaped = engine::core::reshape_tensor(ctx, contiguous(ctx, value), TensorShape::from_dims({batch, 1, channels}));
    return engine::modules::RepeatModule({TensorShape::from_dims({batch, tokens, channels})}).build(ctx, reshaped);
}

TensorValue timestep_frequency_embedding(
    engine::core::ModuleBuildContext & ctx,
    ggml_tensor * freqs_tensor,
    const TensorValue & timestep) {
    const int64_t batch = timestep.shape.dims[0];
    auto freqs = engine::core::wrap_tensor(freqs_tensor, TensorShape::from_dims({128}), GGML_TYPE_F32);
    auto freqs_b = engine::core::reshape_tensor(ctx, freqs, TensorShape::from_dims({1, 128}));
    freqs_b = engine::modules::RepeatModule({TensorShape::from_dims({batch, 128})}).build(ctx, freqs_b);
    auto t = engine::core::reshape_tensor(ctx, timestep, TensorShape::from_dims({batch, 1}));
    t = engine::modules::RepeatModule({TensorShape::from_dims({batch, 128})}).build(ctx, t);
    auto args = mul(ctx, engine::core::wrap_tensor(ggml_scale(ctx.ggml, t.tensor, 1000.0F), t.shape, GGML_TYPE_F32), freqs_b);
    auto cos_part = engine::core::wrap_tensor(ggml_cos(ctx.ggml, args.tensor), args.shape, GGML_TYPE_F32);
    auto sin_part = engine::core::wrap_tensor(ggml_sin(ctx.ggml, args.tensor), args.shape, GGML_TYPE_F32);
    return engine::modules::ConcatModule({1}).build(ctx, cos_part, sin_part);
}

std::vector<float> make_timestep_freqs() {
    std::vector<float> out(128, 0.0F);
    for (int64_t i = 0; i < 128; ++i) {
        out[static_cast<size_t>(i)] =
            std::exp(-std::log(10000.0F) * static_cast<float>(i) / 128.0F);
    }
    return out;
}

struct V1LayerWeights {
    engine::modules::LinearWeights attention_wqkv;
    engine::modules::LinearWeights attention_wo;
    engine::modules::LinearWeights attention_norm_project;
    engine::modules::NormWeights attention_norm;
    engine::modules::LinearWeights ffn_norm_project;
    engine::modules::NormWeights ffn_norm;
    engine::modules::LinearWeights ff_w1;
    engine::modules::LinearWeights ff_w2;
    engine::modules::LinearWeights ff_w3;
    engine::modules::LinearWeights skip_in_linear;
};

struct V1CfmWeights {
    struct WavenetWeights {
        engine::modules::LinearWeights t_embedder2_0;
        engine::modules::LinearWeights t_embedder2_2;
        engine::modules::LinearWeights conv1;
        engine::modules::Conv1dWeights conv2;
        engine::modules::LinearWeights res_projection;
        engine::modules::LinearWeights final_modulation;
        engine::modules::LinearWeights final_linear;
        engine::modules::Conv1dWeights cond_layer;
        std::vector<engine::modules::Conv1dWeights> in_layers;
        std::vector<engine::modules::Conv1dWeights> res_skip_layers;
    };

    engine::modules::LinearWeights cond_projection;
    engine::modules::LinearWeights cond_x_merge_linear;
    engine::modules::LinearWeights t_embedder_0;
    engine::modules::LinearWeights t_embedder_2;
    std::optional<engine::modules::LinearWeights> style_in;
    std::optional<engine::modules::LinearWeights> final_mlp_0;
    std::optional<engine::modules::LinearWeights> final_mlp_2;
    engine::modules::LinearWeights final_norm_project;
    engine::modules::NormWeights final_norm;
    std::optional<engine::modules::LinearWeights> long_skip_linear;
    std::optional<WavenetWeights> wavenet;
    std::vector<V1LayerWeights> layers;
};

engine::modules::Conv1dWeights conv1d_weights(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type) {
    return engine::modules::Conv1dWeights{
        require_tensor(store, source, prefix + ".weight", storage_type),
        require_tensor(store, source, prefix + ".bias", storage_type)};
}

V1CfmWeights load_v1_cfm_weights(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    int64_t layers,
    engine::assets::TensorStorageType storage_type) {
    const std::string root = "cfm.estimator.";
    V1CfmWeights out;
    out.cond_projection = linear_weights(store, source, root + "cond_projection", storage_type);
    out.cond_x_merge_linear = linear_weights(store, source, root + "cond_x_merge_linear", storage_type);
    out.t_embedder_0 = linear_weights(store, source, root + "t_embedder.mlp.0", storage_type);
    out.t_embedder_2 = linear_weights(store, source, root + "t_embedder.mlp.2", storage_type);
    if (source.has_tensor(root + "style_in.weight")) {
        out.style_in = linear_weights(store, source, root + "style_in", storage_type);
    }
    if (source.has_tensor(root + "skip_linear.weight")) {
        out.long_skip_linear = linear_weights(store, source, root + "skip_linear", storage_type);
    }
    if (source.has_tensor(root + "final_mlp.0.weight")) {
        out.final_mlp_0 = linear_weights(store, source, root + "final_mlp.0", storage_type);
        out.final_mlp_2 = linear_weights(store, source, root + "final_mlp.2", storage_type);
    }
    if (source.has_tensor(root + "wavenet.cond_layer.conv.conv.weight")) {
        V1CfmWeights::WavenetWeights wavenet;
        wavenet.t_embedder2_0 = linear_weights(store, source, root + "t_embedder2.mlp.0", storage_type);
        wavenet.t_embedder2_2 = linear_weights(store, source, root + "t_embedder2.mlp.2", storage_type);
        wavenet.conv1 = linear_weights(store, source, root + "conv1", storage_type);
        wavenet.conv2 = conv1d_weights(store, source, root + "conv2", storage_type);
        wavenet.res_projection = linear_weights(store, source, root + "res_projection", storage_type);
        wavenet.final_modulation = linear_weights(store, source, root + "final_layer.adaLN_modulation.1", storage_type);
        wavenet.final_linear = linear_weights(store, source, root + "final_layer.linear", storage_type);
        wavenet.cond_layer = conv1d_weights(store, source, root + "wavenet.cond_layer.conv.conv", storage_type);
        for (int64_t layer = 0; layer < 8; ++layer) {
            wavenet.in_layers.push_back(
                conv1d_weights(store, source, root + "wavenet.in_layers." + std::to_string(layer) + ".conv.conv", storage_type));
            wavenet.res_skip_layers.push_back(
                conv1d_weights(store, source, root + "wavenet.res_skip_layers." + std::to_string(layer) + ".conv.conv", storage_type));
        }
        out.wavenet = std::move(wavenet);
    }
    out.layers.reserve(static_cast<size_t>(layers));
    for (int64_t layer = 0; layer < layers; ++layer) {
        const std::string prefix = root + "transformer.layers." + std::to_string(layer);
        V1LayerWeights item;
        item.attention_wqkv = linear_weights_no_bias(store, source, prefix + ".attention.wqkv", storage_type);
        item.attention_wo = linear_weights_no_bias(store, source, prefix + ".attention.wo", storage_type);
        item.attention_norm_project = linear_weights(store, source, prefix + ".attention_norm.project_layer", storage_type);
        item.attention_norm = rms_weight(store, source, prefix + ".attention_norm.norm.weight", storage_type);
        item.ffn_norm_project = linear_weights(store, source, prefix + ".ffn_norm.project_layer", storage_type);
        item.ffn_norm = rms_weight(store, source, prefix + ".ffn_norm.norm.weight", storage_type);
        item.ff_w1 = linear_weights_no_bias(store, source, prefix + ".feed_forward.w1", storage_type);
        item.ff_w2 = linear_weights_no_bias(store, source, prefix + ".feed_forward.w2", storage_type);
        item.ff_w3 = linear_weights_no_bias(store, source, prefix + ".feed_forward.w3", storage_type);
        item.skip_in_linear = linear_weights(store, source, prefix + ".skip_in_linear", storage_type);
        out.layers.push_back(item);
    }
    out.final_norm_project = linear_weights(store, source, root + "transformer.norm.project_layer", storage_type);
    out.final_norm = rms_weight(store, source, root + "transformer.norm.norm.weight", storage_type);
    return out;
}

TensorValue build_timestep_embedding(
    engine::core::ModuleBuildContext & ctx,
    ggml_tensor * freqs_tensor,
    const TensorValue & timestep,
    const engine::modules::LinearWeights & first,
    const engine::modules::LinearWeights & second,
    int64_t hidden_dim) {
    auto hidden = timestep_frequency_embedding(ctx, freqs_tensor, timestep);
    hidden = engine::modules::LinearModule({256, hidden_dim, true, GGML_PREC_F32}).build(ctx, hidden, first);
    hidden = engine::modules::SiluModule{}.build(ctx, hidden);
    return engine::modules::LinearModule({hidden_dim, hidden_dim, true, GGML_PREC_F32})
        .build(ctx, hidden, second);
}

TensorValue build_adaptive_rms_norm(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const TensorValue & timestep_token,
    const engine::modules::LinearWeights & project,
    const engine::modules::NormWeights & norm,
    const SeedVcV1DitConfig & config) {
    const int64_t batch = input.shape.dims[0];
    const int64_t tokens = input.shape.dims[1];
    auto modulation = engine::modules::LinearModule({config.hidden_dim, config.hidden_dim * 2, true, GGML_PREC_F32})
                          .build(ctx, timestep_token, project);
    auto weight = expand_batch_token(ctx, slice_last(ctx, modulation, 0, config.hidden_dim), batch, tokens, config.hidden_dim);
    auto bias = expand_batch_token(ctx, slice_last(ctx, modulation, config.hidden_dim, config.hidden_dim), batch, tokens, config.hidden_dim);
    auto normalized = engine::modules::RMSNormModule({config.hidden_dim, 1.0e-5F, true, false})
                          .build(ctx, input, norm);
    return add(ctx, mul(ctx, weight, normalized), bias);
}

TensorValue build_plain_rms_norm(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const engine::modules::NormWeights & norm,
    const SeedVcV1DitConfig & config) {
    return engine::modules::RMSNormModule({config.hidden_dim, 1.0e-5F, true, false})
        .build(ctx, input, norm);
}

TensorValue build_attention(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const TensorValue & positions,
    const V1LayerWeights & weights,
    const SeedVcV1DitConfig & config) {
    const int64_t batch = input.shape.dims[0];
    const int64_t seq_len = input.shape.dims[1];
    const int64_t head_dim = config.hidden_dim / config.num_heads;
    auto qkv = engine::modules::LinearModule({config.hidden_dim, config.hidden_dim * 3, false, GGML_PREC_F32})
                   .build(ctx, input, weights.attention_wqkv);
    auto q = slice_last(ctx, qkv, 0, config.hidden_dim);
    auto k = slice_last(ctx, qkv, config.hidden_dim, config.hidden_dim);
    auto v = slice_last(ctx, qkv, config.hidden_dim * 2, config.hidden_dim);
    q = engine::core::reshape_tensor(ctx, contiguous(ctx, q), TensorShape::from_dims({batch, seq_len, config.num_heads, head_dim}));
    k = engine::core::reshape_tensor(ctx, contiguous(ctx, k), TensorShape::from_dims({batch, seq_len, config.num_heads, head_dim}));
    v = engine::core::reshape_tensor(ctx, contiguous(ctx, v), TensorShape::from_dims({batch, seq_len, config.num_heads, head_dim}));
    q = engine::modules::RoPEModule({head_dim, GGML_ROPE_TYPE_NORMAL, 10000.0F}).build(ctx, q, positions);
    k = engine::modules::RoPEModule({head_dim, GGML_ROPE_TYPE_NORMAL, 10000.0F}).build(ctx, k, positions);
    q = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    k = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
    v = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
    q = contiguous(ctx, q);
    k = contiguous(ctx, k);
    v = contiguous(ctx, v);
    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q.tensor,
        k.tensor,
        v.tensor,
        nullptr,
        1.0F / std::sqrt(static_cast<float>(head_dim)),
        0.0F,
        0.0F);
    ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
    auto context = engine::core::wrap_tensor(
        flash,
        TensorShape::from_dims({batch, seq_len, config.num_heads, head_dim}),
        GGML_TYPE_F32);
    context = engine::core::reshape_tensor(ctx, contiguous(ctx, context), TensorShape::from_dims({batch, seq_len, config.hidden_dim}));
    return engine::modules::LinearModule({config.hidden_dim, config.hidden_dim, false, GGML_PREC_F32})
        .build(ctx, context, weights.attention_wo);
}

TensorValue build_feed_forward(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const V1LayerWeights & weights,
    const SeedVcV1DitConfig & config) {
    const int64_t intermediate = weights.ff_w1.weight.shape.dims[0];
    auto gate = engine::modules::LinearModule({config.hidden_dim, intermediate, false, GGML_PREC_F32})
                    .build(ctx, input, weights.ff_w1);
    gate = engine::modules::SiluModule{}.build(ctx, gate);
    auto up = engine::modules::LinearModule({config.hidden_dim, intermediate, false, GGML_PREC_F32})
                  .build(ctx, input, weights.ff_w3);
    auto hidden = mul(ctx, gate, up);
    return engine::modules::LinearModule({intermediate, config.hidden_dim, false, GGML_PREC_F32})
        .build(ctx, hidden, weights.ff_w2);
}

TensorValue build_wavenet(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input_bct,
    const TensorValue & conditioning_bc,
    const V1CfmWeights::WavenetWeights & weights,
    const SeedVcV1WavenetConfig & config) {
    const int64_t batch = input_bct.shape.dims[0];
    const int64_t channels = input_bct.shape.dims[1];
    const int64_t frames = input_bct.shape.dims[2];
    auto g = engine::core::reshape_tensor(
        ctx,
        contiguous(ctx, conditioning_bc),
        TensorShape::from_dims({batch, channels, 1}));
    g = engine::modules::Conv1dModule({channels, 2 * channels * config.num_layers, 1, 1, 0, 1, true})
            .build(ctx, g, weights.cond_layer);

    auto x = input_bct;
    auto output = engine::core::wrap_tensor(
        ggml_scale(ctx.ggml, contiguous(ctx, input_bct).tensor, 0.0F),
        input_bct.shape,
        GGML_TYPE_F32);
    for (int64_t layer = 0; layer < config.num_layers; ++layer) {
        const int dilation = static_cast<int>(std::pow(static_cast<double>(config.dilation_rate), static_cast<double>(layer)));
        const int padding = static_cast<int>((config.kernel_size * dilation - dilation) / 2);
        auto x_padded = engine::modules::ReflectPad1dModule({padding, padding}).build(ctx, contiguous(ctx, x));
        auto x_in = engine::modules::Conv1dModule(
                        {channels, 2 * channels, config.kernel_size, 1, 0, dilation, true})
                        .build(ctx, x_padded, weights.in_layers[static_cast<size_t>(layer)]);
        auto g_l = slice_channels(ctx, g, layer * 2 * channels, 2 * channels);
        g_l = engine::modules::RepeatModule({TensorShape::from_dims({batch, 2 * channels, frames})}).build(ctx, g_l);
        const auto acts_in = add(ctx, x_in, g_l);
        const auto tanh_part = engine::modules::TanhModule{}.build(ctx, slice_channels(ctx, acts_in, 0, channels));
        const auto sigmoid_part = engine::modules::SigmoidModule{}.build(ctx, slice_channels(ctx, acts_in, channels, channels));
        const auto acts = mul(ctx, tanh_part, sigmoid_part);
        auto res_skip = engine::modules::Conv1dModule(
                            {channels,
                             layer < config.num_layers - 1 ? 2 * channels : channels,
                             1,
                             1,
                             0,
                             1,
                             true})
                            .build(ctx, acts, weights.res_skip_layers[static_cast<size_t>(layer)]);
        if (layer < config.num_layers - 1) {
            auto res = slice_channels(ctx, res_skip, 0, channels);
            auto skip = slice_channels(ctx, res_skip, channels, channels);
            x = add(ctx, x, res);
            output = add(ctx, output, skip);
        } else {
            output = add(ctx, output, res_skip);
        }
    }
    return output;
}

TensorValue build_wavenet_final_layer(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input_btc,
    const TensorValue & timestep_token,
    const V1CfmWeights::WavenetWeights & weights,
    int64_t hidden_dim) {
    auto modulation = engine::modules::SiluModule{}.build(ctx, timestep_token);
    modulation = engine::modules::LinearModule({hidden_dim, hidden_dim * 2, true, GGML_PREC_F32})
                     .build(ctx, modulation, weights.final_modulation);
    auto shift = slice_last(ctx, modulation, 0, hidden_dim);
    auto scale = slice_last(ctx, modulation, hidden_dim, hidden_dim);
    shift = expand_batch_token(ctx, shift, input_btc.shape.dims[0], input_btc.shape.dims[1], hidden_dim);
    scale = expand_batch_token(ctx, scale, input_btc.shape.dims[0], input_btc.shape.dims[1], hidden_dim);
    auto normalized = engine::modules::LayerNormModule({hidden_dim, 1.0e-6F, false, false})
                          .build(ctx, input_btc, engine::modules::NormWeights{std::nullopt, std::nullopt});
    normalized = add(ctx, add(ctx, normalized, mul(ctx, normalized, scale)), shift);
    return engine::modules::LinearModule({hidden_dim, hidden_dim, true, GGML_PREC_F32})
        .build(ctx, normalized, weights.final_linear);
}

TensorValue build_transformer_layer(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const TensorValue & timestep_token,
    const TensorValue & positions,
    const V1LayerWeights & weights,
    const SeedVcV1DitConfig & config) {
    auto attn_in = config.time_as_token
        ? build_plain_rms_norm(ctx, input, weights.attention_norm, config)
        : build_adaptive_rms_norm(
            ctx,
            input,
            timestep_token,
            weights.attention_norm_project,
            weights.attention_norm,
            config);
    auto hidden = add(ctx, input, build_attention(ctx, attn_in, positions, weights, config));
    auto ff_in = config.time_as_token
        ? build_plain_rms_norm(ctx, hidden, weights.ffn_norm, config)
        : build_adaptive_rms_norm(
            ctx,
            hidden,
            timestep_token,
            weights.ffn_norm_project,
            weights.ffn_norm,
            config);
    return add(ctx, hidden, build_feed_forward(ctx, ff_in, weights, config));
}

TensorValue build_estimator_graph(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & x_bct,
    const TensorValue & prompt_bct,
    const TensorValue & cond_btc,
    const TensorValue & style_bc,
    const TensorValue & timestep_b,
    const TensorValue & positions,
    ggml_tensor * freqs_tensor,
    const V1CfmWeights & weights,
    const SeedVcV1DitConfig & config,
    const SeedVcV1WavenetConfig & wavenet_config,
    int64_t style_dim) {
    auto t_emb = build_timestep_embedding(
        ctx,
        freqs_tensor,
        timestep_b,
        weights.t_embedder_0,
        weights.t_embedder_2,
        config.hidden_dim);
    auto t_token = engine::core::reshape_tensor(ctx, contiguous(ctx, t_emb), TensorShape::from_dims({t_emb.shape.dims[0], 1, config.hidden_dim}));
    auto cond = engine::modules::LinearModule({config.content_dim, config.hidden_dim, true, GGML_PREC_F32})
                    .build(ctx, cond_btc, weights.cond_projection);
    auto x = engine::modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, x_bct);
    auto prompt = engine::modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, prompt_bct);
    auto merged = engine::modules::ConcatModule({2}).build(ctx, x, prompt);
    merged = engine::modules::ConcatModule({2}).build(ctx, merged, cond);
    if (config.style_condition && !config.style_as_token) {
        auto style = engine::core::reshape_tensor(ctx, contiguous(ctx, style_bc), TensorShape::from_dims({style_bc.shape.dims[0], 1, style_dim}));
        style = engine::modules::RepeatModule({TensorShape::from_dims({style_bc.shape.dims[0], x_bct.shape.dims[2], style_dim})})
                    .build(ctx, style);
        merged = engine::modules::ConcatModule({2}).build(ctx, merged, style);
    }
    const int64_t merge_dim =
        config.in_channels * 2 + config.hidden_dim + (config.style_condition && !config.style_as_token ? style_dim : 0);
    auto hidden = engine::modules::LinearModule({merge_dim, config.hidden_dim, true, GGML_PREC_F32})
                      .build(ctx, merged, weights.cond_x_merge_linear);

    if (config.style_as_token) {
        if (!weights.style_in.has_value()) {
            throw std::runtime_error("Seed-VC V1 CFM style-token config requires style_in weights");
        }
        auto style = engine::modules::LinearModule({style_dim, config.hidden_dim, true, GGML_PREC_F32})
                         .build(ctx, style_bc, *weights.style_in);
        style = engine::core::reshape_tensor(ctx, contiguous(ctx, style), TensorShape::from_dims({style_bc.shape.dims[0], 1, config.hidden_dim}));
        hidden = engine::modules::ConcatModule({1}).build(ctx, style, hidden);
    }
    if (config.time_as_token) {
        hidden = engine::modules::ConcatModule({1}).build(ctx, t_token, hidden);
    }
    std::vector<TensorValue> uvit_skips;
    uvit_skips.reserve(static_cast<size_t>(config.depth / 2));
    for (int64_t layer_index = 0; layer_index < static_cast<int64_t>(weights.layers.size()); ++layer_index) {
        const auto & layer = weights.layers[static_cast<size_t>(layer_index)];
        if (config.uvit_skip_connection && layer_index > config.depth / 2) {
            if (uvit_skips.empty()) {
                throw std::runtime_error("Seed-VC V1 CFM U-ViT skip stack underflow");
            }
            auto skip = uvit_skips.back();
            uvit_skips.pop_back();
            auto cat = engine::modules::ConcatModule({2}).build(ctx, hidden, skip);
            hidden = engine::modules::LinearModule({config.hidden_dim * 2, config.hidden_dim, true, GGML_PREC_F32})
                         .build(ctx, cat, layer.skip_in_linear);
        }
        hidden = build_transformer_layer(ctx, hidden, t_token, positions, layer, config);
        if (config.uvit_skip_connection && layer_index < config.depth / 2) {
            uvit_skips.push_back(hidden);
        }
    }
    hidden = build_adaptive_rms_norm(
        ctx,
        hidden,
        t_token,
        weights.final_norm_project,
        weights.final_norm,
        config);
    if (config.time_as_token) {
        hidden = slice_tokens(ctx, hidden, 1, x_bct.shape.dims[2] + (config.style_as_token ? 1 : 0));
    }
    if (config.style_as_token) {
        hidden = slice_tokens(ctx, hidden, 1, x_bct.shape.dims[2]);
    }
    if (config.long_skip_connection) {
        if (!weights.long_skip_linear.has_value()) {
            throw std::runtime_error("Seed-VC V1 CFM long-skip config requires skip_linear weights");
        }
        auto cat = engine::modules::ConcatModule({2}).build(ctx, hidden, x);
        hidden = engine::modules::LinearModule({config.hidden_dim + config.in_channels, config.hidden_dim, true, GGML_PREC_F32})
                     .build(ctx, cat, *weights.long_skip_linear);
    }
    if (config.final_layer_type == "wavenet") {
        if (!weights.wavenet.has_value()) {
            throw std::runtime_error("Seed-VC V1 CFM wavenet final path requires WaveNet weights");
        }
        auto hidden_w = engine::modules::LinearModule({config.hidden_dim, wavenet_config.hidden_dim, true, GGML_PREC_F32})
                            .build(ctx, hidden, weights.wavenet->conv1);
        hidden_w = engine::modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, hidden_w);
        auto t_emb2 = build_timestep_embedding(
            ctx,
            freqs_tensor,
            timestep_b,
            weights.wavenet->t_embedder2_0,
            weights.wavenet->t_embedder2_2,
            wavenet_config.hidden_dim);
        hidden_w = build_wavenet(ctx, hidden_w, t_emb2, *weights.wavenet, wavenet_config);
        hidden_w = engine::modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, hidden_w);
        auto projected_res = engine::modules::LinearModule({config.hidden_dim, wavenet_config.hidden_dim, true, GGML_PREC_F32})
                                 .build(ctx, hidden, weights.wavenet->res_projection);
        hidden_w = add(ctx, hidden_w, projected_res);
        hidden_w = build_wavenet_final_layer(ctx, hidden_w, t_emb, *weights.wavenet, wavenet_config.hidden_dim);
        hidden_w = engine::modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, hidden_w);
        return engine::modules::Conv1dModule({wavenet_config.hidden_dim, config.in_channels, 1, 1, 0, 1, true})
            .build(ctx, hidden_w, weights.wavenet->conv2);
    }
    if (!weights.final_mlp_0.has_value() || !weights.final_mlp_2.has_value()) {
        throw std::runtime_error("Seed-VC V1 CFM MLP final path requires final_mlp weights");
    }
    hidden = engine::modules::LinearModule({config.hidden_dim, config.hidden_dim, true, GGML_PREC_F32})
                 .build(ctx, hidden, *weights.final_mlp_0);
    hidden = engine::modules::SiluModule{}.build(ctx, hidden);
    hidden = engine::modules::LinearModule({config.hidden_dim, config.in_channels, true, GGML_PREC_F32})
                 .build(ctx, hidden, *weights.final_mlp_2);
    return engine::modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, hidden);
}

class V1CfmEstimatorRunner {
public:
    V1CfmEstimatorRunner(
        engine::core::ExecutionContext & execution_context,
        V1CfmWeights weights,
        SeedVcV1DitConfig config,
        SeedVcV1WavenetConfig wavenet_config,
        int64_t style_dim)
        : execution_context_(execution_context),
          config_(std::move(config)),
          wavenet_config_(std::move(wavenet_config)),
          style_dim_(style_dim),
          weights_(std::move(weights)) {
        if (config_.in_channels <= 0 || config_.content_dim <= 0 || style_dim_ <= 0) {
            throw std::runtime_error("Seed-VC V1 CFM config is invalid");
        }
    }

    ~V1CfmEstimatorRunner() {
        release_graph();
    }

    SeedVcV1CfmEstimatorOutput run(const SeedVcV1CfmEstimatorInput & input) {
        std::lock_guard<std::mutex> lock(mutex_);
        validate_input(input);
        ensure_graph(input.batch, input.frames);
        engine::core::write_tensor_f32(x_, input.x);
        engine::core::write_tensor_f32(prompt_, input.prompt);
        engine::core::write_tensor_f32(cond_, input.cond);
        engine::core::write_tensor_f32(style_, input.style);
        engine::core::write_tensor_f32(timestep_, input.timestep);
        ggml_backend_tensor_set(freqs_, freqs_values_.data(), 0, freqs_values_.size() * sizeof(float));
        ggml_backend_tensor_set(positions_, position_values_.data(), 0, position_values_.size() * sizeof(int32_t));
        if (engine::core::compute_backend_graph(execution_context_.backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for Seed-VC V1 CFM estimator");
        }
        SeedVcV1CfmEstimatorOutput out;
        out.velocity = engine::core::read_tensor_f32(output_);
        out.batch = input.batch;
        out.channels = config_.in_channels;
        out.frames = input.frames;
        return out;
    }

private:
    void validate_input(const SeedVcV1CfmEstimatorInput & input) const {
        if (input.batch <= 0 || input.frames <= 0) {
            throw std::runtime_error("Seed-VC V1 CFM estimator requires positive batch and frame count");
        }
        if (static_cast<int64_t>(input.x.size()) != input.batch * config_.in_channels * input.frames ||
            static_cast<int64_t>(input.prompt.size()) != input.batch * config_.in_channels * input.frames ||
            static_cast<int64_t>(input.cond.size()) != input.batch * input.frames * config_.content_dim ||
            static_cast<int64_t>(input.style.size()) != input.batch * style_dim_ ||
            static_cast<int64_t>(input.timestep.size()) != input.batch) {
            throw std::runtime_error("Seed-VC V1 CFM estimator input shape mismatch");
        }
    }

    void release_graph() {
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (ctx_ != nullptr) {
            ggml_free(ctx_);
            ctx_ = nullptr;
        }
        graph_ = nullptr;
        output_ = nullptr;
        x_ = {};
        prompt_ = {};
        cond_ = {};
        style_ = {};
        timestep_ = {};
        freqs_ = nullptr;
        positions_ = nullptr;
        batch_ = 0;
        frames_ = 0;
        graph_tokens_ = 0;
    }

    void ensure_graph(int64_t batch, int64_t frames) {
        const int64_t graph_tokens = frames + (config_.time_as_token ? 1 : 0) + (config_.style_as_token ? 1 : 0);
        if (ctx_ != nullptr && batch_ == batch && frames_ == frames && graph_tokens_ == graph_tokens) {
            return;
        }
        release_graph();
        ggml_init_params params{1024ull * 1024ull * 1024ull, nullptr, true};
        ctx_ = ggml_init(params);
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Seed-VC V1 CFM estimator graph context");
        }
        engine::core::ModuleBuildContext ctx{ctx_, "seed_vc.v1_cfm.estimator", execution_context_.backend_type()};
        x_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, config_.in_channels, frames}));
        prompt_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, config_.in_channels, frames}));
        cond_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, frames, config_.content_dim}));
        style_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, style_dim_}));
        timestep_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({batch}));
        positions_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, graph_tokens);
        freqs_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 128);
        ggml_set_input(x_.tensor);
        ggml_set_input(prompt_.tensor);
        ggml_set_input(cond_.tensor);
        ggml_set_input(style_.tensor);
        ggml_set_input(timestep_.tensor);
        ggml_set_input(positions_);
        ggml_set_input(freqs_);
        auto output = build_estimator_graph(
            ctx,
            x_,
            prompt_,
            cond_,
            style_,
            timestep_,
            engine::core::wrap_tensor(positions_, TensorShape::from_dims({graph_tokens}), GGML_TYPE_I32),
            freqs_,
            weights_,
            config_,
            wavenet_config_,
            style_dim_);
        output_ = output.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_, 262144, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_context_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate Seed-VC V1 CFM estimator graph memory");
        }
        freqs_values_ = make_timestep_freqs();
        position_values_.assign(static_cast<size_t>(graph_tokens), 0);
        for (int64_t i = 0; i < graph_tokens; ++i) {
            position_values_[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        batch_ = batch;
        frames_ = frames;
        graph_tokens_ = graph_tokens;
    }

    engine::core::ExecutionContext & execution_context_;
    SeedVcV1DitConfig config_;
    SeedVcV1WavenetConfig wavenet_config_;
    int64_t style_dim_ = 0;
    V1CfmWeights weights_;
    std::mutex mutex_;
    ggml_context * ctx_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    TensorValue x_;
    TensorValue prompt_;
    TensorValue cond_;
    TensorValue style_;
    TensorValue timestep_;
    ggml_tensor * freqs_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * output_ = nullptr;
    std::vector<float> freqs_values_;
    std::vector<int32_t> position_values_;
    int64_t batch_ = 0;
    int64_t frames_ = 0;
    int64_t graph_tokens_ = 0;
};

void zero_prompt_region(std::vector<float> & values, int64_t batch, int64_t channels, int64_t frames, int64_t prompt_frames) {
    if (prompt_frames <= 0) {
        return;
    }
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            const size_t base = static_cast<size_t>((b * channels + c) * frames);
            std::fill(
                values.begin() + static_cast<std::ptrdiff_t>(base),
                values.begin() + static_cast<std::ptrdiff_t>(base + static_cast<size_t>(prompt_frames)),
                0.0F);
        }
    }
}

std::vector<float> make_prompt_x(
    const std::vector<float> & prompt,
    int64_t batch,
    int64_t channels,
    int64_t frames,
    int64_t prompt_frames) {
    if (prompt_frames < 0 || prompt_frames > frames) {
        throw std::runtime_error("Seed-VC V1 CFM prompt length is out of range");
    }
    std::vector<float> out(static_cast<size_t>(batch * channels * frames), 0.0F);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            const size_t base = static_cast<size_t>((b * channels + c) * frames);
            std::copy(
                prompt.begin() + static_cast<std::ptrdiff_t>(base),
                prompt.begin() + static_cast<std::ptrdiff_t>(base + static_cast<size_t>(prompt_frames)),
                out.begin() + static_cast<std::ptrdiff_t>(base));
        }
    }
    return out;
}

std::vector<float> repeat_or_zero_batch(
    const std::vector<float> & values,
    int64_t row_values,
    bool keep_first,
    bool keep_second) {
    std::vector<float> out(static_cast<size_t>(2 * row_values), 0.0F);
    if (keep_first) {
        std::copy(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(row_values), out.begin());
    }
    if (keep_second) {
        std::copy(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(row_values), out.begin() + static_cast<std::ptrdiff_t>(row_values));
    }
    return out;
}

std::vector<float> duplicate_batch(const std::vector<float> & values, int64_t row_values) {
    return repeat_or_zero_batch(values, row_values, true, true);
}

std::vector<float> combine_cfg(
    const std::vector<float> & velocity,
    int64_t channels,
    int64_t frames,
    float cfg_rate) {
    const int64_t row_values = channels * frames;
    std::vector<float> out(static_cast<size_t>(row_values), 0.0F);
    for (int64_t i = 0; i < row_values; ++i) {
        out[static_cast<size_t>(i)] =
            (1.0F + cfg_rate) * velocity[static_cast<size_t>(i)] -
            cfg_rate * velocity[static_cast<size_t>(row_values + i)];
    }
    return out;
}

void add_scaled(std::vector<float> & x, const std::vector<float> & velocity, float scale) {
    if (x.size() != velocity.size()) {
        throw std::runtime_error("Seed-VC V1 CFM Euler update shape mismatch");
    }
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] += scale * velocity[i];
    }
}

}  // namespace

struct SeedVcV1CfmEstimator::State {
    std::shared_ptr<engine::core::ExecutionContext> execution_context;
    std::shared_ptr<engine::core::BackendWeightStore> store;
    std::unique_ptr<V1CfmEstimatorRunner> runner;
};

SeedVcV1CfmEstimator::SeedVcV1CfmEstimator(
    std::shared_ptr<const engine::assets::TensorSource> source,
    engine::core::BackendConfig backend,
    engine::assets::TensorStorageType storage_type,
    SeedVcV1DitConfig config,
    SeedVcV1WavenetConfig wavenet_config,
    int64_t style_dim)
    : config_(std::move(config)),
      wavenet_config_(std::move(wavenet_config)),
      style_dim_(style_dim) {
    if (source == nullptr) {
        throw std::runtime_error("Seed-VC V1 CFM requires weights");
    }
    if (!config_.style_condition || !config_.uvit_skip_connection) {
        throw std::runtime_error("Seed-VC V1 CFM config does not match supported Python V1 DiT paths");
    }
    if (config_.final_layer_type != "mlp" && config_.final_layer_type != "wavenet") {
        throw std::runtime_error("Seed-VC V1 CFM unsupported final layer type: " + config_.final_layer_type);
    }
    state_ = std::make_shared<State>();
    state_->execution_context = std::make_shared<engine::core::ExecutionContext>(backend);
    state_->store = std::make_shared<engine::core::BackendWeightStore>(
        state_->execution_context->backend(),
        state_->execution_context->backend_type(),
        "seed_vc.v1_cfm.estimator.weights",
        256ull * 1024ull * 1024ull);
    auto weights = load_v1_cfm_weights(*state_->store, *source, config_.depth, storage_type);
    state_->store->upload();
    state_->runner =
        std::make_unique<V1CfmEstimatorRunner>(*state_->execution_context, std::move(weights), config_, wavenet_config_, style_dim_);
}

SeedVcV1CfmEstimator::~SeedVcV1CfmEstimator() = default;
SeedVcV1CfmEstimator::SeedVcV1CfmEstimator(SeedVcV1CfmEstimator &&) noexcept = default;
SeedVcV1CfmEstimator & SeedVcV1CfmEstimator::operator=(SeedVcV1CfmEstimator &&) noexcept = default;

SeedVcV1CfmEstimatorOutput SeedVcV1CfmEstimator::run(const SeedVcV1CfmEstimatorInput & input) const {
    if (state_ == nullptr || state_->runner == nullptr) {
        throw std::runtime_error("Seed-VC V1 CFM estimator is not initialized");
    }
    return state_->runner->run(input);
}

SeedVcV1CfmEstimatorOutput SeedVcV1CfmEstimator::infer(const SeedVcV1CfmInferenceInput & input) const {
    if (input.batch != 1 || input.frames <= 0 || input.prompt_frames < 0 || input.prompt_frames > input.frames ||
        input.num_inference_steps <= 0 || !(input.temperature > 0.0F)) {
        throw std::runtime_error("Seed-VC V1 CFM inference input is invalid");
    }
    const int64_t channels = config_.in_channels;
    const int64_t row_values = channels * input.frames;
    if (static_cast<int64_t>(input.mu.size()) != input.frames * config_.content_dim ||
        static_cast<int64_t>(input.prompt.size()) != row_values ||
        static_cast<int64_t>(input.initial_noise.size()) != row_values ||
        static_cast<int64_t>(input.style.size()) != style_dim_) {
        throw std::runtime_error("Seed-VC V1 CFM inference shape mismatch");
    }

    std::vector<float> x = input.initial_noise;
    if (input.temperature != 1.0F) {
        for (float & value : x) {
            value *= input.temperature;
        }
    }
    zero_prompt_region(x, 1, channels, input.frames, input.prompt_frames);
    auto prompt_x = make_prompt_x(input.prompt, 1, channels, input.frames, input.prompt_frames);

    for (int step = 1; step <= input.num_inference_steps; ++step) {
        const float t = static_cast<float>(step - 1) / static_cast<float>(input.num_inference_steps);
        const float dt = 1.0F / static_cast<float>(input.num_inference_steps);
        std::vector<float> velocity;
        if (input.inference_cfg_rate > 0.0F) {
            SeedVcV1CfmEstimatorInput estimator_input;
            estimator_input.x = duplicate_batch(x, row_values);
            estimator_input.prompt = repeat_or_zero_batch(prompt_x, row_values, true, false);
            estimator_input.cond = repeat_or_zero_batch(input.mu, input.frames * config_.content_dim, true, false);
            estimator_input.style = repeat_or_zero_batch(input.style, style_dim_, true, false);
            estimator_input.timestep = {t, t};
            estimator_input.batch = 2;
            estimator_input.frames = input.frames;
            velocity = combine_cfg(run(estimator_input).velocity, channels, input.frames, input.inference_cfg_rate);
        } else {
            SeedVcV1CfmEstimatorInput estimator_input;
            estimator_input.x = x;
            estimator_input.prompt = prompt_x;
            estimator_input.cond = input.mu;
            estimator_input.style = input.style;
            estimator_input.timestep = {t};
            estimator_input.batch = 1;
            estimator_input.frames = input.frames;
            velocity = run(estimator_input).velocity;
        }
        add_scaled(x, velocity, dt);
        zero_prompt_region(x, 1, channels, input.frames, input.prompt_frames);
    }

    SeedVcV1CfmEstimatorOutput out;
    out.velocity = std::move(x);
    out.batch = 1;
    out.channels = channels;
    out.frames = input.frames;
    return out;
}

}  // namespace engine::models::seed_vc
