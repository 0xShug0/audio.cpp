#include "engine/community_models/glm_tts/flow.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include "ggml.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

namespace engine::models::glm_tts {
namespace {

using core::TensorShape;
using core::TensorValue;

TensorValue contiguous(core::ModuleBuildContext & ctx, const TensorValue & value) {
    return core::ensure_backend_addressable_layout(ctx, value);
}

TensorValue add(core::ModuleBuildContext & ctx, const TensorValue & lhs, const TensorValue & rhs) {
    return modules::AddModule{}.build(ctx, lhs, rhs);
}

TensorValue mul(core::ModuleBuildContext & ctx, const TensorValue & lhs, const TensorValue & rhs) {
    return modules::MulModule{}.build(ctx, lhs, rhs);
}

TensorValue add_one(core::ModuleBuildContext & ctx, const TensorValue & value) {
    return core::wrap_tensor(
        ggml_scale_bias(ctx.ggml, value.tensor, 1.0F, 1.0F),
        value.shape,
        GGML_TYPE_F32);
}

TensorValue slice_last(
    core::ModuleBuildContext & ctx,
    const TensorValue & input,
    int64_t start,
    int64_t length) {
    return modules::SliceModule(
        {static_cast<int>(input.shape.rank - 1), start, length})
        .build(ctx, input);
}

TensorValue repeat_to(
    core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const TensorShape & shape) {
    return modules::RepeatModule({shape}).build(ctx, input);
}

TensorValue expand_batch_frames(
    core::ModuleBuildContext & ctx,
    const TensorValue & value,
    int64_t batch,
    int64_t frames,
    int64_t channels) {
    auto reshaped = core::reshape_tensor(
        ctx,
        contiguous(ctx, value),
        TensorShape::from_dims({batch, 1, channels}));
    return repeat_to(
        ctx, reshaped, TensorShape::from_dims({batch, frames, channels}));
}

TensorValue mish(core::ModuleBuildContext & ctx, const TensorValue & input) {
    const auto input_contiguous = contiguous(ctx, input);
    auto softplus = core::wrap_tensor(
        ggml_softplus(ctx.ggml, input_contiguous.tensor),
        input.shape,
        GGML_TYPE_F32);
    auto gate = core::wrap_tensor(
        ggml_tanh(ctx.ggml, softplus.tensor), input.shape, GGML_TYPE_F32);
    return mul(ctx, input_contiguous, gate);
}

TensorValue require_tensor(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & name,
    assets::TensorStorageType storage_type) {
    return modules::binding::tensor_from_named_source(
        store, source, name, storage_type);
}

TensorValue require_f32_tensor(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & name) {
    return modules::binding::f32_tensor_from_named_source(
        store, source, name);
}

modules::LinearWeights load_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    bool use_bias = true) {
    modules::LinearWeights out{
        require_tensor(store, source, prefix + ".weight", storage_type),
        std::nullopt};
    if (use_bias) {
        out.bias = require_f32_tensor(store, source, prefix + ".bias");
    }
    return out;
}

modules::Conv1dWeights load_conv1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type) {
    return modules::Conv1dWeights{
        require_tensor(store, source, prefix + ".weight", storage_type),
        require_f32_tensor(store, source, prefix + ".bias")};
}

modules::NormWeights load_norm(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix) {
    return modules::NormWeights{
        require_f32_tensor(store, source, prefix + ".weight"),
        require_f32_tensor(store, source, prefix + ".bias")};
}

struct TextBlockWeights {
    modules::Conv1dWeights dwconv;
    modules::NormWeights norm;
    modules::LinearWeights pwconv1;
    TensorValue grn_gamma;
    TensorValue grn_beta;
    modules::LinearWeights pwconv2;
};

struct TransformerBlockWeights {
    modules::LinearWeights modulation;
    modules::LinearWeights q;
    modules::LinearWeights k;
    modules::LinearWeights v;
    modules::LinearWeights out;
    modules::LinearWeights ff1;
    modules::LinearWeights ff2;
};

struct FlowWeights {
    TensorValue text_embedding;
    std::vector<TextBlockWeights> text_blocks;
    modules::LinearWeights input_projection;
    std::vector<modules::Conv1dWeights> position_conv1;
    std::vector<modules::Conv1dWeights> position_conv2;
    modules::LinearWeights time_mlp1;
    modules::LinearWeights time_mlp2;
    std::vector<TransformerBlockWeights> transformer_blocks;
    modules::LinearWeights final_modulation;
    modules::LinearWeights output_projection;
};

std::vector<modules::Conv1dWeights> load_grouped_position_conv(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    int64_t groups,
    int64_t kernel) {
    const int64_t group_channels = channels / groups;
    const auto weight = source.require_f32_tensor(
        prefix + ".weight", {channels, group_channels, kernel});
    const auto bias =
        source.require_f32_tensor(prefix + ".bias", {channels});
    const int64_t group_weight_values =
        group_channels * group_channels * kernel;
    std::vector<modules::Conv1dWeights> out;
    out.reserve(static_cast<size_t>(groups));
    for (int64_t group = 0; group < groups; ++group) {
        const auto weight_begin =
            weight.values.begin() +
            static_cast<std::ptrdiff_t>(group * group_weight_values);
        const auto bias_begin =
            bias.values.begin() +
            static_cast<std::ptrdiff_t>(group * group_channels);
        std::vector<float> group_weight(
            weight_begin,
            weight_begin +
                static_cast<std::ptrdiff_t>(group_weight_values));
        std::vector<float> group_bias(
            bias_begin,
            bias_begin + static_cast<std::ptrdiff_t>(group_channels));
        out.push_back(
            {store.make_f32(
                 TensorShape::from_dims(
                     {group_channels, group_channels, kernel}),
                 std::move(group_weight)),
             store.make_f32(
                 TensorShape::from_dims({group_channels}),
                 std::move(group_bias))});
    }
    return out;
}

FlowWeights load_flow_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const GlmTTSFlowConfig & config,
    assets::TensorStorageType storage_type) {
    const std::string root = "estimator.";
    FlowWeights out;
    out.text_embedding = require_tensor(
        store,
        source,
        root + "text_emb_layer.text_embed.weight",
        storage_type);
    out.text_blocks.reserve(static_cast<size_t>(config.conv_layers));
    for (int64_t layer = 0; layer < config.conv_layers; ++layer) {
        const auto prefix =
            root + "text_emb_layer.text_blocks." + std::to_string(layer);
        TextBlockWeights block;
        block.dwconv =
            load_conv1d(store, source, prefix + ".dwconv", storage_type);
        block.norm = load_norm(store, source, prefix + ".norm");
        block.pwconv1 =
            load_linear(store, source, prefix + ".pwconv1", storage_type);
        block.grn_gamma =
            require_f32_tensor(store, source, prefix + ".grn.gamma");
        block.grn_beta =
            require_f32_tensor(store, source, prefix + ".grn.beta");
        block.pwconv2 =
            load_linear(store, source, prefix + ".pwconv2", storage_type);
        out.text_blocks.push_back(std::move(block));
    }
    out.input_projection =
        load_linear(store, source, root + "emb_concator.proj", storage_type);
    out.position_conv1 = load_grouped_position_conv(
        store,
        source,
        root + "emb_concator.conv_pos_embed.conv1d.0",
        config.trans_dim,
        16,
        31);
    out.position_conv2 = load_grouped_position_conv(
        store,
        source,
        root + "emb_concator.conv_pos_embed.conv1d.2",
        config.trans_dim,
        16,
        31);
    out.time_mlp1 =
        load_linear(store, source, root + "time_embed.time_mlp.0", storage_type);
    out.time_mlp2 =
        load_linear(store, source, root + "time_embed.time_mlp.2", storage_type);
    out.transformer_blocks.reserve(static_cast<size_t>(config.depth));
    for (int64_t layer = 0; layer < config.depth; ++layer) {
        const auto prefix =
            root + "transformer_blocks." + std::to_string(layer);
        TransformerBlockWeights block;
        block.modulation = load_linear(
            store, source, prefix + ".attn_norm.linear", storage_type);
        block.q =
            load_linear(store, source, prefix + ".attn.to_q", storage_type);
        block.k =
            load_linear(store, source, prefix + ".attn.to_k", storage_type);
        block.v =
            load_linear(store, source, prefix + ".attn.to_v", storage_type);
        block.out = load_linear(
            store, source, prefix + ".attn.to_out.0", storage_type);
        block.ff1 =
            load_linear(store, source, prefix + ".ff.ff.0.0", storage_type);
        block.ff2 =
            load_linear(store, source, prefix + ".ff.ff.2", storage_type);
        out.transformer_blocks.push_back(std::move(block));
    }
    out.final_modulation =
        load_linear(store, source, root + "norm_out.linear", storage_type);
    out.output_projection =
        load_linear(store, source, root + "proj_out", storage_type);
    return out;
}

TensorValue build_grn(
    core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const TextBlockWeights & weights) {
    const int64_t batch = input.shape.dims[0];
    const int64_t frames = input.shape.dims[1];
    const int64_t channels = input.shape.dims[2];
    auto squared = core::wrap_tensor(
        ggml_sqr(ctx.ggml, input.tensor), input.shape, GGML_TYPE_F32);
    auto sum_time = modules::ReduceSumModule({1}).build(ctx, squared);
    auto gx = core::wrap_tensor(
        ggml_sqrt(ctx.ggml, sum_time.tensor), sum_time.shape, GGML_TYPE_F32);
    auto mean_channels = modules::ReduceMeanModule({2}).build(ctx, gx);
    auto denominator = core::wrap_tensor(
        ggml_scale_bias(ctx.ggml, mean_channels.tensor, 1.0F, 1.0e-6F),
        mean_channels.shape,
        GGML_TYPE_F32);
    auto nx = core::wrap_tensor(
        ggml_div(
            ctx.ggml,
            gx.tensor,
            repeat_to(ctx, denominator, gx.shape).tensor),
        gx.shape,
        GGML_TYPE_F32);
    auto nx_full = repeat_to(
        ctx, nx, TensorShape::from_dims({batch, frames, channels}));
    auto gamma = repeat_to(ctx, weights.grn_gamma, input.shape);
    auto beta = repeat_to(ctx, weights.grn_beta, input.shape);
    return add(ctx, add(ctx, mul(ctx, gamma, mul(ctx, input, nx_full)), beta), input);
}

TensorValue build_depthwise_text_conv(
    core::ModuleBuildContext & ctx,
    const TensorValue & input_btc,
    const modules::Conv1dWeights & weights,
    int64_t channels) {
    auto input_bct = contiguous(
        ctx,
        modules::TransposeModule({{0, 2, 1, 3}, 3})
            .build(ctx, input_btc));
    auto input_4d = core::reshape_tensor(
        ctx,
        contiguous(ctx, input_bct),
        TensorShape::from_dims(
            {input_bct.shape.dims[0], channels, 1, input_bct.shape.dims[2]}));
    auto weight_4d = core::reshape_tensor(
        ctx,
        contiguous(ctx, weights.weight),
        TensorShape::from_dims({channels, 1, 1, 7}));
    modules::Conv2dWeights conv_weights{weight_4d, weights.bias};
    auto output_4d = modules::DepthwiseConv2dModule(
        {channels, 1, 7, 1, 1, 0, 3, 1, 1, true})
                         .build(ctx, input_4d, conv_weights);
    auto output_bct = core::reshape_tensor(
        ctx,
        contiguous(ctx, output_4d),
        TensorShape::from_dims(
            {input_bct.shape.dims[0], channels, input_bct.shape.dims[2]}));
    return modules::TransposeModule({{0, 2, 1, 3}, 3})
        .build(ctx, output_bct);
}

TensorValue build_text_embedding(
    core::ModuleBuildContext & ctx,
    const TensorValue & tokens,
    const TensorValue & position_embedding,
    const FlowWeights & weights,
    const GlmTTSFlowConfig & config) {
    auto hidden = modules::EmbeddingModule(
        {config.vocab_size + 1, config.speech_token_dim})
                      .build(ctx, tokens, weights.text_embedding);
    hidden = add(ctx, hidden, position_embedding);
    for (const auto & block : weights.text_blocks) {
        auto residual = hidden;
        hidden = build_depthwise_text_conv(
            ctx, hidden, block.dwconv, config.speech_token_dim);
        hidden = modules::LayerNormModule(
            {config.speech_token_dim, 1.0e-6F, true, true})
                     .build(ctx, hidden, block.norm);
        hidden = modules::LinearModule(
            {config.speech_token_dim,
             config.speech_token_dim * 2,
             true,
             GGML_PREC_F32})
                     .build(ctx, hidden, block.pwconv1);
        hidden = modules::GeluModule({modules::GeluApproximation::ExactErf})
                     .build(ctx, hidden);
        hidden = build_grn(ctx, hidden, block);
        hidden = modules::LinearModule(
            {config.speech_token_dim * 2,
             config.speech_token_dim,
             true,
             GGML_PREC_F32})
                     .build(ctx, hidden, block.pwconv2);
        hidden = add(ctx, residual, hidden);
    }
    return hidden;
}

TensorValue grouped_position_conv(
    core::ModuleBuildContext & ctx,
    const TensorValue & input_btc,
    const std::vector<modules::Conv1dWeights> & weights,
    int64_t channels,
    int64_t groups) {
    const int64_t group_channels = channels / groups;
    auto input_bct = contiguous(
        ctx,
        modules::TransposeModule({{0, 2, 1, 3}, 3})
            .build(ctx, input_btc));
    TensorValue joined;
    for (int64_t group = 0; group < groups; ++group) {
        auto group_input = modules::SliceModule(
            {1, group * group_channels, group_channels})
                               .build(ctx, input_bct);
        auto group_output = modules::Conv1dModule(
            {group_channels,
             group_channels,
             31,
             1,
             15,
             1,
             true})
                                .build(
                                    ctx,
                                    group_input,
                                    weights[static_cast<size_t>(group)]);
        joined = group == 0
            ? group_output
            : modules::ConcatModule({1}).build(ctx, joined, group_output);
    }
    return modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, joined);
}

TensorValue build_time_embedding(
    core::ModuleBuildContext & ctx,
    const TensorValue & time_features,
    const FlowWeights & weights,
    const GlmTTSFlowConfig & config) {
    auto hidden = modules::LinearModule(
        {256, config.trans_dim, true, GGML_PREC_F32})
                      .build(ctx, time_features, weights.time_mlp1);
    hidden = modules::SiluModule{}.build(ctx, hidden);
    return modules::LinearModule(
        {config.trans_dim, config.trans_dim, true, GGML_PREC_F32})
        .build(ctx, hidden, weights.time_mlp2);
}

struct AdaptiveNormOutput {
    TensorValue normalized;
    TensorValue gate_msa;
    TensorValue shift_mlp;
    TensorValue scale_mlp;
    TensorValue gate_mlp;
};

AdaptiveNormOutput build_adaptive_norm(
    core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const TensorValue & condition,
    const TransformerBlockWeights & weights,
    const GlmTTSFlowConfig & config) {
    const int64_t batch = input.shape.dims[0];
    const int64_t frames = input.shape.dims[1];
    auto modulation = modules::SiluModule{}.build(ctx, condition);
    modulation = modules::LinearModule(
        {config.trans_dim + 192,
         config.trans_dim * 6,
         true,
         GGML_PREC_F32})
                     .build(ctx, modulation, weights.modulation);
    auto shift_msa = expand_batch_frames(
        ctx,
        slice_last(ctx, modulation, 0, config.trans_dim),
        batch,
        frames,
        config.trans_dim);
    auto scale_msa = expand_batch_frames(
        ctx,
        slice_last(ctx, modulation, config.trans_dim, config.trans_dim),
        batch,
        frames,
        config.trans_dim);
    auto gate_msa = expand_batch_frames(
        ctx,
        slice_last(ctx, modulation, config.trans_dim * 2, config.trans_dim),
        batch,
        frames,
        config.trans_dim);
    auto shift_mlp = expand_batch_frames(
        ctx,
        slice_last(ctx, modulation, config.trans_dim * 3, config.trans_dim),
        batch,
        frames,
        config.trans_dim);
    auto scale_mlp = expand_batch_frames(
        ctx,
        slice_last(ctx, modulation, config.trans_dim * 4, config.trans_dim),
        batch,
        frames,
        config.trans_dim);
    auto gate_mlp = expand_batch_frames(
        ctx,
        slice_last(ctx, modulation, config.trans_dim * 5, config.trans_dim),
        batch,
        frames,
        config.trans_dim);
    auto normalized = modules::LayerNormModule(
        {config.trans_dim, 1.0e-6F, false, false})
                          .build(ctx, input, {});
    normalized =
        add(ctx, mul(ctx, normalized, add_one(ctx, scale_msa)), shift_msa);
    return {
        normalized,
        gate_msa,
        shift_mlp,
        scale_mlp,
        gate_mlp};
}

TensorValue build_attention(
    core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const TensorValue & positions,
    const TransformerBlockWeights & weights,
    const GlmTTSFlowConfig & config) {
    const int64_t batch = input.shape.dims[0];
    const int64_t frames = input.shape.dims[1];
    const int64_t hidden = config.heads * config.dim_head;
    auto q = modules::LinearModule(
        {config.trans_dim, hidden, true, GGML_PREC_F32})
                 .build(ctx, input, weights.q);
    auto k = modules::LinearModule(
        {config.trans_dim, hidden, true, GGML_PREC_F32})
                 .build(ctx, input, weights.k);
    auto v = modules::LinearModule(
        {config.trans_dim, hidden, true, GGML_PREC_F32})
                 .build(ctx, input, weights.v);
    auto apply_official_rope = [&](const TensorValue & projected) {
        auto rotated = slice_last(
            ctx, projected, 0, config.dim_head);
        rotated = core::reshape_tensor(
            ctx,
            contiguous(ctx, rotated),
            TensorShape::from_dims(
                {batch, frames, 1, config.dim_head}));
        rotated = modules::RoPEModule(
            {config.dim_head, GGML_ROPE_TYPE_NORMAL, 10000.0F})
                      .build(ctx, rotated, positions);
        rotated = core::reshape_tensor(
            ctx,
            contiguous(ctx, rotated),
            TensorShape::from_dims(
                {batch, frames, config.dim_head}));
        if (hidden == config.dim_head) {
            return rotated;
        }
        auto unrotated = slice_last(
            ctx,
            projected,
            config.dim_head,
            hidden - config.dim_head);
        return modules::ConcatModule({2}).build(
            ctx, rotated, unrotated);
    };
    q = apply_official_rope(q);
    k = apply_official_rope(k);
    q = core::reshape_tensor(
        ctx,
        contiguous(ctx, q),
        TensorShape::from_dims(
            {batch, frames, config.heads, config.dim_head}));
    k = core::reshape_tensor(
        ctx,
        contiguous(ctx, k),
        TensorShape::from_dims(
            {batch, frames, config.heads, config.dim_head}));
    v = core::reshape_tensor(
        ctx,
        contiguous(ctx, v),
        TensorShape::from_dims(
            {batch, frames, config.heads, config.dim_head}));
    q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
    v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
    q = contiguous(ctx, q);
    k = contiguous(ctx, k);
    v = contiguous(ctx, v);
    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q.tensor,
        k.tensor,
        v.tensor,
        nullptr,
        1.0F / std::sqrt(static_cast<float>(config.dim_head)),
        0.0F,
        0.0F);
    ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
    auto context = core::wrap_tensor(
        flash,
        TensorShape::from_dims(
            {batch, frames, config.heads, config.dim_head}),
        GGML_TYPE_F32);
    context = core::reshape_tensor(
        ctx,
        contiguous(ctx, context),
        TensorShape::from_dims({batch, frames, hidden}));
    return modules::LinearModule(
        {hidden, config.trans_dim, true, GGML_PREC_F32})
        .build(ctx, context, weights.out);
}

TensorValue build_transformer_block(
    core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const TensorValue & condition,
    const TensorValue & positions,
    const TransformerBlockWeights & weights,
    const GlmTTSFlowConfig & config) {
    auto modulation =
        build_adaptive_norm(ctx, input, condition, weights, config);
    auto attention =
        build_attention(ctx, modulation.normalized, positions, weights, config);
    auto hidden = add(ctx, input, mul(ctx, modulation.gate_msa, attention));
    auto ff_input = modules::LayerNormModule(
        {config.trans_dim, 1.0e-6F, false, false})
                        .build(ctx, hidden, {});
    ff_input = add(
        ctx,
        mul(ctx, ff_input, add_one(ctx, modulation.scale_mlp)),
        modulation.shift_mlp);
    auto ff = modules::LinearModule(
        {config.trans_dim, config.trans_dim * 2, true, GGML_PREC_F32})
                  .build(ctx, ff_input, weights.ff1);
    ff = modules::GeluModule({modules::GeluApproximation::Tanh})
             .build(ctx, ff);
    ff = modules::LinearModule(
        {config.trans_dim * 2, config.trans_dim, true, GGML_PREC_F32})
             .build(ctx, ff, weights.ff2);
    return add(ctx, hidden, mul(ctx, modulation.gate_mlp, ff));
}

TensorValue build_estimator(
    core::ModuleBuildContext & ctx,
    const TensorValue & x,
    const TensorValue & mel_condition,
    const TensorValue & tokens,
    const TensorValue & speaker,
    const TensorValue & time_features,
    const TensorValue & text_position_embedding,
    const TensorValue & positions,
    const FlowWeights & weights,
    const GlmTTSFlowConfig & config) {
    auto time =
        build_time_embedding(ctx, time_features, weights, config);
    auto transformer_condition =
        modules::ConcatModule({1}).build(ctx, time, speaker);
    auto text = build_text_embedding(
        ctx, tokens, text_position_embedding, weights, config);
    auto merged = modules::ConcatModule({2}).build(ctx, x, mel_condition);
    merged = modules::ConcatModule({2}).build(ctx, merged, text);
    auto hidden = modules::LinearModule(
        {config.mel_dim * 2 + config.speech_token_dim,
         config.trans_dim,
         true,
         GGML_PREC_F32})
                      .build(ctx, merged, weights.input_projection);
    auto position = grouped_position_conv(
        ctx, hidden, weights.position_conv1, config.trans_dim, 16);
    position = mish(ctx, position);
    position = grouped_position_conv(
        ctx, position, weights.position_conv2, config.trans_dim, 16);
    position = mish(ctx, position);
    hidden = add(ctx, hidden, position);
    for (const auto & block : weights.transformer_blocks) {
        hidden = build_transformer_block(
            ctx,
            hidden,
            transformer_condition,
            positions,
            block,
            config);
    }
    auto final_modulation =
        modules::SiluModule{}.build(ctx, transformer_condition);
    final_modulation = modules::LinearModule(
        {config.trans_dim + 192,
         config.trans_dim * 2,
         true,
         GGML_PREC_F32})
                           .build(
                               ctx,
                               final_modulation,
                               weights.final_modulation);
    auto scale = expand_batch_frames(
        ctx,
        slice_last(ctx, final_modulation, 0, config.trans_dim),
        hidden.shape.dims[0],
        hidden.shape.dims[1],
        config.trans_dim);
    auto shift = expand_batch_frames(
        ctx,
        slice_last(
            ctx, final_modulation, config.trans_dim, config.trans_dim),
        hidden.shape.dims[0],
        hidden.shape.dims[1],
        config.trans_dim);
    hidden = modules::LayerNormModule(
        {config.trans_dim, 1.0e-6F, false, false})
                 .build(ctx, hidden, {});
    hidden = add(ctx, mul(ctx, hidden, add_one(ctx, scale)), shift);
    return modules::LinearModule(
        {config.trans_dim, config.mel_dim, true, GGML_PREC_F32})
        .build(ctx, hidden, weights.output_projection);
}

std::vector<float> make_text_position_embedding(
    int64_t batch,
    int64_t frames,
    int64_t dim) {
    std::vector<float> out(
        static_cast<size_t>(batch * frames * dim), 0.0F);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t frame = 0; frame < frames; ++frame) {
            const int64_t position = std::min<int64_t>(frame, 4095);
            for (int64_t i = 0; i < dim / 2; ++i) {
                const double frequency = 1.0 /
                    std::pow(
                        10000.0,
                        static_cast<double>(i * 2) /
                            static_cast<double>(dim));
                const double angle =
                    static_cast<double>(position) * frequency;
                const auto base = static_cast<size_t>(
                    (b * frames + frame) * dim);
                out[base + static_cast<size_t>(i)] =
                    static_cast<float>(std::cos(angle));
                out[base + static_cast<size_t>(dim / 2 + i)] =
                    static_cast<float>(std::sin(angle));
            }
        }
    }
    return out;
}

std::vector<float> make_time_features(int64_t batch, float timestep) {
    constexpr int64_t kDim = 256;
    constexpr int64_t kHalf = kDim / 2;
    std::vector<float> out(static_cast<size_t>(batch * kDim), 0.0F);
    const double log_scale = std::log(10000.0) /
        static_cast<double>(kHalf - 1);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t i = 0; i < kHalf; ++i) {
            const double frequency =
                std::exp(-static_cast<double>(i) * log_scale);
            const double angle =
                1000.0 * static_cast<double>(timestep) * frequency;
            out[static_cast<size_t>(b * kDim + i)] =
                static_cast<float>(std::sin(angle));
            out[static_cast<size_t>(b * kDim + kHalf + i)] =
                static_cast<float>(std::cos(angle));
        }
    }
    return out;
}

std::vector<int32_t> interpolate_tokens(
    const std::vector<int32_t> & tokens,
    int64_t batch,
    int64_t frames) {
    if (tokens.empty()) {
        throw std::runtime_error("GLM-TTS Flow requires speech tokens");
    }
    std::vector<int32_t> row(static_cast<size_t>(frames), 0);
    const int64_t source_frames = static_cast<int64_t>(tokens.size());
    for (int64_t frame = 0; frame < frames; ++frame) {
        const int64_t source = std::min<int64_t>(
            source_frames - 1,
            (frame * source_frames) / frames);
        row[static_cast<size_t>(frame)] =
            tokens[static_cast<size_t>(source)];
    }
    std::vector<int32_t> out(
        static_cast<size_t>(batch * frames), 0);
    for (int64_t b = 0; b < batch; ++b) {
        std::copy(
            row.begin(),
            row.end(),
            out.begin() + static_cast<std::ptrdiff_t>(b * frames));
    }
    return out;
}

std::vector<float> repeat_row(
    const std::vector<float> & values,
    int64_t batch) {
    std::vector<float> out(
        static_cast<size_t>(batch) * values.size(), 0.0F);
    for (int64_t b = 0; b < batch; ++b) {
        std::copy(
            values.begin(),
            values.end(),
            out.begin() +
                static_cast<std::ptrdiff_t>(b * values.size()));
    }
    return out;
}

class FlowRunner {
public:
    FlowRunner(
        core::ExecutionContext & execution_context,
        FlowWeights weights,
        GlmTTSFlowConfig config)
        : execution_context_(execution_context),
          weights_(std::move(weights)),
          config_(std::move(config)) {
    }

    ~FlowRunner() {
        release_graph();
    }

    std::vector<float> run(
        const std::vector<float> & x,
        const std::vector<float> & mel_condition,
        const std::vector<int32_t> & tokens,
        const std::vector<float> & speaker,
        float timestep,
        int64_t batch,
        int64_t frames) {
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_graph(batch, frames);
        core::write_tensor_f32(x_, x);
        core::write_tensor_f32(mel_condition_, mel_condition);
        ggml_backend_tensor_set(
            tokens_.tensor,
            tokens.data(),
            0,
            tokens.size() * sizeof(int32_t));
        core::write_tensor_f32(speaker_, speaker);
        core::write_tensor_f32(
            time_features_, make_time_features(batch, timestep));
        core::write_tensor_f32(text_position_, text_position_values_);
        ggml_backend_tensor_set(
            positions_.tensor,
            position_values_.data(),
            0,
            position_values_.size() * sizeof(int32_t));
        if (core::compute_backend_graph(
                execution_context_.backend(), graph_) !=
            GGML_STATUS_SUCCESS) {
            throw std::runtime_error(
                "GLM-TTS Flow estimator graph computation failed");
        }
        return core::read_tensor_f32(output_);
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
        mel_condition_ = {};
        tokens_ = {};
        speaker_ = {};
        time_features_ = {};
        text_position_ = {};
        positions_ = {};
        text_position_values_.clear();
        position_values_.clear();
        batch_ = 0;
        frames_ = 0;
    }

private:
    void ensure_graph(int64_t batch, int64_t frames) {
        if (ctx_ != nullptr && batch_ == batch && frames_ == frames) {
            return;
        }
        release_graph();
        ggml_init_params params{
            1024ull * 1024ull * 1024ull, nullptr, true};
        ctx_ = ggml_init(params);
        if (ctx_ == nullptr) {
            throw std::runtime_error(
                "failed to initialize GLM-TTS Flow graph context");
        }
        core::ModuleBuildContext ctx{
            ctx_, "glm_tts.flow.estimator", execution_context_.backend_type()};
        x_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            TensorShape::from_dims({batch, frames, config_.mel_dim}));
        mel_condition_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            TensorShape::from_dims({batch, frames, config_.mel_dim}));
        tokens_ = core::make_tensor(
            ctx,
            GGML_TYPE_I32,
            TensorShape::from_dims({batch, frames}));
        speaker_ = core::make_tensor(
            ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, 192}));
        time_features_ = core::make_tensor(
            ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, 256}));
        text_position_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            TensorShape::from_dims(
                {batch, frames, config_.speech_token_dim}));
        positions_ = core::make_tensor(
            ctx, GGML_TYPE_I32, TensorShape::from_dims({frames}));
        ggml_set_input(x_.tensor);
        ggml_set_input(mel_condition_.tensor);
        ggml_set_input(tokens_.tensor);
        ggml_set_input(speaker_.tensor);
        ggml_set_input(time_features_.tensor);
        ggml_set_input(text_position_.tensor);
        ggml_set_input(positions_.tensor);
        auto output = build_estimator(
            ctx,
            x_,
            mel_condition_,
            tokens_,
            speaker_,
            time_features_,
            text_position_,
            positions_,
            weights_,
            config_);
        output_ = output.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_, 262144, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(
                execution_context_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            release_graph();
            throw std::runtime_error(
                "failed to allocate GLM-TTS Flow graph memory");
        }
        text_position_values_ =
            make_text_position_embedding(
                batch, frames, config_.speech_token_dim);
        ggml_backend_tensor_set(
            text_position_.tensor,
            text_position_values_.data(),
            0,
            text_position_values_.size() * sizeof(float));
        position_values_.assign(static_cast<size_t>(frames), 0);
        for (int64_t frame = 0; frame < frames; ++frame) {
            position_values_[static_cast<size_t>(frame)] =
                static_cast<int32_t>(frame);
        }
        ggml_backend_tensor_set(
            positions_.tensor,
            position_values_.data(),
            0,
            position_values_.size() * sizeof(int32_t));
        batch_ = batch;
        frames_ = frames;
    }

    core::ExecutionContext & execution_context_;
    FlowWeights weights_;
    GlmTTSFlowConfig config_;
    std::mutex mutex_;
    ggml_context * ctx_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    TensorValue x_;
    TensorValue mel_condition_;
    TensorValue tokens_;
    TensorValue speaker_;
    TensorValue time_features_;
    TensorValue text_position_;
    TensorValue positions_;
    std::vector<float> text_position_values_;
    std::vector<int32_t> position_values_;
    ggml_tensor * output_ = nullptr;
    int64_t batch_ = 0;
    int64_t frames_ = 0;
};

}  // namespace

struct GlmTTSFlowRuntime::State {
    std::shared_ptr<core::ExecutionContext> execution_context;
    std::shared_ptr<core::BackendWeightStore> store;
    std::unique_ptr<FlowRunner> runner;
};

GlmTTSFlowRuntime::GlmTTSFlowRuntime(
    std::shared_ptr<const assets::TensorSource> source,
    core::BackendConfig backend,
    assets::TensorStorageType storage_type,
    GlmTTSFlowConfig config)
    : config_(std::move(config)),
      state_(std::make_shared<State>()) {
    if (source == nullptr) {
        throw std::runtime_error("GLM-TTS Flow requires weights");
    }
    if (config_.mel_dim <= 0 ||
        config_.speech_token_dim <= 0 ||
        config_.trans_dim <= 0 ||
        config_.depth <= 0 ||
        config_.heads <= 0 ||
        config_.dim_head <= 0 ||
        config_.heads * config_.dim_head != config_.trans_dim ||
        config_.trans_dim % 16 != 0) {
        throw std::runtime_error("GLM-TTS Flow config is invalid");
    }
    state_->execution_context =
        std::make_shared<core::ExecutionContext>(backend);
    state_->store = std::make_shared<core::BackendWeightStore>(
        state_->execution_context->backend(),
        state_->execution_context->backend_type(),
        "glm_tts.flow.weights",
        1024ull * 1024ull * 1024ull);
    auto weights = load_flow_weights(
        *state_->store, *source, config_, storage_type);
    state_->store->upload();
    state_->runner = std::make_unique<FlowRunner>(
        *state_->execution_context, std::move(weights), config_);
}

GlmTTSFlowRuntime::~GlmTTSFlowRuntime() = default;
GlmTTSFlowRuntime::GlmTTSFlowRuntime(GlmTTSFlowRuntime &&) noexcept = default;
GlmTTSFlowRuntime & GlmTTSFlowRuntime::operator=(
    GlmTTSFlowRuntime &&) noexcept = default;

GlmTTSFlowOutput GlmTTSFlowRuntime::generate(
    const GlmTTSFlowInput & input) const {
    if (state_ == nullptr || state_->runner == nullptr) {
        throw std::runtime_error("GLM-TTS Flow is not initialized");
    }
    if (input.speech_tokens.empty()) {
        throw std::runtime_error("GLM-TTS Flow requires speech tokens");
    }
    if (input.prompt_frames < 0 ||
        static_cast<int64_t>(input.prompt_mel.size()) !=
            input.prompt_frames * config_.mel_dim) {
        throw std::runtime_error(
            "GLM-TTS Flow prompt mel shape mismatch");
    }
    if (input.speaker_embedding.size() != 192) {
        throw std::runtime_error(
            "GLM-TTS Flow requires a 192-value speaker embedding");
    }
    if (input.inference_steps <= 0) {
        throw std::runtime_error(
            "GLM-TTS Flow inference steps must be positive");
    }
    const int64_t frames = static_cast<int64_t>(
        static_cast<double>(input.speech_tokens.size()) /
        static_cast<double>(config_.input_frame_rate) *
        static_cast<double>(config_.mel_framerate));
    if (frames <= 0 || input.prompt_frames > frames) {
        throw std::runtime_error(
            "GLM-TTS Flow computed an invalid mel length");
    }
    const int64_t row_values = frames * config_.mel_dim;
    if (static_cast<int64_t>(input.initial_noise.size()) != row_values) {
        throw std::runtime_error(
            "GLM-TTS Flow initial noise shape mismatch");
    }
    constexpr int64_t kBatch = 2;
    auto x = repeat_row(input.initial_noise, kBatch);
    std::vector<float> condition(
        static_cast<size_t>(kBatch * row_values), 0.0F);
    std::copy(
        input.prompt_mel.begin(),
        input.prompt_mel.end(),
        condition.begin());
    auto speaker = repeat_row(input.speaker_embedding, kBatch);
    std::fill(
        speaker.begin() + 192,
        speaker.end(),
        0.0F);
    for (int64_t b = 0; b < kBatch; ++b) {
        float norm_sq = 0.0F;
        for (int64_t i = 0; i < 192; ++i) {
            const float value =
                speaker[static_cast<size_t>(b * 192 + i)];
            norm_sq += value * value;
        }
        const float inverse =
            norm_sq > 0.0F ? 1.0F / std::sqrt(norm_sq) : 0.0F;
        for (int64_t i = 0; i < 192; ++i) {
            speaker[static_cast<size_t>(b * 192 + i)] *= inverse;
        }
    }
    const auto tokens =
        interpolate_tokens(input.speech_tokens, kBatch, frames);
    std::vector<float> schedule(
        static_cast<size_t>(input.inference_steps + 1), 0.0F);
    for (int step = 0; step <= input.inference_steps; ++step) {
        const float base =
            static_cast<float>(step) /
            static_cast<float>(input.inference_steps);
        schedule[static_cast<size_t>(step)] =
            1.0F -
            std::cos(base * static_cast<float>(M_PI) * 0.5F);
    }
    float timestep = schedule.front();
    float dt = schedule[1] - schedule[0];
    for (int step = 1; step <= input.inference_steps; ++step) {
        const auto velocity = state_->runner->run(
            x,
            condition,
            tokens,
            speaker,
            timestep,
            kBatch,
            frames);
        if (static_cast<int64_t>(velocity.size()) !=
            kBatch * row_values) {
            throw std::runtime_error(
                "GLM-TTS Flow estimator output shape mismatch");
        }
        for (int64_t i = 0; i < row_values; ++i) {
            const float guided =
                (1.0F + input.cfg_rate) *
                    velocity[static_cast<size_t>(i)] -
                input.cfg_rate *
                    velocity[static_cast<size_t>(row_values + i)];
            const float next = x[static_cast<size_t>(i)] + dt * guided;
            x[static_cast<size_t>(i)] = next;
            x[static_cast<size_t>(row_values + i)] = next;
        }
        timestep += dt;
        if (step < input.inference_steps) {
            dt = schedule[static_cast<size_t>(step + 1)] - timestep;
        }
    }
    GlmTTSFlowOutput out;
    out.frames = frames - input.prompt_frames;
    out.prompt_frames = input.prompt_frames;
    out.mel.resize(
        static_cast<size_t>(out.frames * config_.mel_dim));
    std::copy(
        x.begin() +
            static_cast<std::ptrdiff_t>(
                input.prompt_frames * config_.mel_dim),
        x.begin() +
            static_cast<std::ptrdiff_t>(row_values),
        out.mel.begin());
    return out;
}

void GlmTTSFlowRuntime::release_graph() {
    if (state_ != nullptr && state_->runner != nullptr) {
        state_->runner->release_graph();
    }
}

}  // namespace engine::models::glm_tts
