#include "engine/models/index_tts2/gpt.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/relative_attention.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/optimizations/fast_projection_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/torch_random.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <optional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::index_tts2 {
namespace {

namespace binding = engine::modules::binding;
namespace core = engine::core;
namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

constexpr int64_t kSemanticHidden = 1024;
constexpr int64_t kModelDim = 1280;
constexpr int64_t kConditionDim = 512;
constexpr int64_t kSpeakerConditionLayers = 6;
constexpr int64_t kEmotionConditionLayers = 4;
constexpr int64_t kGptLayers = 24;
constexpr int64_t kGptMlpDim = 5120;
constexpr int64_t kTextTokens = 12001;
constexpr int64_t kMelCodes = 8194;
constexpr int64_t kMelPositions = 1818;
constexpr int64_t kTextPositions = 602;
constexpr int64_t kConditionPosFrames = 5000;
constexpr int64_t kConditionConvKernel = 15;
constexpr int64_t kGptHeads = 20;
constexpr int64_t kGptHeadDim = kModelDim / kGptHeads;
constexpr int64_t kConditionTokens = 34;
constexpr int32_t kStartTextToken = 0;
constexpr int32_t kStopTextToken = 1;
constexpr int32_t kStartMelToken = 8192;
constexpr int32_t kStopMelToken = 8193;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

core::TensorValue div(core::ModuleBuildContext & ctx, const core::TensorValue & lhs, const core::TensorValue & rhs) {
    core::validate_shape(rhs, lhs.shape, "Div rhs");
    return core::wrap_tensor(ggml_div(ctx.ggml, lhs.tensor, rhs.tensor), lhs.shape, GGML_TYPE_F32);
}

core::TensorValue scale(core::ModuleBuildContext & ctx, const core::TensorValue & input, float value) {
    return core::wrap_tensor(ggml_scale(ctx.ggml, input.tensor, value), input.shape, GGML_TYPE_F32);
}

core::TensorValue transpose_btc_bct(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    return modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
}

core::TensorValue build_biased_gpt_projection(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t in_features,
    int64_t out_features,
    const modules::LinearWeights & weights) {
    if (ctx.backend_type != core::BackendType::Cuda) {
        return modules::LinearModule({in_features, out_features, true, GGML_PREC_F32}).build(ctx, input, weights);
    }
    if (out_features % 4 != 0) {
        throw std::runtime_error("IndexTTS2 GPT fast projection requires output features divisible by 4");
    }
    auto projected = modules::FastPackedProjection4Module({in_features, out_features, GGML_PREC_F32})
                         .build(ctx, input, {weights.weight, std::nullopt});
    if (!weights.bias.has_value()) {
        throw std::runtime_error("IndexTTS2 GPT linear bias is missing");
    }
    const auto matrix_shape = core::TensorShape::from_dims({projected.shape.prefix_elements(), out_features});
    auto matrix = core::reshape_tensor(ctx, projected, matrix_shape);
    matrix = core::wrap_tensor(ggml_add(ctx.ggml, matrix.tensor, weights.bias->tensor), matrix_shape, GGML_TYPE_F32);
    return core::reshape_tensor(ctx, matrix, input.shape.with_last_dim(out_features));
}

core::TensorValue repeat_bias(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & bias,
    const core::TensorValue & like,
    int64_t heads,
    int64_t dim) {
    auto view = core::reshape_tensor(ctx, bias, core::TensorShape::from_dims({1, heads, 1, dim}));
    return modules::RepeatModule({like.shape}).build(ctx, view);
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::reshape_tensor(ctx, contiguous, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

core::TensorValue gelu_geglu(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input) {
    const int64_t half = input.shape.last_dim() / 2;
    auto x = modules::SliceModule({static_cast<int>(input.shape.rank - 1), 0, half}).build(ctx, input);
    auto gate = modules::SliceModule({static_cast<int>(input.shape.rank - 1), half, half}).build(ctx, input);
    gate = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, gate);
    return modules::MulModule{}.build(ctx, x, gate);
}

core::TensorValue glu_axis(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int axis) {
    if (axis < 0 || axis >= static_cast<int>(input.shape.rank) || input.shape.dims[static_cast<size_t>(axis)] % 2 != 0) {
        throw std::runtime_error("IndexTTS2 GLU axis shape mismatch");
    }
    const int64_t half = input.shape.dims[static_cast<size_t>(axis)] / 2;
    auto value = modules::SliceModule({axis, 0, half}).build(ctx, input);
    auto gate = modules::SliceModule({axis, half, half}).build(ctx, input);
    gate = modules::SigmoidModule{}.build(ctx, gate);
    return modules::MulModule{}.build(ctx, value, gate);
}

core::TensorValue rms_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & gamma) {
    const auto squared = core::wrap_tensor(ggml_sqr(ctx.ggml, input.tensor), input.shape, GGML_TYPE_F32);
    auto sum = modules::ReduceSumModule({static_cast<int>(input.shape.rank - 1)}).build(ctx, squared);
    sum = core::wrap_tensor(ggml_sqrt(ctx.ggml, sum.tensor), sum.shape, GGML_TYPE_F32);
    auto normed = div(ctx, input, modules::RepeatModule({input.shape}).build(ctx, sum));
    normed = scale(ctx, normed, std::sqrt(static_cast<float>(input.shape.last_dim())));
    auto gamma_view = core::reshape_tensor(ctx, gamma, core::TensorShape::from_dims({1, 1, gamma.shape.dims[0]}));
    return modules::MulModule{}.build(ctx, normed, modules::RepeatModule({input.shape}).build(ctx, gamma_view));
}

core::TensorValue condition_subsample(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const IndexTTS2GptConditionEncoderWeights & weights) {
    const int64_t frames = input.shape.dims[1];
    const int64_t frames_after = (frames - 3) / 2 + 1;
    auto x = core::reshape_tensor(ctx, input, core::TensorShape::from_dims({1, 1, frames, kSemanticHidden}));
    x = modules::Conv2dModule({1, kConditionDim, 3, 3, 2, 2, 0, 0, 1, 1, true}).build(ctx, x, weights.subsampling.conv);
    x = modules::ReluModule{}.build(ctx, x);
    x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    x = core::wrap_tensor(ggml_cont(ctx.ggml, x.tensor), x.shape, x.type);
    x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, frames_after, kConditionDim * ((kSemanticHidden - 1) / 2)}));
    x = modules::LinearModule({kConditionDim * ((kSemanticHidden - 1) / 2), kConditionDim, true})
            .build(ctx, x, weights.subsampling.out);
    return scale(ctx, x, std::sqrt(static_cast<float>(kConditionDim)));
}

core::TensorValue condition_rel_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & pos_emb,
    const IndexTTS2GptConditionLayerWeights & weights,
    int64_t heads) {
    const int64_t dim = kConditionDim / heads;
    auto q = modules::LinearModule({kConditionDim, kConditionDim, true})
                 .build(ctx, input, {weights.self_attn.attention.q_weight, weights.self_attn.attention.q_bias});
    auto k = modules::LinearModule({kConditionDim, kConditionDim, true})
                 .build(ctx, input, {weights.self_attn.attention.k_weight, weights.self_attn.attention.k_bias});
    auto v = modules::LinearModule({kConditionDim, kConditionDim, true})
                 .build(ctx, input, {weights.self_attn.attention.v_weight, weights.self_attn.attention.v_bias});
    auto p = modules::LinearModule({kConditionDim, kConditionDim, false})
                 .build(ctx, pos_emb, {weights.self_attn.pos_weight, std::nullopt});

    q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, q, heads, dim));
    k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, k, heads, dim));
    v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, v, heads, dim));
    p = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, p, heads, dim));

    const auto q_u = modules::AddModule{}.build(ctx, q, repeat_bias(ctx, weights.self_attn.pos_bias_u, q, heads, dim));
    const auto q_v = modules::AddModule{}.build(ctx, q, repeat_bias(ctx, weights.self_attn.pos_bias_v, q, heads, dim));
    const auto k_t = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, k);
    const auto p_t = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, p);
    auto scores = modules::AddModule{}.build(ctx, modules::MatMulModule{}.build(ctx, q_u, k_t), modules::MatMulModule{}.build(ctx, q_v, p_t));
    scores = scale(ctx, scores, 1.0F / std::sqrt(static_cast<float>(dim)));
    auto attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, core::ensure_backend_addressable_layout(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
    auto context = modules::MatMulModule{}.build(ctx, attn, v);
    context = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, context);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], kConditionDim}));
    return modules::LinearModule({kConditionDim, kConditionDim, true})
        .build(ctx, context, {weights.self_attn.attention.out_weight, weights.self_attn.attention.out_bias});
}

core::TensorValue condition_conv_module(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const IndexTTS2GptConditionLayerWeights & weights) {
    auto x = transpose_btc_bct(ctx, input);
    x = modules::Conv1dModule({kConditionDim, 2 * kConditionDim, 1, 1, 0, 1, true}).build(ctx, x, weights.conv_pointwise_in);
    x = glu_axis(ctx, x, 1);
    x = modules::DepthwiseConv1dModule({kConditionDim, kConditionConvKernel, 1, 7, 1, true}).build(ctx, x, weights.conv_depthwise);
    x = transpose_btc_bct(ctx, x);
    x = modules::LayerNormModule({x.shape.last_dim(), 1.0e-5F, true, true}).build(ctx, x, weights.conv_norm);
    x = modules::SiluModule{}.build(ctx, x);
    x = transpose_btc_bct(ctx, x);
    x = modules::Conv1dModule({kConditionDim, kConditionDim, 1, 1, 0, 1, true}).build(ctx, x, weights.conv_pointwise_out);
    return transpose_btc_bct(ctx, x);
}

core::TensorValue condition_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & pos_emb,
    const IndexTTS2GptConditionLayerWeights & weights,
    int64_t heads) {
    auto x = input;
    auto y = modules::LayerNormModule({x.shape.last_dim(), 1.0e-5F, true, true}).build(ctx, x, weights.norm_mha);
    y = condition_rel_attention(ctx, y, pos_emb, weights, heads);
    x = modules::AddModule{}.build(ctx, x, y);

    y = modules::LayerNormModule({x.shape.last_dim(), 1.0e-5F, true, true}).build(ctx, x, weights.norm_conv);
    y = condition_conv_module(ctx, y, weights);
    x = modules::AddModule{}.build(ctx, x, y);

    y = modules::LayerNormModule({x.shape.last_dim(), 1.0e-5F, true, true}).build(ctx, x, weights.norm_ff);
    y = modules::LinearModule({kConditionDim, weights.feed_forward_in.weight.shape.dims[0], true}).build(ctx, y, weights.feed_forward_in);
    y = modules::SiluModule{}.build(ctx, y);
    y = modules::LinearModule({weights.feed_forward_in.weight.shape.dims[0], kConditionDim, true}).build(ctx, y, weights.feed_forward_out);
    x = modules::AddModule{}.build(ctx, x, y);
    return modules::LayerNormModule({x.shape.last_dim(), 1.0e-5F, true, true}).build(ctx, x, weights.norm_final);
}

core::TensorValue perceiver_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & latents,
    const core::TensorValue & context,
    const IndexTTS2PerceiverAttentionWeights & weights,
    int64_t dim,
    int64_t heads,
    int64_t inner) {
    const int64_t head_dim = inner / heads;
    const auto full_context = modules::ConcatModule({1}).build(ctx, latents, context);
    auto q = modules::LinearModule({dim, inner, false}).build(ctx, latents, weights.q);
    auto kv = modules::LinearModule({dim, 2 * inner, false}).build(ctx, full_context, weights.kv);
    auto k = modules::SliceModule({2, 0, inner}).build(ctx, kv);
    auto v = modules::SliceModule({2, inner, inner}).build(ctx, kv);
    q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, q, heads, head_dim));
    k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, k, heads, head_dim));
    v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, v, heads, head_dim));
    auto scores = modules::MatMulModule{}.build(ctx, q, modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, k));
    scores = scale(ctx, scores, 1.0F / std::sqrt(static_cast<float>(head_dim)));
    auto attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, core::ensure_backend_addressable_layout(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
    auto output = modules::MatMulModule{}.build(ctx, attn, v);
    output = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, output);
    output = core::ensure_backend_addressable_layout(ctx, output);
    output = core::reshape_tensor(ctx, output, core::TensorShape::from_dims({latents.shape.dims[0], latents.shape.dims[1], inner}));
    return modules::LinearModule({inner, dim, false}).build(ctx, output, weights.out);
}

core::TensorValue perceiver_ff(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const IndexTTS2PerceiverFeedForwardWeights & weights,
    int64_t dim,
    int64_t ff_in) {
    auto hidden = modules::LinearModule({dim, ff_in, true}).build(ctx, input, weights.in);
    hidden = gelu_geglu(ctx, hidden);
    return modules::LinearModule({ff_in / 2, dim, true}).build(ctx, hidden, weights.out);
}

core::TensorValue perceiver(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const IndexTTS2PerceiverWeights & weights,
    int64_t latents_count,
    int64_t dim,
    int64_t heads,
    int64_t inner,
    int64_t ff_in) {
    auto context = modules::LinearModule({kConditionDim, dim, true}).build(ctx, input, weights.project_context);
    auto latents = modules::RepeatModule({core::TensorShape::from_dims({1, latents_count, dim})})
                       .build(ctx, core::reshape_tensor(ctx, weights.latents, core::TensorShape::from_dims({1, latents_count, dim})));
    for (const auto & layer : weights.layers) {
        latents = modules::AddModule{}.build(ctx, latents, perceiver_attention(ctx, latents, context, layer.attention, dim, heads, inner));
        latents = modules::AddModule{}.build(ctx, latents, perceiver_ff(ctx, latents, layer.feed_forward, dim, ff_in));
    }
    return rms_norm(ctx, latents, weights.norm_gamma);
}

core::TensorValue condition_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const IndexTTS2GptConditionEncoderWeights & encoder,
    const IndexTTS2PerceiverWeights & perceiver_weights,
    int64_t encoder_heads,
    int64_t perceiver_latents,
    int64_t perceiver_dim,
    int64_t perceiver_heads,
    int64_t perceiver_inner,
    int64_t perceiver_ff_in) {
    auto x = condition_subsample(ctx, input, encoder);
    const int64_t frames = x.shape.dims[1];
    auto pos_emb = modules::SliceModule({1, 0, frames}).build(ctx, encoder.subsampling.pos_enc);
    for (const auto & layer : encoder.layers) {
        x = condition_layer(ctx, x, pos_emb, layer, encoder_heads);
    }
    x = modules::LayerNormModule({x.shape.last_dim(), 1.0e-5F, true, true}).build(ctx, x, encoder.after_norm);
    return perceiver(ctx, x, perceiver_weights, perceiver_latents, perceiver_dim, perceiver_heads, perceiver_inner, perceiver_ff_in);
}

engine::modules::LinearWeights load_hf_conv1d_linear(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type,
    int64_t in_features,
    int64_t out_features,
    bool use_bias) {
    engine::modules::LinearWeights weights;
    const auto source_weight = source.require_f32(prefix + ".weight", {in_features, out_features});
    std::vector<float> transposed(static_cast<size_t>(out_features * in_features));
    for (int64_t in = 0; in < in_features; ++in) {
        for (int64_t out = 0; out < out_features; ++out) {
            transposed[static_cast<size_t>(out * in_features + in)] =
                source_weight[static_cast<size_t>(in * out_features + out)];
        }
    }
    weights.weight = store.make_from_f32(
        engine::core::TensorShape::from_dims({out_features, in_features}),
        storage_type,
        std::move(transposed));
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    }
    return weights;
}

engine::modules::LinearWeights load_biasless_linear(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type,
    int64_t out_features,
    int64_t in_features) {
    return binding::linear_from_source(store, source, prefix, storage_type, out_features, in_features, false);
}

engine::modules::RelativeAttentionWeights load_condition_relative_attention(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type,
    int64_t heads) {
    engine::modules::RelativeAttentionWeights weights;
    weights.attention.q_weight = store.load_tensor(source, prefix + ".linear_q.weight", storage_type, {kConditionDim, kConditionDim});
    weights.attention.q_bias = store.load_f32_tensor(source, prefix + ".linear_q.bias", {kConditionDim});
    weights.attention.k_weight = store.load_tensor(source, prefix + ".linear_k.weight", storage_type, {kConditionDim, kConditionDim});
    weights.attention.k_bias = store.load_f32_tensor(source, prefix + ".linear_k.bias", {kConditionDim});
    weights.attention.v_weight = store.load_tensor(source, prefix + ".linear_v.weight", storage_type, {kConditionDim, kConditionDim});
    weights.attention.v_bias = store.load_f32_tensor(source, prefix + ".linear_v.bias", {kConditionDim});
    weights.attention.out_weight = store.load_tensor(source, prefix + ".linear_out.weight", storage_type, {kConditionDim, kConditionDim});
    weights.attention.out_bias = store.load_f32_tensor(source, prefix + ".linear_out.bias", {kConditionDim});
    weights.pos_weight = store.load_tensor(source, prefix + ".linear_pos.weight", storage_type, {kConditionDim, kConditionDim});
    weights.pos_bias_u = store.load_f32_tensor(source, prefix + ".pos_bias_u", {heads, kConditionDim / heads});
    weights.pos_bias_v = store.load_f32_tensor(source, prefix + ".pos_bias_v", {heads, kConditionDim / heads});
    return weights;
}

IndexTTS2GptConditionLayerWeights load_condition_layer(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t linear_units,
    int64_t heads,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    IndexTTS2GptConditionLayerWeights layer;
    layer.norm_ff = binding::norm_from_source(store, source, prefix + ".norm_ff", kConditionDim);
    layer.norm_mha = binding::norm_from_source(store, source, prefix + ".norm_mha", kConditionDim);
    layer.norm_conv = binding::norm_from_source(store, source, prefix + ".norm_conv", kConditionDim);
    layer.norm_final = binding::norm_from_source(store, source, prefix + ".norm_final", kConditionDim);
    layer.feed_forward_in = binding::linear_from_source(
        store,
        source,
        prefix + ".feed_forward.w_1",
        matmul_storage_type,
        linear_units,
        kConditionDim,
        true);
    layer.feed_forward_out = binding::linear_from_source(
        store,
        source,
        prefix + ".feed_forward.w_2",
        matmul_storage_type,
        kConditionDim,
        linear_units,
        true);
    layer.self_attn = load_condition_relative_attention(store, source, prefix + ".self_attn", matmul_storage_type, heads);
    layer.conv_pointwise_in = binding::conv1d_from_source(
        store,
        source,
        prefix + ".conv_module.pointwise_conv1",
        conv_storage_type,
        2 * kConditionDim,
        kConditionDim,
        1,
        true);
    layer.conv_depthwise = binding::depthwise_conv1d_from_source(
        store,
        source,
        prefix + ".conv_module.depthwise_conv",
        conv_storage_type,
        kConditionDim,
        kConditionConvKernel,
        true);
    layer.conv_norm = binding::norm_from_source(store, source, prefix + ".conv_module.norm", kConditionDim);
    layer.conv_pointwise_out = binding::conv1d_from_source(
        store,
        source,
        prefix + ".conv_module.pointwise_conv2",
        conv_storage_type,
        kConditionDim,
        kConditionDim,
        1,
        true);
    return layer;
}

IndexTTS2GptConditionEncoderWeights load_condition_encoder(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t layers,
    int64_t linear_units,
    int64_t heads,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    IndexTTS2GptConditionEncoderWeights encoder;
    encoder.subsampling.conv = binding::conv2d_from_source(
        store,
        source,
        prefix + ".embed.conv.0",
        conv_storage_type,
        kConditionDim,
        1,
        3,
        3,
        true);
    encoder.subsampling.out = binding::linear_from_source(
        store,
        source,
        prefix + ".embed.out.0",
        matmul_storage_type,
        kConditionDim,
        kConditionDim * ((kSemanticHidden - 1) / 2),
        true);
    encoder.subsampling.pos_enc = store.load_f32_tensor(
        source,
        prefix + ".embed.pos_enc.pe",
        {1, kConditionPosFrames, kConditionDim});
    encoder.layers.reserve(static_cast<size_t>(layers));
    for (int64_t i = 0; i < layers; ++i) {
        encoder.layers.push_back(load_condition_layer(
            store,
            source,
            prefix + ".encoders." + std::to_string(i),
            linear_units,
            heads,
            matmul_storage_type,
            conv_storage_type));
    }
    encoder.after_norm = binding::norm_from_source(store, source, prefix + ".after_norm", kConditionDim);
    return encoder;
}

IndexTTS2PerceiverLayerWeights load_perceiver_layer(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t dim,
    int64_t inner,
    int64_t ff_in,
    engine::assets::TensorStorageType storage_type) {
    IndexTTS2PerceiverLayerWeights layer;
    layer.attention.q = load_biasless_linear(store, source, prefix + ".0.to_q", storage_type, inner, dim);
    layer.attention.kv = load_biasless_linear(store, source, prefix + ".0.to_kv", storage_type, inner * 2, dim);
    layer.attention.out = load_biasless_linear(store, source, prefix + ".0.to_out", storage_type, dim, inner);
    layer.feed_forward.in = binding::linear_from_source(store, source, prefix + ".1.0", storage_type, ff_in, dim, true);
    layer.feed_forward.out = binding::linear_from_source(store, source, prefix + ".1.2", storage_type, dim, ff_in / 2, true);
    return layer;
}

IndexTTS2PerceiverWeights load_perceiver(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t latents,
    int64_t dim,
    int64_t context_dim,
    int64_t inner,
    int64_t ff_in,
    engine::assets::TensorStorageType storage_type) {
    IndexTTS2PerceiverWeights weights;
    weights.latents = store.load_f32_tensor(source, prefix + ".latents", {latents, dim});
    weights.project_context = binding::linear_from_source(
        store,
        source,
        prefix + ".proj_context",
        storage_type,
        dim,
        context_dim,
        true);
    weights.layers.reserve(2);
    for (int64_t i = 0; i < 2; ++i) {
        weights.layers.push_back(load_perceiver_layer(
            store,
            source,
            prefix + ".layers." + std::to_string(i),
            dim,
            inner,
            ff_in,
            storage_type));
    }
    weights.norm_gamma = store.load_f32_tensor(source, prefix + ".norm.gamma", {dim});
    return weights;
}

IndexTTS2Gpt2LayerWeights load_gpt2_layer(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    int64_t layer_index,
    engine::assets::TensorStorageType storage_type) {
    const std::string prefix = "gpt.h." + std::to_string(layer_index);
    IndexTTS2Gpt2LayerWeights layer;
    layer.attn_norm = binding::norm_from_source(store, source, prefix + ".ln_1", kModelDim);
    layer.qkv = load_hf_conv1d_linear(store, source, prefix + ".attn.c_attn", storage_type, kModelDim, 3 * kModelDim, true);
    layer.attn_out = load_hf_conv1d_linear(store, source, prefix + ".attn.c_proj", storage_type, kModelDim, kModelDim, true);
    layer.mlp_norm = binding::norm_from_source(store, source, prefix + ".ln_2", kModelDim);
    layer.mlp_in = load_hf_conv1d_linear(store, source, prefix + ".mlp.c_fc", storage_type, kModelDim, kGptMlpDim, true);
    layer.mlp_out = load_hf_conv1d_linear(store, source, prefix + ".mlp.c_proj", storage_type, kGptMlpDim, kModelDim, true);
    return layer;
}

struct Gpt2LayerOutput {
    core::TensorValue output;
    core::TensorValue key;
    core::TensorValue value;
};

core::TensorValue gpt_attention_from_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    const std::optional<core::TensorValue> & attention_mask) {
    if (attention_mask.has_value()) {
        auto q_contiguous = core::ensure_backend_addressable_layout(ctx, q_heads);
        auto * flash = ggml_flash_attn_ext(
            ctx.ggml,
            q_contiguous.tensor,
            k_heads.tensor,
            v_heads.tensor,
            attention_mask->tensor,
            1.0F / std::sqrt(static_cast<float>(kGptHeadDim)),
            0.0F,
            0.0F);
        ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
        return core::wrap_tensor(
            flash,
            core::TensorShape::from_dims({q_contiguous.shape.dims[0], q_contiguous.shape.dims[2], q_contiguous.shape.dims[1], kGptHeadDim}),
            GGML_TYPE_F32);
    }
    auto scores = modules::MatMulModule{}.build(ctx, q_heads, modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, k_heads));
    scores = core::wrap_tensor(
        ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(static_cast<float>(kGptHeadDim))),
        scores.shape,
        GGML_TYPE_F32);
    scores = core::wrap_tensor(ggml_diag_mask_inf(ctx.ggml, scores.tensor, 0), scores.shape, GGML_TYPE_F32);
    scores = core::wrap_tensor(ggml_soft_max(ctx.ggml, core::ensure_backend_addressable_layout(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
    return modules::MatMulModule{}.build(ctx, scores, v_heads);
}

core::TensorValue gpt_mlp(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const IndexTTS2Gpt2LayerWeights & weights) {
    auto hidden = build_biased_gpt_projection(ctx, input, kModelDim, kGptMlpDim, weights.mlp_in);
    hidden = modules::GeluModule({modules::GeluApproximation::Tanh}).build(ctx, hidden);
    return build_biased_gpt_projection(ctx, hidden, kGptMlpDim, kModelDim, weights.mlp_out);
}

Gpt2LayerOutput gpt2_layer_full(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const IndexTTS2Gpt2LayerWeights & weights,
    const std::optional<core::TensorValue> & attention_mask = std::nullopt) {
    auto normed = modules::LayerNormModule({kModelDim, 1.0e-5F, true, true}).build(ctx, input, weights.attn_norm);
    auto qkv = build_biased_gpt_projection(ctx, normed, kModelDim, 3 * kModelDim, weights.qkv);
    auto q = modules::SliceModule({2, 0, kModelDim}).build(ctx, qkv);
    auto k = modules::SliceModule({2, kModelDim, kModelDim}).build(ctx, qkv);
    auto v = modules::SliceModule({2, 2 * kModelDim, kModelDim}).build(ctx, qkv);
    auto k_cache = reshape_heads(ctx, k, kGptHeads, kGptHeadDim);
    auto v_cache = reshape_heads(ctx, v, kGptHeads, kGptHeadDim);
    q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, q, kGptHeads, kGptHeadDim));
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k_cache);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v_cache);
    auto context = gpt_attention_from_heads(ctx, q, k_heads, v_heads, attention_mask);
    context = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, context);
    context = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, context), input.shape);
    auto x = modules::AddModule{}.build(ctx, input, build_biased_gpt_projection(ctx, context, kModelDim, kModelDim, weights.attn_out));
    auto mlp_in = modules::LayerNormModule({kModelDim, 1.0e-5F, true, true}).build(ctx, x, weights.mlp_norm);
    return {modules::AddModule{}.build(ctx, x, gpt_mlp(ctx, mlp_in, weights)), k_cache, v_cache};
}

Gpt2LayerOutput gpt2_layer_cached_tail(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const IndexTTS2Gpt2LayerWeights & weights,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const core::TensorValue & cache_slots,
    const core::TensorValue & attention_mask) {
    if (cache_key.shape.dims[0] != input.shape.dims[0] ||
        cache_value.shape.dims[0] != input.shape.dims[0] ||
        cache_key.shape.dims[1] != cache_value.shape.dims[1] ||
        cache_key.shape.dims[2] != cache_value.shape.dims[2] ||
        cache_key.shape.dims[3] != cache_value.shape.dims[3] ||
        cache_slots.shape.dims[0] != input.shape.dims[0]) {
        throw std::runtime_error("IndexTTS2 GPT cached layer batch cache shape mismatch");
    }
    auto normed = modules::LayerNormModule({kModelDim, 1.0e-5F, true, true}).build(ctx, input, weights.attn_norm);
    auto qkv = build_biased_gpt_projection(ctx, normed, kModelDim, 3 * kModelDim, weights.qkv);
    auto q = modules::SliceModule({2, 0, kModelDim}).build(ctx, qkv);
    auto k = modules::SliceModule({2, kModelDim, kModelDim}).build(ctx, qkv);
    auto v = modules::SliceModule({2, 2 * kModelDim, kModelDim}).build(ctx, qkv);
    q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, q, kGptHeads, kGptHeadDim));
    k = reshape_heads(ctx, k, kGptHeads, kGptHeadDim);
    v = reshape_heads(ctx, v, kGptHeads, kGptHeadDim);

    const modules::FastKVSetRowsModule set_rows;
    auto updated_key = set_rows.build(ctx, cache_key, k, cache_slots);
    auto updated_value = set_rows.build(ctx, cache_value, v, cache_slots);

    auto context = gpt_attention_from_heads(
        ctx,
        q,
        modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, updated_key),
        modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, updated_value),
        attention_mask);
    context = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, context), input.shape);
    auto x = modules::AddModule{}.build(ctx, input, build_biased_gpt_projection(ctx, context, kModelDim, kModelDim, weights.attn_out));
    auto mlp_in = modules::LayerNormModule({kModelDim, 1.0e-5F, true, true}).build(ctx, x, weights.mlp_norm);
    return {modules::AddModule{}.build(ctx, x, gpt_mlp(ctx, mlp_in, weights)), k, v};
}

struct TopPItem {
    size_t index = 0;
    float score = 0.0F;
    float weight = 0.0F;
};

struct SampleScore {
    size_t flat_index = 0;
    float score = 0.0F;
};

struct RankedSample {
    size_t score_index = 0;
    size_t flat_index = 0;
    double rank = 0.0;
};

struct IndexTTS2SamplerWorkspace {
    std::vector<float> scores;
    std::vector<float> top_k_heap;
    std::vector<TopPItem> top_p_items;
    std::vector<uint32_t> seen_tokens;
    uint32_t seen_generation = 1;
    std::vector<size_t> finite_score_indices;
    std::vector<SampleScore> sample_scores;
    std::vector<RankedSample> ranked_samples;
    std::vector<size_t> selected_scores;
};

void apply_repetition_penalty(
    std::vector<float> & logits,
    const std::vector<int32_t> & codes,
    float penalty,
    IndexTTS2SamplerWorkspace & workspace) {
    if (penalty == 1.0F) {
        return;
    }
    if (!(penalty > 0.0F)) {
        throw std::runtime_error("IndexTTS2 GPT repetition_penalty must be positive");
    }
    if (workspace.seen_tokens.size() != logits.size()) {
        workspace.seen_tokens.assign(logits.size(), 0);
        workspace.seen_generation = 1;
    } else if (workspace.seen_generation == 0) {
        std::fill(workspace.seen_tokens.begin(), workspace.seen_tokens.end(), 0);
        workspace.seen_generation = 1;
    }
    const uint32_t generation = workspace.seen_generation++;
    const auto apply_token = [&](int32_t token) {
        if (token < 0 || static_cast<size_t>(token) >= logits.size() || workspace.seen_tokens[static_cast<size_t>(token)] == generation) {
            return;
        }
        workspace.seen_tokens[static_cast<size_t>(token)] = generation;
        float & value = logits[static_cast<size_t>(token)];
        value = value < 0.0F ? value * penalty : value / penalty;
    };
    apply_token(kStopTextToken);
    apply_token(kStartMelToken);
    for (const int32_t token : codes) {
        apply_token(token);
    }
}

float kth_largest_threshold(const std::vector<float> & scores, size_t keep_count, std::vector<float> & heap) {
    heap.clear();
    heap.reserve(keep_count);
    const auto greater = std::greater<float>{};
    for (const float score : scores) {
        if (heap.size() < keep_count) {
            heap.push_back(score);
            std::push_heap(heap.begin(), heap.end(), greater);
        } else if (score > heap.front()) {
            std::pop_heap(heap.begin(), heap.end(), greater);
            heap.back() = score;
            std::push_heap(heap.begin(), heap.end(), greater);
        }
    }
    return heap.front();
}

void index_tts2_log_probs(
    const std::vector<float> & logits,
    const std::vector<int32_t> & codes,
    float repetition_penalty,
    int top_k,
    float top_p,
    float temperature,
    IndexTTS2SamplerWorkspace & workspace) {
    if (!(temperature > 0.0F)) {
        throw std::runtime_error("IndexTTS2 GPT temperature must be positive");
    }
    float max_logit = -std::numeric_limits<float>::infinity();
    for (float logit : logits) {
        max_logit = std::max(max_logit, logit);
    }
    float total = 0.0F;
    for (float logit : logits) {
        total += std::exp(logit - max_logit);
    }
    if (!(total > 0.0F)) {
        throw std::runtime_error("IndexTTS2 GPT sampler invalid logit mass");
    }
    const float log_total = std::log(total);
    auto & scores = workspace.scores;
    scores.resize(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        scores[i] = logits[i] - max_logit - log_total;
    }
    apply_repetition_penalty(scores, codes, repetition_penalty, workspace);
    for (float & score : scores) {
        score /= temperature;
    }
    const size_t min_tokens_to_keep = 2;
    auto & finite_indices = workspace.finite_score_indices;
    finite_indices.clear();
    if (top_k > 0 && static_cast<size_t>(top_k) < scores.size()) {
        const size_t keep_count = std::max(static_cast<size_t>(top_k), min_tokens_to_keep);
        const float threshold = kth_largest_threshold(scores, keep_count, workspace.top_k_heap);
        float max_score = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < scores.size(); ++i) {
            if (scores[i] < threshold) {
                scores[i] = -std::numeric_limits<float>::infinity();
            } else {
                finite_indices.push_back(i);
                max_score = std::max(max_score, scores[i]);
            }
        }
        if (!std::isfinite(max_score)) {
            throw std::runtime_error("IndexTTS2 GPT sampler has no finite score");
        }
    } else {
        finite_indices.reserve(scores.size());
        float max_score = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < scores.size(); ++i) {
            if (std::isfinite(scores[i])) {
                finite_indices.push_back(i);
                max_score = std::max(max_score, scores[i]);
            }
        }
        if (!std::isfinite(max_score)) {
            throw std::runtime_error("IndexTTS2 GPT sampler has no finite score");
        }
    }
    if (top_p > 0.0F && top_p < 1.0F) {
        auto & sorted = workspace.top_p_items;
        sorted.clear();
        sorted.reserve(finite_indices.size());
        float total = 0.0F;
        float max_score = -std::numeric_limits<float>::infinity();
        for (const size_t i : finite_indices) {
            max_score = std::max(max_score, scores[i]);
        }
        for (const size_t i : finite_indices) {
            const float weight = std::exp(scores[i] - max_score);
            sorted.push_back({i, scores[i], weight});
            total += weight;
        }
        if (!(total > 0.0F)) {
            throw std::runtime_error("IndexTTS2 GPT sampler invalid top-p mass");
        }
        std::sort(sorted.begin(), sorted.end(), [](const TopPItem & lhs, const TopPItem & rhs) {
            if (lhs.score == rhs.score) {
                return lhs.index < rhs.index;
            }
            return lhs.score < rhs.score;
        });
        float cumulative = 0.0F;
        const float remove_mass = 1.0F - top_p;
        const size_t keep_from = sorted.size() > min_tokens_to_keep ? sorted.size() - min_tokens_to_keep : 0;
        finite_indices.clear();
        for (size_t i = 0; i < sorted.size(); ++i) {
            cumulative += sorted[i].weight / total;
            if (i < keep_from && cumulative <= remove_mass) {
                scores[sorted[i].index] = -std::numeric_limits<float>::infinity();
            } else {
                finite_indices.push_back(sorted[i].index);
            }
        }
    }
}

void sample_index_tts2_indices(
    const std::vector<SampleScore> & scores,
    size_t total_score_count,
    size_t count,
    uint64_t seed,
    uint64_t step,
    const engine::sampling::TorchCudaSamplingPolicy & policy,
    std::vector<RankedSample> & ranked,
    std::vector<size_t> & selected) {
    float max_score = -std::numeric_limits<float>::infinity();
    for (const auto & score : scores) {
        max_score = std::max(max_score, score.score);
    }
    if (!std::isfinite(max_score)) {
        throw std::runtime_error("IndexTTS2 GPT sampler has no finite beam score");
    }
    ranked.clear();
    ranked.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        const auto & score = scores[i];
        const float probability = std::exp(score.score - max_score);
        const float exponential = engine::sampling::torch_cuda_tensor_iterator_exponential_element(
            seed,
            static_cast<uint64_t>(total_score_count),
            static_cast<uint64_t>(score.flat_index),
            step,
            policy.multiprocessor_count,
            policy.max_threads_per_multiprocessor);
        ranked.push_back({i, score.flat_index, static_cast<double>(probability) / static_cast<double>(exponential)});
    }
    if (ranked.empty()) {
        throw std::runtime_error("IndexTTS2 GPT sampler failed to select beam candidates");
    }
    const size_t keep = std::min(count, ranked.size());
    std::partial_sort(
        ranked.begin(),
        ranked.begin() + static_cast<std::ptrdiff_t>(keep),
        ranked.end(),
        [](const RankedSample & lhs, const RankedSample & rhs) {
            if (lhs.rank == rhs.rank) {
                return lhs.flat_index < rhs.flat_index;
            }
            return lhs.rank > rhs.rank;
        });
    selected.clear();
    selected.reserve(keep);
    for (size_t i = 0; i < keep; ++i) {
        selected.push_back(ranked[i].score_index);
    }
}

}  // namespace

std::shared_ptr<const IndexTTS2GptWeights> load_index_tts2_gpt_weights(
    const IndexTTS2Assets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes) {
    if (assets.gpt_weights == nullptr) {
        throw std::runtime_error("IndexTTS2 GPT requires tensor source");
    }
    auto weights = std::make_shared<IndexTTS2GptWeights>();
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        backend,
        backend_type,
        "index_tts2.gpt.weights",
        weight_context_bytes);

    const auto & source = *assets.gpt_weights;
    weights->speaker_conditioner = load_condition_encoder(
        *weights->store,
        source,
        "conditioning_encoder",
        kSpeakerConditionLayers,
        2048,
        8,
        matmul_storage_type,
        conv_storage_type);
    weights->emotion_conditioner = load_condition_encoder(
        *weights->store,
        source,
        "emo_conditioning_encoder",
        kEmotionConditionLayers,
        1024,
        4,
        matmul_storage_type,
        conv_storage_type);
    weights->speaker_perceiver = load_perceiver(
        *weights->store,
        source,
        "perceiver_encoder",
        32,
        kModelDim,
        kConditionDim,
        512,
        3412,
        matmul_storage_type);
    weights->emotion_perceiver = load_perceiver(
        *weights->store,
        source,
        "emo_perceiver_encoder",
        1,
        kSemanticHidden,
        kConditionDim,
        256,
        2730,
        matmul_storage_type);
    weights->text_embedding = weights->store->load_tensor(
        source,
        "text_embedding.weight",
        matmul_storage_type,
        {kTextTokens, kModelDim});
    weights->mel_embedding = weights->store->load_tensor(
        source,
        "mel_embedding.weight",
        matmul_storage_type,
        {kMelCodes, kModelDim});
    weights->text_pos_embedding = weights->store->load_f32_tensor(
        source,
        "text_pos_embedding.emb.weight",
        {kTextPositions, kModelDim});
    weights->mel_pos_embedding = weights->store->load_f32_tensor(
        source,
        "mel_pos_embedding.emb.weight",
        {kMelPositions, kModelDim});
    weights->emotion_vec_projection = binding::linear_from_source(
        *weights->store,
        source,
        "emovec_layer",
        matmul_storage_type,
        kModelDim,
        kSemanticHidden,
        true);
    weights->emotion_layer = binding::linear_from_source(
        *weights->store,
        source,
        "emo_layer",
        matmul_storage_type,
        kModelDim,
        kModelDim,
        true);
    weights->speed_embedding_values = source.require_f32("speed_emb.weight", {2, kModelDim});
    weights->gpt_layers.reserve(static_cast<size_t>(kGptLayers));
    for (int64_t i = 0; i < kGptLayers; ++i) {
        weights->gpt_layers.push_back(load_gpt2_layer(*weights->store, source, i, matmul_storage_type));
    }
    weights->gpt_final_norm = binding::norm_from_source(*weights->store, source, "gpt.ln_f", kModelDim);
    weights->final_norm = binding::norm_from_source(*weights->store, source, "final_norm", kModelDim);
    weights->mel_head = binding::linear_from_source(
        *weights->store,
        source,
        "mel_head",
        matmul_storage_type,
        kMelCodes,
        kModelDim,
        true);
    weights->text_head = binding::linear_from_source(
        *weights->store,
        source,
        "text_head",
        matmul_storage_type,
        kTextTokens,
        kModelDim,
        true);

    weights->store->upload();
    assets.gpt_weights->release_storage();
    return weights;
}

class IndexTTS2GptRuntime::ConditioningGraph {
public:
    ConditioningGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const IndexTTS2GptWeights> weights,
        int64_t frames,
        bool emotion,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          frames_(frames),
          emotion_(emotion) {
        if (frames_ <= 0) {
            throw std::runtime_error("IndexTTS2 GPT conditioning graph requires positive frame count");
        }
        if (weights_ == nullptr) {
            throw std::runtime_error("IndexTTS2 GPT conditioning graph requires weights");
        }

        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 GPT conditioning graph context");
        }
        ggml_init_params input_params{16ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 GPT conditioning input context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), emotion_ ? "index_tts2.gpt.emo_condition" : "index_tts2.gpt.spk_condition", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{
            input_ctx_.get(),
            emotion_ ? "index_tts2.gpt.emo_condition.inputs" : "index_tts2.gpt.spk_condition.inputs",
            execution_.backend_type()};
        semantic_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, frames_, kSemanticHidden})).tensor;
        ggml_set_input(semantic_);
        auto input = core::wrap_tensor(semantic_, core::TensorShape::from_dims({1, frames_, kSemanticHidden}), GGML_TYPE_F32);
        core::TensorValue out;
        if (emotion_) {
            out = condition_encoder(
                ctx,
                input,
                weights_->emotion_conditioner,
                weights_->emotion_perceiver,
                4,
                1,
                kSemanticHidden,
                4,
                256,
                2730);
        } else {
            out = condition_encoder(
                ctx,
                input,
                weights_->speaker_conditioner,
                weights_->speaker_perceiver,
                8,
                32,
                kModelDim,
                8,
                512,
                3412);
        }
        output_ = core::ensure_backend_addressable_layout(ctx, out).tensor;
        output_frames_ = out.shape.dims[1];
        output_dims_ = out.shape.dims[2];
        ggml_set_output(output_);

        graph_ = ggml_new_graph_custom(ctx_.get(), static_cast<size_t>(std::max<int64_t>(65536, frames_ * 4096 + 8192)), false);
        ggml_build_forward_expand(graph_, output_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate IndexTTS2 GPT conditioning input buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate IndexTTS2 GPT conditioning graph");
        }
        debug::timing_log_scalar(
            emotion_ ? "index_tts2.gpt.emo_condition.graph.build_ms" : "index_tts2.gpt.spk_condition.graph.build_ms",
            engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar(
            emotion_ ? "index_tts2.gpt.emo_condition.frames" : "index_tts2.gpt.spk_condition.frames",
            frames_);
    }

    ~ConditioningGraph() {
        clear_graph();
    }

    int64_t frames() const noexcept {
        return frames_;
    }

    bool emotion() const noexcept {
        return emotion_;
    }

    IndexTTS2GptLatent run(const std::vector<float> & semantic_btc) {
        if (static_cast<int64_t>(semantic_btc.size()) != frames_ * kSemanticHidden) {
            throw std::runtime_error("IndexTTS2 GPT conditioning input value count mismatch");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(semantic_, semantic_btc.data(), 0, semantic_btc.size() * sizeof(float));
        debug::timing_log_scalar(
            emotion_ ? "index_tts2.gpt.emo_condition.input_upload_ms" : "index_tts2.gpt.spk_condition.input_upload_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));

        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        timing_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        debug::timing_log_scalar(
            emotion_ ? "index_tts2.gpt.emo_condition.graph.compute_ms" : "index_tts2.gpt.spk_condition.graph.compute_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("IndexTTS2 GPT conditioning graph compute failed");
        }

        IndexTTS2GptLatent output;
        output.frames = output_frames_;
        output.dims = output_dims_;
        output.values.resize(static_cast<size_t>(output.frames * output.dims));
        timing_start = Clock::now();
        ggml_backend_tensor_get(output_, output.values.data(), 0, output.values.size() * sizeof(float));
        debug::timing_log_scalar(
            emotion_ ? "index_tts2.gpt.emo_condition.output_read_ms" : "index_tts2.gpt.spk_condition.output_read_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));
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
    std::shared_ptr<const IndexTTS2GptWeights> weights_;
    int64_t frames_ = 0;
    bool emotion_ = false;
    int64_t output_frames_ = 0;
    int64_t output_dims_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * semantic_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
	};

class IndexTTS2GptRuntime::EmotionVectorGraph {
public:
    EmotionVectorGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const IndexTTS2GptWeights> weights,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)) {
        if (weights_ == nullptr) {
            throw std::runtime_error("IndexTTS2 GPT emotion vector graph requires weights");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 GPT emotion vector graph context");
        }
        ggml_init_params input_params{16ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 GPT emotion vector input context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "index_tts2.gpt.emotion_vector", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{
            input_ctx_.get(),
            "index_tts2.gpt.emotion_vector.inputs",
            execution_.backend_type()};
        input_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, kSemanticHidden})).tensor;
        ggml_set_input(input_);
        auto x = core::wrap_tensor(input_, core::TensorShape::from_dims({1, kSemanticHidden}), GGML_TYPE_F32);
        x = modules::LinearModule({kSemanticHidden, kModelDim, true, GGML_PREC_F32}).build(ctx, x, weights_->emotion_vec_projection);
        x = modules::LinearModule({kModelDim, kModelDim, true, GGML_PREC_F32}).build(ctx, x, weights_->emotion_layer);
        output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 8192, false);
        ggml_build_forward_expand(graph_, output_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate IndexTTS2 GPT emotion vector input buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate IndexTTS2 GPT emotion vector graph");
        }
        debug::timing_log_scalar("index_tts2.gpt.emotion_vector.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
    }

    ~EmotionVectorGraph() {
        clear_graph();
    }

    std::vector<float> run(const IndexTTS2GptLatent & emotion_conditioning) {
        if (emotion_conditioning.frames != 1 ||
            emotion_conditioning.dims != kSemanticHidden ||
            static_cast<int64_t>(emotion_conditioning.values.size()) != kSemanticHidden) {
            throw std::runtime_error("IndexTTS2 GPT emotion vector input shape mismatch");
        }
        ggml_backend_tensor_set(input_, emotion_conditioning.values.data(), 0, emotion_conditioning.values.size() * sizeof(float));
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("IndexTTS2 GPT emotion vector graph compute failed");
        }
        std::vector<float> out(static_cast<size_t>(kModelDim));
        ggml_backend_tensor_get(output_, out.data(), 0, out.size() * sizeof(float));
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
    std::shared_ptr<const IndexTTS2GptWeights> weights_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
};

struct GptPrefillOutput {
    std::vector<float> logits;
    std::vector<float> latent;
    runtime::TransformerKVState kv_state;
};

class IndexTTS2GptRuntime::PrefillGraph {
public:
    PrefillGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const IndexTTS2GptWeights> weights,
        int64_t text_tokens,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          text_tokens_(text_tokens),
          text_steps_(text_tokens + 2),
          prompt_steps_(kConditionTokens + text_tokens + 3) {
        if (weights_ == nullptr || text_tokens_ <= 0) {
            throw std::runtime_error("IndexTTS2 GPT prefill graph requires weights and text tokens");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 GPT prefill graph context");
        }
        ggml_init_params input_params{16ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 GPT prefill input context");
        }
        ggml_init_params output_params{16ull * 1024ull * 1024ull, nullptr, true};
        output_ctx_.reset(ggml_init(output_params));
        if (output_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 GPT prefill output context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "index_tts2.gpt.prefill", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{
            input_ctx_.get(),
            "index_tts2.gpt.prefill.inputs",
            execution_.backend_type()};
        core::ModuleBuildContext output_ctx{
            output_ctx_.get(),
            "index_tts2.gpt.prefill.outputs",
            execution_.backend_type()};
        conds_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, kConditionTokens, kModelDim})).tensor;
        text_ids_ = ggml_new_tensor_1d(input_ctx_.get(), GGML_TYPE_I32, text_steps_);
        start_mel_id_ = ggml_new_tensor_1d(input_ctx_.get(), GGML_TYPE_I32, 1);
        ggml_set_input(conds_);
        ggml_set_input(text_ids_);
        ggml_set_input(start_mel_id_);
        auto conds = core::wrap_tensor(conds_, core::TensorShape::from_dims({1, kConditionTokens, kModelDim}), GGML_TYPE_F32);
        auto text_ids = core::wrap_tensor(text_ids_, core::TensorShape::from_dims({text_steps_}), GGML_TYPE_I32);
        auto text = modules::EmbeddingModule({kTextTokens, kModelDim}).build(ctx, text_ids, weights_->text_embedding);
        auto text_pos = modules::SliceModule({0, 0, text_steps_}).build(ctx, weights_->text_pos_embedding);
        text = modules::AddModule{}.build(ctx, text, text_pos);
        text = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, text), core::TensorShape::from_dims({1, text_steps_, kModelDim}));
        auto mel_id = core::wrap_tensor(start_mel_id_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto mel = modules::EmbeddingModule({kMelCodes, kModelDim}).build(ctx, mel_id, weights_->mel_embedding);
        auto mel_pos = modules::SliceModule({0, 0, 1}).build(ctx, weights_->mel_pos_embedding);
        mel = modules::AddModule{}.build(ctx, mel, mel_pos);
        mel = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, mel), core::TensorShape::from_dims({1, 1, kModelDim}));
        auto x = modules::ConcatModule({1}).build(ctx, conds, text);
        x = modules::ConcatModule({1}).build(ctx, x, mel);
        graph_ = ggml_new_graph_custom(ctx_.get(), static_cast<size_t>(std::max<int64_t>(65536, prompt_steps_ * 8192)), false);
        for (const auto & layer : weights_->gpt_layers) {
            auto out = gpt2_layer_full(ctx, x, layer);
            x = out.output;
            auto key = core::ensure_backend_addressable_layout(ctx, out.key);
            auto value = core::ensure_backend_addressable_layout(ctx, out.value);
            auto * key_output = core::make_tensor(output_ctx, GGML_TYPE_F32, key.shape).tensor;
            auto * value_output = core::make_tensor(output_ctx, GGML_TYPE_F32, value.shape).tensor;
            keys_.push_back(key_output);
            values_.push_back(value_output);
            ggml_build_forward_expand(graph_, ggml_cpy(ctx_.get(), key.tensor, key_output));
            ggml_build_forward_expand(graph_, ggml_cpy(ctx_.get(), value.tensor, value_output));
        }
        x = modules::LayerNormModule({kModelDim, 1.0e-5F, true, true}).build(ctx, x, weights_->gpt_final_norm);
        x = modules::SliceModule({1, prompt_steps_ - 1, 1}).build(ctx, x);
        x = modules::LayerNormModule({kModelDim, 1.0e-5F, true, true}).build(ctx, x, weights_->final_norm);
        auto latent = core::ensure_backend_addressable_layout(ctx, x);
        auto logits = core::ensure_backend_addressable_layout(
            ctx,
            modules::LinearModule({kModelDim, kMelCodes, true, GGML_PREC_F32}).build(ctx, x, weights_->mel_head));
        latent_ = core::make_tensor(output_ctx, GGML_TYPE_F32, latent.shape).tensor;
        logits_ = core::make_tensor(output_ctx, GGML_TYPE_F32, logits.shape).tensor;
        ggml_set_output(latent_);
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, ggml_cpy(ctx_.get(), latent.tensor, latent_));
        ggml_build_forward_expand(graph_, ggml_cpy(ctx_.get(), logits.tensor, logits_));
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate IndexTTS2 GPT prefill input buffer");
        }
        output_buffer_ = ggml_backend_alloc_ctx_tensors(output_ctx_.get(), execution_.backend());
        if (output_buffer_ == nullptr) {
            clear_graph();
            throw std::runtime_error("failed to allocate IndexTTS2 GPT prefill output buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate IndexTTS2 GPT prefill graph");
        }
        const int32_t start_mel = kStartMelToken;
        ggml_backend_tensor_set(start_mel_id_, &start_mel, 0, sizeof(int32_t));
        debug::timing_log_scalar("index_tts2.gpt.prefill.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("index_tts2.gpt.prefill.prompt_steps", prompt_steps_);
    }

    ~PrefillGraph() {
        clear_graph();
    }

    bool matches(int64_t text_tokens) const noexcept {
        return text_tokens_ == text_tokens;
    }

    int64_t prompt_steps() const noexcept {
        return prompt_steps_;
    }

    GptPrefillOutput run(const std::vector<float> & conds, const std::vector<int32_t> & text_tokens) {
        if (static_cast<int64_t>(conds.size()) != kConditionTokens * kModelDim ||
            static_cast<int64_t>(text_tokens.size()) != text_tokens_) {
            throw std::runtime_error("IndexTTS2 GPT prefill input shape mismatch");
        }
        std::vector<int32_t> ids;
        ids.reserve(static_cast<size_t>(text_steps_));
        ids.push_back(kStartTextToken);
        ids.insert(ids.end(), text_tokens.begin(), text_tokens.end());
        ids.push_back(kStopTextToken);
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(conds_, conds.data(), 0, conds.size() * sizeof(float));
        ggml_backend_tensor_set(text_ids_, ids.data(), 0, ids.size() * sizeof(int32_t));
        debug::timing_log_scalar("index_tts2.gpt.prefill.input_upload_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        timing_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        debug::timing_log_scalar("index_tts2.gpt.prefill.graph.compute_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("IndexTTS2 GPT prefill graph compute failed");
        }
        GptPrefillOutput out;
        out.logits.resize(static_cast<size_t>(kMelCodes));
        out.latent.resize(static_cast<size_t>(kModelDim));
        ggml_backend_tensor_get(logits_, out.logits.data(), 0, out.logits.size() * sizeof(float));
        ggml_backend_tensor_get(latent_, out.latent.data(), 0, out.latent.size() * sizeof(float));
        out.kv_state.current_end = prompt_steps_;
        out.kv_state.layers.resize(keys_.size());
        const size_t layer_values = static_cast<size_t>(prompt_steps_ * kGptHeads * kGptHeadDim);
        for (size_t layer = 0; layer < keys_.size(); ++layer) {
            auto & state = out.kv_state.layers[layer];
            state.valid_steps = prompt_steps_;
            state.key.resize(layer_values);
            state.value.resize(layer_values);
            ggml_backend_tensor_get(keys_[layer], state.key.data(), 0, state.key.size() * sizeof(float));
            ggml_backend_tensor_get(values_[layer], state.value.data(), 0, state.value.size() * sizeof(float));
        }
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
        if (output_buffer_ != nullptr) {
            ggml_backend_buffer_free(output_buffer_);
            output_buffer_ = nullptr;
        }
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const IndexTTS2GptWeights> weights_;
    int64_t text_tokens_ = 0;
    int64_t text_steps_ = 0;
    int64_t prompt_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> output_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * conds_ = nullptr;
    ggml_tensor * text_ids_ = nullptr;
    ggml_tensor * start_mel_id_ = nullptr;
    ggml_tensor * latent_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    ggml_backend_buffer_t output_buffer_ = nullptr;
};

class IndexTTS2GptRuntime::ForwardGraph {
public:
    ForwardGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const IndexTTS2GptWeights> weights,
        int64_t text_tokens,
        int64_t code_count,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          text_tokens_(text_tokens),
          code_count_(code_count),
          text_steps_(text_tokens + 2),
          mel_steps_(code_count + 2) {
        if (weights_ == nullptr || text_tokens_ <= 0 || code_count_ <= 0) {
            throw std::runtime_error("IndexTTS2 GPT forward graph requires weights and positive lengths");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 GPT forward graph context");
        }
        ggml_init_params input_params{16ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 GPT forward input context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "index_tts2.gpt.forward", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{
            input_ctx_.get(),
            "index_tts2.gpt.forward.inputs",
            execution_.backend_type()};
        conds_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, kConditionTokens, kModelDim})).tensor;
        text_ids_ = ggml_new_tensor_1d(input_ctx_.get(), GGML_TYPE_I32, text_steps_);
        mel_ids_ = ggml_new_tensor_1d(input_ctx_.get(), GGML_TYPE_I32, mel_steps_);
        ggml_set_input(conds_);
        ggml_set_input(text_ids_);
        ggml_set_input(mel_ids_);

        auto conds = core::wrap_tensor(conds_, core::TensorShape::from_dims({1, kConditionTokens, kModelDim}), GGML_TYPE_F32);
        auto text_ids = core::wrap_tensor(text_ids_, core::TensorShape::from_dims({text_steps_}), GGML_TYPE_I32);
        auto text = modules::EmbeddingModule({kTextTokens, kModelDim}).build(ctx, text_ids, weights_->text_embedding);
        auto text_pos = modules::SliceModule({0, 0, text_steps_}).build(ctx, weights_->text_pos_embedding);
        text = modules::AddModule{}.build(ctx, text, text_pos);
        text = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, text), core::TensorShape::from_dims({1, text_steps_, kModelDim}));

        auto mel_ids = core::wrap_tensor(mel_ids_, core::TensorShape::from_dims({mel_steps_}), GGML_TYPE_I32);
        auto mel = modules::EmbeddingModule({kMelCodes, kModelDim}).build(ctx, mel_ids, weights_->mel_embedding);
        auto mel_pos = modules::SliceModule({0, 0, mel_steps_}).build(ctx, weights_->mel_pos_embedding);
        mel = modules::AddModule{}.build(ctx, mel, mel_pos);
        mel = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, mel), core::TensorShape::from_dims({1, mel_steps_, kModelDim}));

        auto x = modules::ConcatModule({1}).build(ctx, conds, text);
        x = modules::ConcatModule({1}).build(ctx, x, mel);
        for (const auto & layer : weights_->gpt_layers) {
            auto out = gpt2_layer_full(ctx, x, layer);
            x = out.output;
        }
        x = modules::LayerNormModule({kModelDim, 1.0e-5F, true, true}).build(ctx, x, weights_->gpt_final_norm);
        x = modules::SliceModule({1, kConditionTokens + text_steps_, code_count_}).build(ctx, x);
        x = modules::LayerNormModule({kModelDim, 1.0e-5F, true, true}).build(ctx, x, weights_->final_norm);
        output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        ggml_set_output(output_);

        graph_ = ggml_new_graph_custom(
            ctx_.get(),
            static_cast<size_t>(std::max<int64_t>(65536, (kConditionTokens + text_steps_ + mel_steps_) * 8192)),
            false);
        ggml_build_forward_expand(graph_, output_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate IndexTTS2 GPT forward input buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate IndexTTS2 GPT forward graph");
        }
        debug::timing_log_scalar("index_tts2.gpt.forward.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("index_tts2.gpt.forward.text_steps", text_steps_);
        debug::trace_log_scalar("index_tts2.gpt.forward.mel_steps", mel_steps_);
    }

    ~ForwardGraph() {
        clear_graph();
    }

    bool matches(int64_t text_tokens, int64_t code_count) const noexcept {
        return text_tokens_ == text_tokens && code_count_ == code_count;
    }

    IndexTTS2GptLatent run(
        const std::vector<float> & conds,
        const std::vector<int32_t> & text_tokens,
        const std::vector<int32_t> & codes) {
        if (static_cast<int64_t>(conds.size()) != kConditionTokens * kModelDim ||
            static_cast<int64_t>(text_tokens.size()) != text_tokens_ ||
            static_cast<int64_t>(codes.size()) != code_count_) {
            throw std::runtime_error("IndexTTS2 GPT forward graph input shape mismatch");
        }
        std::vector<int32_t> text_ids;
        text_ids.reserve(static_cast<size_t>(text_steps_));
        text_ids.push_back(kStartTextToken);
        text_ids.insert(text_ids.end(), text_tokens.begin(), text_tokens.end());
        text_ids.push_back(kStopTextToken);

        std::vector<int32_t> mel_ids;
        mel_ids.reserve(static_cast<size_t>(mel_steps_));
        mel_ids.push_back(kStartMelToken);
        mel_ids.insert(mel_ids.end(), codes.begin(), codes.end());
        mel_ids.push_back(kStopMelToken);

        auto timing_start = Clock::now();
        ggml_backend_tensor_set(conds_, conds.data(), 0, conds.size() * sizeof(float));
        ggml_backend_tensor_set(text_ids_, text_ids.data(), 0, text_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(mel_ids_, mel_ids.data(), 0, mel_ids.size() * sizeof(int32_t));
        debug::timing_log_scalar("index_tts2.gpt.forward.input_upload_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));

        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        timing_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        debug::timing_log_scalar("index_tts2.gpt.forward.graph.compute_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("IndexTTS2 GPT forward graph compute failed");
        }

        IndexTTS2GptLatent out;
        out.frames = code_count_;
        out.dims = kModelDim;
        out.values.resize(static_cast<size_t>(out.frames * out.dims));
        timing_start = Clock::now();
        ggml_backend_tensor_get(output_, out.values.data(), 0, out.values.size() * sizeof(float));
        debug::timing_log_scalar("index_tts2.gpt.forward.output_read_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
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
    std::shared_ptr<const IndexTTS2GptWeights> weights_;
    int64_t text_tokens_ = 0;
    int64_t code_count_ = 0;
    int64_t text_steps_ = 0;
    int64_t mel_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * conds_ = nullptr;
    ggml_tensor * text_ids_ = nullptr;
    ggml_tensor * mel_ids_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
};

class IndexTTS2GptRuntime::DecodeGraph {
public:
    struct StepOutput {
        std::vector<float> logits;
    };

    struct BatchOutput {
        std::vector<StepOutput> steps;
    };

    DecodeGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const IndexTTS2GptWeights> weights,
        int64_t cache_steps,
        int64_t beam_count,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          cache_steps_(cache_steps),
          beam_count_(beam_count),
          beam_slots_(2 * beam_count) {
        if (weights_ == nullptr || cache_steps_ <= 0 || beam_count_ <= 0) {
            throw std::runtime_error("IndexTTS2 GPT decode graph requires weights and cache steps");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 GPT decode graph context");
        }
        ggml_init_params state_params{256ull * 1024ull * 1024ull, nullptr, true};
        state_ctx_.reset(ggml_init(state_params));
        if (state_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 GPT decode state context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "index_tts2.gpt.decode", execution_.backend_type()};
        core::ModuleBuildContext state_ctx{
            state_ctx_.get(),
            "index_tts2.gpt.decode.state",
            execution_.backend_type()};
        token_ids_ = ggml_new_tensor_1d(state_ctx_.get(), GGML_TYPE_I32, beam_count_);
        mel_positions_ = ggml_new_tensor_1d(state_ctx_.get(), GGML_TYPE_I32, beam_count_);
        cache_slots_ = ggml_new_tensor_1d(state_ctx_.get(), GGML_TYPE_I32, beam_count_);
        attention_mask_ = ggml_new_tensor_4d(state_ctx_.get(), GGML_TYPE_F16, cache_steps_, 1, 1, beam_count_);
        ggml_set_input(token_ids_);
        ggml_set_input(mel_positions_);
        ggml_set_input(cache_slots_);
        ggml_set_input(attention_mask_);
        for (int64_t bank = 0; bank < 2; ++bank) {
            auto & keys = bank_keys_[static_cast<size_t>(bank)];
            auto & values = bank_values_[static_cast<size_t>(bank)];
            keys.reserve(weights_->gpt_layers.size());
            values.reserve(weights_->gpt_layers.size());
            for (size_t layer = 0; layer < weights_->gpt_layers.size(); ++layer) {
                keys.push_back(core::make_tensor(
                    state_ctx,
                    GGML_TYPE_F32,
                    core::TensorShape::from_dims({beam_count_, cache_steps_, kGptHeads, kGptHeadDim})));
                values.push_back(core::make_tensor(
                    state_ctx,
                    GGML_TYPE_F32,
                    core::TensorShape::from_dims({beam_count_, cache_steps_, kGptHeads, kGptHeadDim})));
            }
        }
        build_prefix_views();
        build_bank_graph(ctx, 0);
        build_bank_graph(ctx, 1);
        state_buffer_ = ggml_backend_alloc_ctx_tensors(state_ctx_.get(), execution_.backend());
        if (state_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate IndexTTS2 GPT decode state buffer");
        }
        debug::trace_log_scalar(
            "index_tts2.gpt.decode.state_buffer_mib",
            static_cast<double>(ggml_backend_buffer_get_size(state_buffer_)) / (1024.0 * 1024.0));
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, bank_graphs_[0].graph) ||
            !ggml_gallocr_reserve(gallocr_, bank_graphs_[1].graph) ||
            !ggml_gallocr_alloc_graph(gallocr_, bank_graphs_[0].graph)) {
            clear_graph();
            throw std::runtime_error("failed to allocate IndexTTS2 GPT decode graph");
        }
        attention_mask_values_.assign(static_cast<size_t>(beam_count_ * cache_steps_), ggml_fp32_to_fp16(-INFINITY));
        token_values_.assign(static_cast<size_t>(beam_count_), 0);
        position_values_.assign(static_cast<size_t>(beam_count_), 0);
        cache_slot_values_.assign(static_cast<size_t>(beam_count_), 0);
        debug::timing_log_scalar("index_tts2.gpt.decode.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("index_tts2.gpt.decode.cache_steps", cache_steps_);
        debug::trace_log_scalar("index_tts2.gpt.decode.beam_batch", beam_count_);
    }

    ~DecodeGraph() {
        clear_graph();
    }

    void clear_graph() {
        for (auto & graph : bank_graphs_) {
            if (graph.graph != nullptr) {
                core::release_backend_graph_resources(execution_.backend(), graph.graph);
                graph.graph = nullptr;
            }
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (state_buffer_ != nullptr) {
            ggml_backend_buffer_free(state_buffer_);
            state_buffer_ = nullptr;
        }
    }

    bool can_run(int64_t required_steps, int64_t required_beam_slots) const noexcept {
        return cache_steps_ >= required_steps && beam_slots_ >= required_beam_slots && beam_count_ * 2 == required_beam_slots;
    }

    void initialize_beam_slot(int64_t slot, const runtime::TransformerKVState & state) {
        if (slot < 0 || slot >= beam_slots_) {
            throw std::runtime_error("IndexTTS2 GPT beam slot is out of range");
        }
        if (state.layers.size() != weights_->gpt_layers.size()) {
            throw std::runtime_error("IndexTTS2 GPT beam state layer count mismatch");
        }
        for (size_t layer = 0; layer < state.layers.size(); ++layer) {
            const auto & layer_state = state.layers[layer];
            if (!state.layers.empty() && layer_state.valid_steps != state.layers.front().valid_steps) {
                throw std::runtime_error("IndexTTS2 GPT beam state valid step mismatch");
            }
            ggml_backend_tensor_set(
                beam_key_prefix_views_[static_cast<size_t>(slot)][static_cast<size_t>(layer_state.valid_steps)][layer],
                layer_state.key.data(),
                0,
                layer_state.key.size() * sizeof(float));
            ggml_backend_tensor_set(
                beam_value_prefix_views_[static_cast<size_t>(slot)][static_cast<size_t>(layer_state.valid_steps)][layer],
                layer_state.value.data(),
                0,
                layer_state.value.size() * sizeof(float));
        }
    }

    BatchOutput run_batch_from_beams(
        const std::vector<int64_t> & parent_slots,
        const std::vector<int64_t> & child_slots,
        int64_t valid_steps,
        const std::vector<int32_t> & tokens,
        int32_t mel_position) {
        const size_t active = parent_slots.size();
        if (active == 0 || child_slots.size() != active || tokens.size() != active) {
            throw std::runtime_error("IndexTTS2 GPT batched decode input shape mismatch");
        }
        if (active > static_cast<size_t>(beam_count_)) {
            throw std::runtime_error("IndexTTS2 GPT batched decode exceeds beam batch");
        }
        if (valid_steps >= cache_steps_) {
            throw std::runtime_error("IndexTTS2 GPT decode cache exhausted");
        }
        const int64_t child_bank = child_slots.front() / beam_count_;
        if (child_bank < 0 || child_bank > 1) {
            throw std::runtime_error("IndexTTS2 GPT child beam bank is out of range");
        }
        for (size_t row = 0; row < active; ++row) {
            if (parent_slots[row] < 0 || parent_slots[row] >= beam_slots_ ||
                child_slots[row] < 0 || child_slots[row] >= beam_slots_ ||
                child_slots[row] / beam_count_ != child_bank ||
                child_slots[row] % beam_count_ != static_cast<int64_t>(row)) {
                throw std::runtime_error("IndexTTS2 GPT beam slot layout mismatch");
            }
            for (size_t layer = 0; layer < weights_->gpt_layers.size(); ++layer) {
                if (parent_slots[row] != child_slots[row]) {
                    copy_beam_prefix(parent_slots[row], child_slots[row], valid_steps, layer);
                }
            }
            token_values_[row] = tokens[row];
            position_values_[row] = mel_position;
        }
        for (size_t row = active; row < static_cast<size_t>(beam_count_); ++row) {
            const int64_t child_slot = child_bank * beam_count_ + static_cast<int64_t>(row);
            for (size_t layer = 0; layer < weights_->gpt_layers.size(); ++layer) {
                copy_beam_prefix(child_slots.front(), child_slot, valid_steps, layer);
            }
            token_values_[row] = tokens.front();
            position_values_[row] = mel_position;
        }

        const auto masked = ggml_fp32_to_fp16(-INFINITY);
        const auto visible = ggml_fp32_to_fp16(0.0F);
        std::fill(attention_mask_values_.begin(), attention_mask_values_.end(), masked);
        for (int64_t row = 0; row < beam_count_; ++row) {
            auto * row_values = attention_mask_values_.data() + static_cast<std::ptrdiff_t>(row * cache_steps_);
            for (int64_t step = 0; step <= valid_steps; ++step) {
                row_values[static_cast<size_t>(step)] = visible;
            }
            cache_slot_values_[static_cast<size_t>(row)] = static_cast<int32_t>(row * cache_steps_ + valid_steps);
        }
        ggml_backend_tensor_set(token_ids_, token_values_.data(), 0, token_values_.size() * sizeof(int32_t));
        ggml_backend_tensor_set(mel_positions_, position_values_.data(), 0, position_values_.size() * sizeof(int32_t));
        ggml_backend_tensor_set(cache_slots_, cache_slot_values_.data(), 0, cache_slot_values_.size() * sizeof(int32_t));
        ggml_backend_tensor_set(attention_mask_, attention_mask_values_.data(), 0, attention_mask_values_.size() * sizeof(ggml_fp16_t));

        auto & graph = bank_graphs_[static_cast<size_t>(child_bank)];
        if (!ggml_gallocr_alloc_graph(gallocr_, graph.graph)) {
            throw std::runtime_error("failed to allocate IndexTTS2 GPT decode bank graph");
        }
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph.graph);
        ggml_backend_synchronize(execution_.backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("IndexTTS2 GPT decode graph compute failed");
        }

        std::vector<float> logits(static_cast<size_t>(beam_count_ * kMelCodes));
        ggml_backend_tensor_get(graph.logits, logits.data(), 0, logits.size() * sizeof(float));
        BatchOutput out;
        out.steps.reserve(active);
        for (size_t row = 0; row < active; ++row) {
            StepOutput step;
            step.logits.assign(
                logits.begin() + static_cast<std::ptrdiff_t>(row * static_cast<size_t>(kMelCodes)),
                logits.begin() + static_cast<std::ptrdiff_t>((row + 1) * static_cast<size_t>(kMelCodes)));
            out.steps.push_back(std::move(step));
        }
        return out;
    }

private:
    struct BankGraph {
        ggml_cgraph * graph = nullptr;
        ggml_tensor * logits = nullptr;
    };

    void build_prefix_views() {
        beam_key_prefix_views_.assign(static_cast<size_t>(beam_slots_), {});
        beam_value_prefix_views_.assign(static_cast<size_t>(beam_slots_), {});
        for (int64_t slot = 0; slot < beam_slots_; ++slot) {
            const int64_t bank = slot / beam_count_;
            const int64_t row = slot % beam_count_;
            auto & key_steps = beam_key_prefix_views_[static_cast<size_t>(slot)];
            auto & value_steps = beam_value_prefix_views_[static_cast<size_t>(slot)];
            key_steps.assign(static_cast<size_t>(cache_steps_ + 1), {});
            value_steps.assign(static_cast<size_t>(cache_steps_ + 1), {});
            for (int64_t steps = 1; steps <= cache_steps_; ++steps) {
                auto & key_layers = key_steps[static_cast<size_t>(steps)];
                auto & value_layers = value_steps[static_cast<size_t>(steps)];
                key_layers.reserve(weights_->gpt_layers.size());
                value_layers.reserve(weights_->gpt_layers.size());
                const int64_t elems = steps * kGptHeads * kGptHeadDim;
                const size_t byte_offset = static_cast<size_t>(row * cache_steps_ * kGptHeads * kGptHeadDim) * sizeof(float);
                for (size_t layer = 0; layer < weights_->gpt_layers.size(); ++layer) {
                    key_layers.push_back(ggml_view_1d(state_ctx_.get(), bank_keys_[static_cast<size_t>(bank)][layer].tensor, elems, byte_offset));
                    value_layers.push_back(ggml_view_1d(state_ctx_.get(), bank_values_[static_cast<size_t>(bank)][layer].tensor, elems, byte_offset));
                }
            }
        }
    }

    void copy_beam_prefix(int64_t parent_slot, int64_t child_slot, int64_t valid_steps, size_t layer) {
        if (valid_steps <= 0 || parent_slot == child_slot) {
            return;
        }
        if (valid_steps > cache_steps_) {
            throw std::runtime_error("IndexTTS2 GPT beam prefix copy exceeds cache capacity");
        }
        ggml_backend_tensor_copy(
            beam_key_prefix_views_[static_cast<size_t>(parent_slot)][static_cast<size_t>(valid_steps)][layer],
            beam_key_prefix_views_[static_cast<size_t>(child_slot)][static_cast<size_t>(valid_steps)][layer]);
        ggml_backend_tensor_copy(
            beam_value_prefix_views_[static_cast<size_t>(parent_slot)][static_cast<size_t>(valid_steps)][layer],
            beam_value_prefix_views_[static_cast<size_t>(child_slot)][static_cast<size_t>(valid_steps)][layer]);
    }

    void build_bank_graph(core::ModuleBuildContext & ctx, int64_t bank) {
        BankGraph graph;
        graph.graph = ggml_new_graph_custom(ctx_.get(), 65536, false);
        auto token = core::wrap_tensor(token_ids_, core::TensorShape::from_dims({beam_count_}), GGML_TYPE_I32);
        auto x = modules::EmbeddingModule({kMelCodes, kModelDim}).build(ctx, token, weights_->mel_embedding);
        auto pos = core::wrap_tensor(mel_positions_, core::TensorShape::from_dims({beam_count_}), GGML_TYPE_I32);
        auto pos_emb = modules::EmbeddingModule({kMelPositions, kModelDim}).build(ctx, pos, weights_->mel_pos_embedding);
        x = modules::AddModule{}.build(ctx, x, pos_emb);
        x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), core::TensorShape::from_dims({beam_count_, 1, kModelDim}));
        auto mask = core::wrap_tensor(attention_mask_, core::TensorShape::from_dims({beam_count_, 1, 1, cache_steps_}), GGML_TYPE_F16);
        auto cache_slots = core::wrap_tensor(cache_slots_, core::TensorShape::from_dims({beam_count_}), GGML_TYPE_I32);
        for (size_t layer = 0; layer < weights_->gpt_layers.size(); ++layer) {
            auto out = gpt2_layer_cached_tail(
                ctx,
                x,
                weights_->gpt_layers[layer],
                bank_keys_[static_cast<size_t>(bank)][layer],
                bank_values_[static_cast<size_t>(bank)][layer],
                cache_slots,
                mask);
            x = out.output;
        }
        x = modules::LayerNormModule({kModelDim, 1.0e-5F, true, true}).build(ctx, x, weights_->gpt_final_norm);
        x = modules::LayerNormModule({kModelDim, 1.0e-5F, true, true}).build(ctx, x, weights_->final_norm);
        graph.logits = modules::LinearModule({kModelDim, kMelCodes, true, GGML_PREC_F32}).build(ctx, x, weights_->mel_head).tensor;
        ggml_set_output(graph.logits);
        ggml_build_forward_expand(graph.graph, graph.logits);
        bank_graphs_[static_cast<size_t>(bank)] = graph;
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const IndexTTS2GptWeights> weights_;
    int64_t cache_steps_ = 0;
    int64_t beam_count_ = 0;
    int64_t beam_slots_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> state_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_ids_ = nullptr;
    ggml_tensor * mel_positions_ = nullptr;
    ggml_tensor * cache_slots_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    std::array<BankGraph, 2> bank_graphs_;
    std::array<std::vector<core::TensorValue>, 2> bank_keys_;
    std::array<std::vector<core::TensorValue>, 2> bank_values_;
    std::vector<std::vector<std::vector<ggml_tensor *>>> beam_key_prefix_views_;
    std::vector<std::vector<std::vector<ggml_tensor *>>> beam_value_prefix_views_;
    std::vector<ggml_fp16_t> attention_mask_values_;
    std::vector<int32_t> token_values_;
    std::vector<int32_t> position_values_;
    std::vector<int32_t> cache_slot_values_;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t state_buffer_ = nullptr;
};

IndexTTS2GptRuntime::IndexTTS2GptRuntime(
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
        throw std::runtime_error("IndexTTS2 GPT runtime requires assets");
    }
    if (graph_arena_bytes_ == 0) {
        throw std::runtime_error("IndexTTS2 GPT graph arena must be non-zero");
    }
    weights_ = load_index_tts2_gpt_weights(
        *assets_,
        execution.backend(),
        execution.backend_type(),
        matmul_storage_type,
        conv_storage_type,
        weight_context_bytes);
}

IndexTTS2GptRuntime::~IndexTTS2GptRuntime() = default;

void IndexTTS2GptRuntime::prepare_speaker_conditioning(int64_t frames) {
    if (execution_ == nullptr) {
        throw std::runtime_error("IndexTTS2 GPT runtime execution context is missing");
    }
    if (frames <= 0) {
        throw std::runtime_error("IndexTTS2 GPT speaker conditioning prepare requires positive frames");
    }
    if (speaker_conditioning_graph_ != nullptr && speaker_conditioning_graph_->frames() == frames) {
        return;
    }
    speaker_conditioning_graph_.reset();
    speaker_conditioning_graph_ = std::make_unique<ConditioningGraph>(*execution_, weights_, frames, false, graph_arena_bytes_);
}

void IndexTTS2GptRuntime::prepare_emotion_conditioning(int64_t frames) {
    if (execution_ == nullptr) {
        throw std::runtime_error("IndexTTS2 GPT runtime execution context is missing");
    }
    if (frames <= 0) {
        throw std::runtime_error("IndexTTS2 GPT emotion conditioning prepare requires positive frames");
    }
    if (emotion_conditioning_graph_ != nullptr && emotion_conditioning_graph_->frames() == frames) {
        return;
    }
    emotion_conditioning_graph_.reset();
    emotion_conditioning_graph_ = std::make_unique<ConditioningGraph>(*execution_, weights_, frames, true, graph_arena_bytes_);
}

IndexTTS2GptLatent IndexTTS2GptRuntime::speaker_conditioning(const std::vector<float> & semantic_btc, int64_t frames) {
    if (speaker_conditioning_graph_ == nullptr || speaker_conditioning_graph_->frames() != frames) {
        throw std::runtime_error("IndexTTS2 GPT speaker conditioning graph was not prepared for this reference length");
    }
    return speaker_conditioning_graph_->run(semantic_btc);
}

IndexTTS2GptLatent IndexTTS2GptRuntime::emotion_conditioning(const std::vector<float> & semantic_btc, int64_t frames) {
    if (emotion_conditioning_graph_ == nullptr || emotion_conditioning_graph_->frames() != frames) {
        throw std::runtime_error("IndexTTS2 GPT emotion conditioning graph was not prepared for this reference length");
    }
    return emotion_conditioning_graph_->run(semantic_btc);
}

void IndexTTS2GptRuntime::prepare_generation(int64_t text_tokens, int64_t max_mel_tokens, int64_t num_beams) {
    if (execution_ == nullptr) {
        throw std::runtime_error("IndexTTS2 GPT runtime execution context is missing");
    }
    if (text_tokens <= 0 || max_mel_tokens <= 0) {
        throw std::runtime_error("IndexTTS2 GPT generation prepare requires positive lengths");
    }
    if (num_beams != 1) {
        debug::trace_log_scalar("index_tts2.gpt.generation.num_beams", num_beams);
    }
    if (prefill_graph_ == nullptr || !prefill_graph_->matches(text_tokens)) {
        prefill_graph_.reset();
        prefill_graph_ = std::make_unique<PrefillGraph>(*execution_, weights_, text_tokens, graph_arena_bytes_);
    }
    const int64_t required_cache_steps = prefill_graph_->prompt_steps() + max_mel_tokens + 1;
    const int64_t required_beam_slots = 2 * std::max<int64_t>(1, num_beams);
    if (decode_graph_ == nullptr || !decode_graph_->can_run(required_cache_steps, required_beam_slots)) {
        decode_graph_.reset();
        decode_graph_ = std::make_unique<DecodeGraph>(
            *execution_,
            weights_,
            required_cache_steps,
            std::max<int64_t>(1, num_beams),
            graph_arena_bytes_);
    }
}

std::vector<float> IndexTTS2GptRuntime::project_emotion_vector(const IndexTTS2GptLatent & emotion_conditioning) {
    if (emotion_vector_graph_ == nullptr) {
        emotion_vector_graph_ = std::make_unique<EmotionVectorGraph>(*execution_, weights_, graph_arena_bytes_);
    }
    return emotion_vector_graph_->run(emotion_conditioning);
}

std::vector<float> IndexTTS2GptRuntime::merge_emotion_vector(
    const std::vector<float> & speaker_semantic,
    int64_t speaker_frames,
    const std::vector<float> & emotion_semantic,
    int64_t emotion_frames,
    float alpha) {
    prepare_emotion_conditioning(speaker_frames);
    const auto base_condition = emotion_conditioning(speaker_semantic, speaker_frames);
    prepare_emotion_conditioning(emotion_frames);
    const auto emotion_condition = emotion_conditioning(emotion_semantic, emotion_frames);
    auto base = project_emotion_vector(base_condition);
    const auto emotion = project_emotion_vector(emotion_condition);
    for (size_t i = 0; i < base.size(); ++i) {
        base[i] = base[i] + alpha * (emotion[i] - base[i]);
    }
    return base;
}

IndexTTS2GptGeneration IndexTTS2GptRuntime::generate_speech(const IndexTTS2GptGenerationRequest & request) {
    if (request.text_tokens.empty()) {
        throw std::runtime_error("IndexTTS2 GPT generation requires text tokens");
    }
    prepare_speaker_conditioning(request.speaker_frames);
    const auto speech_conditioning = speaker_conditioning(request.speaker_semantic, request.speaker_frames);
    std::vector<float> emotion_vector = request.emotion_vector;
    if (emotion_vector.empty()) {
        prepare_emotion_conditioning(request.emotion_frames);
        emotion_vector = project_emotion_vector(emotion_conditioning(request.emotion_semantic, request.emotion_frames));
    }
    if (static_cast<int64_t>(emotion_vector.size()) != kModelDim) {
        throw std::runtime_error("IndexTTS2 GPT generation emotion vector shape mismatch");
    }
    prepare_generation(
        static_cast<int64_t>(request.text_tokens.size()),
        request.max_mel_tokens,
        request.num_beams);
    std::vector<float> conds(static_cast<size_t>(kConditionTokens * kModelDim), 0.0F);
    for (int64_t token = 0; token < 32; ++token) {
        for (int64_t dim = 0; dim < kModelDim; ++dim) {
            conds[static_cast<size_t>(token * kModelDim + dim)] =
                speech_conditioning.values[static_cast<size_t>(token * kModelDim + dim)] +
                emotion_vector[static_cast<size_t>(dim)];
        }
    }
    const auto & speed = weights_->speed_embedding_values;
    std::copy_n(speed.data() + static_cast<std::ptrdiff_t>(kModelDim), static_cast<size_t>(kModelDim), conds.data() + static_cast<std::ptrdiff_t>(32 * kModelDim));
    std::copy_n(speed.data(), static_cast<size_t>(kModelDim), conds.data() + static_cast<std::ptrdiff_t>(33 * kModelDim));
    auto prefill = prefill_graph_->run(conds, request.text_tokens);
    const auto sampling_policy = engine::sampling::resolve_torch_cuda_sampling_policy(
        execution_->backend_type(),
        execution_->config().device,
        "index_tts2.gpt.cuda_sampling_policy",
        "IndexTTS2",
        engine::sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault);

    struct Beam {
        std::vector<int32_t> codes;
        std::vector<float> logits;
        int64_t slot = 0;
        int64_t valid_steps = 0;
        int64_t current_end = 0;
        float score = 0.0F;
        bool finished = false;
    };
    auto normalized_score = [&](const Beam & beam) {
        if (request.length_penalty == 0.0F) {
            return beam.score;
        }
        const float length = static_cast<float>(std::max<size_t>(1, beam.codes.size()));
        return beam.score / std::pow(length, request.length_penalty);
    };
    struct BeamCandidate {
        size_t parent = 0;
        int32_t token = 0;
        float score = 0.0F;
        bool finished = false;
    };
    const int beam_count = std::max(1, request.num_beams);
    const int64_t prefill_valid_steps = prefill.kv_state.layers.empty() ? 0 : prefill.kv_state.layers.front().valid_steps;
    std::vector<Beam> beams;
    beams.reserve(static_cast<size_t>(beam_count));
    for (int beam = 0; beam < beam_count; ++beam) {
        decode_graph_->initialize_beam_slot(beam, prefill.kv_state);
        Beam initial;
        initial.logits = prefill.logits;
        initial.slot = beam;
        initial.valid_steps = prefill_valid_steps;
        initial.current_end = prefill.kv_state.current_end;
        initial.score = beam == 0 ? 0.0F : -1.0e9F;
        beams.push_back(std::move(initial));
    }

    std::vector<Beam> completed;
    auto add_completed = [&](Beam beam) {
        completed.push_back(std::move(beam));
        if (static_cast<int>(completed.size()) > beam_count) {
            const auto worst = std::min_element(completed.begin(), completed.end(), [&](const Beam & lhs, const Beam & rhs) {
                return normalized_score(lhs) < normalized_score(rhs);
            });
            completed.erase(worst);
        }
    };
    auto completed_worst_score = [&]() {
        if (completed.empty()) {
            return -std::numeric_limits<float>::infinity();
        }
        const auto worst = std::min_element(completed.begin(), completed.end(), [&](const Beam & lhs, const Beam & rhs) {
            return normalized_score(lhs) < normalized_score(rhs);
        });
        return normalized_score(*worst);
    };
    auto should_stop = [&](float best_sum_logprobs, int64_t cur_generated_len) {
        if (static_cast<int>(completed.size()) < beam_count) {
            return false;
        }
        const float length = static_cast<float>(std::max<int64_t>(1, cur_generated_len));
        const float highest_attainable = request.length_penalty == 0.0F
            ? best_sum_logprobs
            : best_sum_logprobs / std::pow(length, request.length_penalty);
        return completed_worst_score() >= highest_attainable;
    };
    bool first_decode_timing_logged = false;
    int active_bank = 0;
    bool beam_search_done = false;
    uint64_t sample_call_index = 0;
    uint64_t rng_offset_blocks = 0;
    double sampling_ms = 0.0;
    double decode_run_ms = 0.0;
    IndexTTS2SamplerWorkspace sampler_workspace;
    std::vector<BeamCandidate> candidates;
    std::vector<Beam> next_beams;
    std::vector<int64_t> parent_slots;
    std::vector<int64_t> child_slots;
    std::vector<int32_t> next_tokens;
    candidates.reserve(static_cast<size_t>(2 * beam_count));
    next_beams.reserve(static_cast<size_t>(beam_count));
    parent_slots.reserve(static_cast<size_t>(beam_count));
    child_slots.reserve(static_cast<size_t>(beam_count));
    next_tokens.reserve(static_cast<size_t>(beam_count));
    for (int step = 0; step < request.max_mel_tokens && !beams.empty(); ++step) {
        const auto sampling_start = Clock::now();
        candidates.clear();
        sampler_workspace.sample_scores.clear();
        sampler_workspace.sample_scores.reserve(beams.size() * static_cast<size_t>(std::max(request.top_k, 1)));
        for (size_t beam_index = 0; beam_index < beams.size(); ++beam_index) {
            index_tts2_log_probs(
                beams[beam_index].logits,
                beams[beam_index].codes,
                request.repetition_penalty,
                request.top_k,
                request.top_p,
                request.temperature,
                sampler_workspace);
            const size_t beam_offset = beam_index * static_cast<size_t>(kMelCodes);
            for (const size_t token : sampler_workspace.finite_score_indices) {
                const float log_prob = sampler_workspace.scores[token];
                sampler_workspace.sample_scores.push_back({beam_offset + token, beams[beam_index].score + log_prob});
            }
        }
        const size_t flat_score_count = beams.size() * static_cast<size_t>(kMelCodes);
        const size_t keep = std::min<size_t>(static_cast<size_t>(2 * beam_count), flat_score_count);
        if (request.do_sample) {
            rng_offset_blocks += engine::sampling::torch_cuda_tensor_iterator_offset_blocks(
                static_cast<uint64_t>(flat_score_count),
                sampling_policy);
            sample_index_tts2_indices(
                sampler_workspace.sample_scores,
                flat_score_count,
                keep,
                request.seed,
                sample_call_index++,
                sampling_policy,
                sampler_workspace.ranked_samples,
                sampler_workspace.selected_scores);
            std::sort(sampler_workspace.selected_scores.begin(), sampler_workspace.selected_scores.end(), [&](size_t lhs, size_t rhs) {
                const auto & lhs_score = sampler_workspace.sample_scores[lhs];
                const auto & rhs_score = sampler_workspace.sample_scores[rhs];
                if (lhs_score.score == rhs_score.score) {
                    return lhs_score.flat_index < rhs_score.flat_index;
                }
                return lhs_score.score > rhs_score.score;
            });
        } else {
            sampler_workspace.selected_scores.resize(sampler_workspace.sample_scores.size());
            std::iota(sampler_workspace.selected_scores.begin(), sampler_workspace.selected_scores.end(), 0);
            const size_t finite_keep = std::min(keep, sampler_workspace.selected_scores.size());
            std::partial_sort(
                sampler_workspace.selected_scores.begin(),
                sampler_workspace.selected_scores.begin() + static_cast<std::ptrdiff_t>(finite_keep),
                sampler_workspace.selected_scores.end(),
                [&](size_t lhs, size_t rhs) {
                    const auto & lhs_score = sampler_workspace.sample_scores[lhs];
                    const auto & rhs_score = sampler_workspace.sample_scores[rhs];
                    if (lhs_score.score == rhs_score.score) {
                        return lhs_score.flat_index < rhs_score.flat_index;
                    }
                    return lhs_score.score > rhs_score.score;
                });
            sampler_workspace.selected_scores.resize(finite_keep);
        }
        for (size_t rank = 0; rank < sampler_workspace.selected_scores.size(); ++rank) {
            const auto & selected = sampler_workspace.sample_scores[sampler_workspace.selected_scores[rank]];
            const size_t flat_index = selected.flat_index;
            const size_t parent = flat_index / static_cast<size_t>(kMelCodes);
            const auto token = static_cast<int32_t>(flat_index % static_cast<size_t>(kMelCodes));
            candidates.push_back({
                parent,
                token,
                selected.score,
                token == kStopMelToken});
        }
        sampling_ms += engine::debug::elapsed_ms(sampling_start, Clock::now());
        next_beams.clear();
        const int next_bank = 1 - active_bank;
        parent_slots.clear();
        child_slots.clear();
        next_tokens.clear();
        for (size_t rank = 0; rank < candidates.size(); ++rank) {
            const auto & candidate = candidates[rank];
            const Beam & parent = beams[candidate.parent];
            Beam next;
            next.codes = parent.codes;
            next.score = candidate.score;
            if (candidate.finished) {
                if (static_cast<int>(rank) < beam_count) {
                    next.finished = true;
                    add_completed(std::move(next));
                }
                continue;
            }
            next.codes.push_back(candidate.token);
            next.slot = static_cast<int64_t>(next_bank * beam_count + static_cast<int>(next_beams.size()));
            next.valid_steps = parent.valid_steps + 1;
            next.current_end = parent.current_end + 1;
            parent_slots.push_back(parent.slot);
            child_slots.push_back(next.slot);
            next_tokens.push_back(candidate.token);
            next_beams.push_back(std::move(next));
            if (static_cast<int>(next_beams.size()) == beam_count) {
                break;
            }
        }
        if (!next_beams.empty()) {
            const auto run_start = Clock::now();
            const int64_t parent_valid_steps = next_beams.front().valid_steps - 1;
            const auto batch_out = decode_graph_->run_batch_from_beams(
                parent_slots,
                child_slots,
                parent_valid_steps,
                next_tokens,
                static_cast<int32_t>(next_beams.front().codes.size() + 1));
            const auto run_ms = engine::debug::elapsed_ms(run_start, Clock::now());
            decode_run_ms += run_ms;
            if (batch_out.steps.size() != next_beams.size()) {
                throw std::runtime_error("IndexTTS2 GPT batched decode output size mismatch");
            }
            for (size_t beam = 0; beam < next_beams.size(); ++beam) {
                next_beams[beam].logits = batch_out.steps[beam].logits;
            }
            if (!first_decode_timing_logged) {
                debug::timing_log_scalar("index_tts2.gpt.decode.first_run_ms", run_ms);
                first_decode_timing_logged = true;
            }
        }
        if (!candidates.empty() && should_stop(candidates.front().score, static_cast<int64_t>(step + 1))) {
            beam_search_done = true;
            break;
        }
        beams.swap(next_beams);
        active_bank = next_bank;
    }
    debug::timing_log_scalar("index_tts2.gpt.sampling_ms", sampling_ms);
    debug::timing_log_scalar("index_tts2.gpt.decode.run_ms", decode_run_ms);
    if (!beam_search_done) {
        for (auto & beam : beams) {
            add_completed(std::move(beam));
        }
    }
    if (completed.empty()) {
        throw std::runtime_error("IndexTTS2 GPT generation produced no beam candidates");
    }
    const auto best = std::max_element(completed.begin(), completed.end(), [&](const Beam & lhs, const Beam & rhs) {
        return normalized_score(lhs) < normalized_score(rhs);
    });
    IndexTTS2GptGeneration out;
    out.speech_conditioning_latent = speech_conditioning;
    out.codes = best->codes;
    out.rng_offset_blocks = rng_offset_blocks;
    bool stop_seen = false;
    for (size_t i = 0; i < out.codes.size(); ++i) {
        if (out.codes[i] == kStopMelToken) {
            stop_seen = true;
        }
    }
    debug::trace_log_scalar("index_tts2.gpt.generated_code_count", static_cast<int64_t>(out.codes.size()));
    debug::trace_log_scalar("index_tts2.gpt.generated_stop_seen", stop_seen);
    return out;
}

IndexTTS2GptLatent IndexTTS2GptRuntime::forward_latent(
    const IndexTTS2GptLatent & speech_conditioning_latent,
    const std::vector<int32_t> & text_tokens,
    const std::vector<int32_t> & codes,
    const std::vector<float> & emotion_semantic,
    int64_t emotion_frames,
    const std::vector<float> & emotion_vector) {
    if (speech_conditioning_latent.frames != 32 ||
        speech_conditioning_latent.dims != kModelDim ||
        static_cast<int64_t>(speech_conditioning_latent.values.size()) != 32 * kModelDim) {
        throw std::runtime_error("IndexTTS2 GPT latent forward speech conditioning shape mismatch");
    }
    if (text_tokens.empty() || codes.empty()) {
        throw std::runtime_error("IndexTTS2 GPT latent forward requires text tokens and generated codes");
    }
    std::vector<float> emo = emotion_vector;
    if (emo.empty()) {
        prepare_emotion_conditioning(emotion_frames);
        emo = project_emotion_vector(emotion_conditioning(emotion_semantic, emotion_frames));
    }
    if (static_cast<int64_t>(emo.size()) != kModelDim) {
        throw std::runtime_error("IndexTTS2 GPT latent forward emotion vector shape mismatch");
    }
    std::vector<float> conds(static_cast<size_t>(kConditionTokens * kModelDim), 0.0F);
    for (int64_t token = 0; token < 32; ++token) {
        for (int64_t dim = 0; dim < kModelDim; ++dim) {
            conds[static_cast<size_t>(token * kModelDim + dim)] =
                speech_conditioning_latent.values[static_cast<size_t>(token * kModelDim + dim)] +
                emo[static_cast<size_t>(dim)];
        }
    }
    const auto & speed = weights_->speed_embedding_values;
    std::copy_n(speed.data() + static_cast<std::ptrdiff_t>(kModelDim), static_cast<size_t>(kModelDim), conds.data() + static_cast<std::ptrdiff_t>(32 * kModelDim));
    std::copy_n(speed.data(), static_cast<size_t>(kModelDim), conds.data() + static_cast<std::ptrdiff_t>(33 * kModelDim));

    const int64_t text_count = static_cast<int64_t>(text_tokens.size());
    const int64_t code_count = static_cast<int64_t>(codes.size());
    if (forward_graph_ == nullptr || !forward_graph_->matches(text_count, code_count)) {
        forward_graph_.reset();
        forward_graph_ = std::make_unique<ForwardGraph>(*execution_, weights_, text_count, code_count, graph_arena_bytes_);
    }
    return forward_graph_->run(conds, text_tokens, codes);
}

void IndexTTS2GptRuntime::release_conditioning_graphs() {
    speaker_conditioning_graph_.reset();
    emotion_conditioning_graph_.reset();
    emotion_vector_graph_.reset();
}

void IndexTTS2GptRuntime::release_generation_graphs() {
    prefill_graph_.reset();
    decode_graph_.reset();
    forward_graph_.reset();
}

}  // namespace engine::models::index_tts2
