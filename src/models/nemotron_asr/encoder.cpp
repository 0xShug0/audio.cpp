#include "engine/models/nemotron_asr/encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention_modules.h"
#include "engine/framework/modules/asr_helpers.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include "../../framework/modules/attention/attention_internal.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <string>
#include <stdexcept>
#include <utility>

namespace engine::models::nemotron_asr {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kEncoderGraphNodes = 2097152;

int64_t causal_conv_output_dim(int64_t input, int64_t kernel, int64_t stride, bool streaming) {
    const int64_t left = streaming ? kernel - stride : kernel - 1;
    const int64_t right = streaming ? 0 : stride - 1;
    return (input + left + right - kernel) / stride + 1;
}

engine::core::TensorValue pad_causal_2d(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t kernel,
    int64_t stride,
    bool streaming) {
    const int64_t freq_left = kernel - 1;
    const int64_t freq_right = stride - 1;
    const int64_t time_left = streaming ? kernel - stride : kernel - 1;
    const int64_t time_right = streaming ? 0 : stride - 1;
    const auto out_shape = engine::core::TensorShape::from_dims({
        input.shape.dims[0],
        input.shape.dims[1],
        input.shape.dims[2] + time_left + time_right,
        input.shape.dims[3] + freq_left + freq_right,
    });
    return engine::core::wrap_tensor(
        ggml_pad_ext(
            ctx.ggml,
            input.tensor,
            static_cast<int>(freq_left),
            static_cast<int>(freq_right),
            static_cast<int>(time_left),
            static_cast<int>(time_right),
            0,
            0,
            0,
            0),
        out_shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue pad_freq_2d(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t kernel,
    int64_t stride) {
    const int64_t freq_left = kernel - 1;
    const int64_t freq_right = stride - 1;
    return engine::core::wrap_tensor(
        ggml_pad_ext(
            ctx.ggml,
            input.tensor,
            static_cast<int>(freq_left),
            static_cast<int>(freq_right),
            0,
            0,
            0,
            0,
            0,
            0),
        engine::core::TensorShape::from_dims({
            input.shape.dims[0],
            input.shape.dims[1],
            input.shape.dims[2],
            input.shape.dims[3] + freq_left + freq_right,
        }),
        GGML_TYPE_F32);
}

engine::core::TensorValue zero_time_prefix_4d(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t frames) {
    auto first = engine::modules::SliceModule({2, 0, 1}).build(ctx, input);
    auto repeated = engine::modules::RepeatModule({engine::core::TensorShape::from_dims({
        input.shape.dims[0],
        input.shape.dims[1],
        frames,
        input.shape.dims[3],
    })}).build(ctx, first);
    auto contiguous = engine::core::wrap_tensor(ggml_cont(ctx.ggml, repeated.tensor), repeated.shape, GGML_TYPE_F32);
    return engine::core::wrap_tensor(ggml_scale(ctx.ggml, contiguous.tensor, 0.0f), repeated.shape, GGML_TYPE_F32);
}

engine::core::TensorValue next_time_cache_4d(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & previous,
    const engine::core::TensorValue & current,
    int64_t cache_frames) {
    if (cache_frames <= 0) {
        return previous;
    }
    const int64_t target = std::min<int64_t>(cache_frames, previous.shape.dims[2] + current.shape.dims[2]);
    if (current.shape.dims[2] >= target) {
        return engine::modules::SliceModule({2, current.shape.dims[2] - target, target}).build(ctx, current);
    }
    const int64_t tail = target - current.shape.dims[2];
    auto prefix = engine::modules::SliceModule({2, previous.shape.dims[2] - tail, tail}).build(ctx, previous);
    return engine::modules::ConcatModule({2}).build(ctx, prefix, current);
}

engine::core::TensorValue next_time_cache_3d(
    engine::core::ModuleBuildContext & ctx,
    const std::optional<engine::core::TensorValue> & previous,
    const engine::core::TensorValue & current,
    int64_t cache_frames) {
    if (cache_frames <= 0) {
        return current;
    }
    if (!previous.has_value()) {
        return current.shape.dims[1] > cache_frames
            ? engine::modules::SliceModule({1, current.shape.dims[1] - cache_frames, cache_frames}).build(ctx, current)
            : current;
    }
    const int64_t target = std::min<int64_t>(cache_frames, previous->shape.dims[1] + current.shape.dims[1]);
    if (current.shape.dims[1] >= target) {
        return engine::modules::SliceModule({1, current.shape.dims[1] - target, target}).build(ctx, current);
    }
    const int64_t tail = target - current.shape.dims[1];
    auto prefix = engine::modules::SliceModule({1, previous->shape.dims[1] - tail, tail}).build(ctx, *previous);
    return engine::modules::ConcatModule({1}).build(ctx, prefix, current);
}

engine::core::TensorValue next_time_cache_3d_axis(
    engine::core::ModuleBuildContext & ctx,
    const std::optional<engine::core::TensorValue> & previous,
    const engine::core::TensorValue & current,
    int64_t cache_frames,
    int axis) {
    if (axis == 1) {
        return next_time_cache_3d(ctx, previous, current, cache_frames);
    }
    if (axis != 2) {
        throw std::runtime_error("Nemotron ASR streaming cache axis must be 1 or 2");
    }
    if (cache_frames <= 0) {
        return current;
    }
    if (!previous.has_value()) {
        return current.shape.dims[2] > cache_frames
            ? engine::modules::SliceModule({2, current.shape.dims[2] - cache_frames, cache_frames}).build(ctx, current)
            : current;
    }
    const int64_t target = std::min<int64_t>(cache_frames, previous->shape.dims[2] + current.shape.dims[2]);
    if (current.shape.dims[2] >= target) {
        return engine::modules::SliceModule({2, current.shape.dims[2] - target, target}).build(ctx, current);
    }
    const int64_t tail = target - current.shape.dims[2];
    auto prefix = engine::modules::SliceModule({2, previous->shape.dims[2] - tail, tail}).build(ctx, *previous);
    return engine::modules::ConcatModule({2}).build(ctx, prefix, current);
}

// Stateless causal dw_striding subsampling: the offline graph and NeMo's
// pad_and_drop_preencoded chunks, which subsample [cache | new] frames afresh.
engine::core::TensorValue build_causal_subsampling(
    engine::core::ModuleBuildContext & ctx,
    engine::core::TensorValue x,
    const NemotronSubsamplingWeights & weights,
    const NemotronEncoderConfig & enc,
    const engine::core::TensorValue & mask1,
    const engine::core::TensorValue & mask2,
    const engine::core::TensorValue & mask3) {
    const int64_t k = enc.subsampling_kernel;
    const int64_t s = enc.subsampling_stride;
    x = pad_causal_2d(ctx, x, k, s, false);
    x = engine::modules::Conv2dModule({1, enc.subsampling_channels, k, k, static_cast<int>(s), static_cast<int>(s), 0, 0, 1, 1, true})
            .build(ctx, x, weights.conv_in);
    x = engine::modules::ReluModule().build(ctx, engine::modules::TimeMask4dModule().build(ctx, x, mask1));

    x = pad_causal_2d(ctx, x, k, s, false);
    x = engine::modules::DepthwiseConv2dModule({enc.subsampling_channels, k, k, static_cast<int>(s), static_cast<int>(s), 0, 0, 1, 1, true})
            .build(ctx, x, {weights.layers[0].depthwise_weight, weights.layers[0].depthwise_bias});
    x = engine::modules::TimeMask4dModule().build(ctx, x, mask2);
    x = engine::modules::Conv2dModule({enc.subsampling_channels, enc.subsampling_channels, 1, 1, 1, 1, 0, 0, 1, 1, true})
            .build(ctx, x, weights.layers[0].pointwise);
    x = engine::modules::ReluModule().build(ctx, engine::modules::TimeMask4dModule().build(ctx, x, mask2));

    x = pad_causal_2d(ctx, x, k, s, false);
    x = engine::modules::DepthwiseConv2dModule({enc.subsampling_channels, k, k, static_cast<int>(s), static_cast<int>(s), 0, 0, 1, 1, true})
            .build(ctx, x, {weights.layers[1].depthwise_weight, weights.layers[1].depthwise_bias});
    x = engine::modules::TimeMask4dModule().build(ctx, x, mask3);
    x = engine::modules::Conv2dModule({enc.subsampling_channels, enc.subsampling_channels, 1, 1, 1, 1, 0, 0, 1, 1, true})
            .build(ctx, x, weights.layers[1].pointwise);
    return engine::modules::ReluModule().build(ctx, engine::modules::TimeMask4dModule().build(ctx, x, mask3));
}

// The prompt input: a one-hot `prompt_id` row for each of `frames` frames.
std::vector<float> prompt_one_hot(int64_t frames, int64_t num_prompts, int64_t prompt_id) {
    std::vector<float> out(static_cast<size_t>(frames * num_prompts), 0.0f);
    for (int64_t t = 0; t < frames; ++t) out[static_cast<size_t>(t * num_prompts + prompt_id)] = 1.0f;
    return out;
}

// One row's values repeated for every batch row.
template <typename T>
std::vector<T> tile_rows(const std::vector<T> & row, int64_t rows) {
    std::vector<T> out;
    out.reserve(row.size() * static_cast<size_t>(rows));
    for (int64_t r = 0; r < rows; ++r) out.insert(out.end(), row.begin(), row.end());
    return out;
}

engine::core::TensorValue pad_causal_1d(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t left) {
    if (left <= 0) {
        return input;
    }
    return engine::core::wrap_tensor(
        ggml_pad_ext(ctx.ggml, input.tensor, static_cast<int>(left), 0, 0, 0, 0, 0, 0, 0),
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], input.shape.dims[2] + left}),
        GGML_TYPE_F32);
}

engine::core::TensorValue build_nemotron_conv_module(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_btc,
    const NemotronEncoderLayerWeights & weights,
    const NemotronEncoderConfig & config,
    const engine::core::TensorValue & keep_mask) {
    auto x = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input_btc);
    x = engine::modules::LinearModule({config.hidden_size, 2 * config.hidden_size, false}).build(
        ctx,
        engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x),
        weights.conv_pointwise1);
    x = engine::modules::GLUModule().build(ctx, x);
    x = engine::modules::MaskingModule().build(ctx, x, keep_mask);
    x = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    x = pad_causal_1d(ctx, x, config.conv_kernel - 1);
    x = engine::modules::DepthwiseConv1dModule({config.hidden_size, config.conv_kernel, 1, 0, 1, false})
            .build(ctx, x, weights.conv_depthwise);
    x = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    x = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true}).build(ctx, x, weights.conv_norm);
    x = engine::modules::SiluModule().build(ctx, x);
    return engine::modules::LinearModule({config.hidden_size, config.hidden_size, false}).build(
        ctx,
        x,
        weights.conv_pointwise2);
}

struct StreamingConvModuleOutputs {
    engine::core::TensorValue output;
    engine::core::TensorValue next_cache;
};

StreamingConvModuleOutputs build_nemotron_streaming_conv_module(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_btc,
    const std::optional<engine::core::TensorValue> & prefix_cache,
    const NemotronEncoderLayerWeights & weights,
    const NemotronEncoderConfig & config,
    const engine::core::TensorValue & keep_mask) {
    auto x = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input_btc);
    x = engine::modules::LinearModule({config.hidden_size, 2 * config.hidden_size, false}).build(
        ctx,
        engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x),
        weights.conv_pointwise1);
    x = engine::modules::GLUModule().build(ctx, x);
    x = engine::modules::MaskingModule().build(ctx, x, keep_mask);
    x = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    x = engine::core::ensure_backend_addressable_layout(ctx, x);

    const int64_t cache_frames = config.conv_kernel - 1;
    const auto next_cache = next_time_cache_3d_axis(ctx, prefix_cache, x, cache_frames, 2);
    auto conv_input = prefix_cache.has_value()
        ? engine::modules::ConcatModule({2}).build(ctx, *prefix_cache, x)
        : pad_causal_1d(ctx, x, cache_frames);
    conv_input = engine::modules::DepthwiseConv1dModule({config.hidden_size, config.conv_kernel, 1, 0, 1, false})
                     .build(ctx, conv_input, weights.conv_depthwise);
    conv_input = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, conv_input);
    conv_input = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true}).build(ctx, conv_input, weights.conv_norm);
    conv_input = engine::modules::SiluModule().build(ctx, conv_input);
    conv_input = engine::modules::LinearModule({config.hidden_size, config.hidden_size, false}).build(
        ctx,
        conv_input,
        weights.conv_pointwise2);
    return {conv_input, next_cache};
}

engine::core::TensorValue build_encoder_layer(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & attention_mask,
    const engine::core::TensorValue & keep_mask,
    const engine::core::TensorValue & projected_pos_emb,
    const NemotronEncoderLayerWeights & weights,
    const NemotronEncoderConfig & config) {
    auto x_norm = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true}).build(ctx, input, weights.norm_feed_forward1);
    auto ff1 = engine::modules::LinearModule({config.hidden_size, config.intermediate_size, false}).build(ctx, x_norm, weights.ff1_linear1);
    ff1 = engine::modules::SiluModule().build(ctx, ff1);
    ff1 = engine::modules::LinearModule({config.intermediate_size, config.hidden_size, false}).build(ctx, ff1, weights.ff1_linear2);
    ff1 = engine::core::wrap_tensor(ggml_scale(ctx.ggml, ff1.tensor, 0.5f), ff1.shape, GGML_TYPE_F32);
    auto x = engine::core::wrap_tensor(ggml_add(ctx.ggml, input.tensor, ff1.tensor), input.shape, GGML_TYPE_F32);

    auto attn_input = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true}).build(ctx, x, weights.norm_self_att);
    auto attn = engine::modules::RelativeSelfAttentionModule({
        config.hidden_size,
        config.heads,
        false,
        -1,
        -1,
        0,
        false,
    }).build(ctx, attn_input, std::nullopt, weights.self_attn, attention_mask, keep_mask, projected_pos_emb);
    x = engine::core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, attn.tensor), x.shape, GGML_TYPE_F32);

    auto conv_input = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true}).build(ctx, x, weights.norm_conv);
    auto conv = build_nemotron_conv_module(ctx, conv_input, weights, config, keep_mask);
    x = engine::core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, conv.tensor), x.shape, GGML_TYPE_F32);

    auto ff2_input = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true}).build(ctx, x, weights.norm_feed_forward2);
    auto ff2 = engine::modules::LinearModule({config.hidden_size, config.intermediate_size, false}).build(ctx, ff2_input, weights.ff2_linear1);
    ff2 = engine::modules::SiluModule().build(ctx, ff2);
    ff2 = engine::modules::LinearModule({config.intermediate_size, config.hidden_size, false}).build(ctx, ff2, weights.ff2_linear2);
    ff2 = engine::core::wrap_tensor(ggml_scale(ctx.ggml, ff2.tensor, 0.5f), ff2.shape, GGML_TYPE_F32);
    x = engine::core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, ff2.tensor), x.shape, GGML_TYPE_F32);
    return engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true}).build(ctx, x, weights.norm_out);
}

struct StreamingLayerOutputs {
    engine::core::TensorValue output;
    engine::core::TensorValue next_key_cache;
    engine::core::TensorValue next_value_cache;
    engine::core::TensorValue next_conv_cache;
};

StreamingLayerOutputs build_projected_cache_streaming_encoder_layer(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & attention_mask,
    const engine::core::TensorValue & keep_mask,
    const engine::core::TensorValue & projected_pos_emb,
    const std::optional<engine::core::TensorValue> & key_cache,
    const std::optional<engine::core::TensorValue> & value_cache,
    const std::optional<engine::core::TensorValue> & conv_cache,
    const NemotronEncoderLayerWeights & weights,
    const NemotronEncoderConfig & config) {
    namespace ai = engine::modules::attention::internal;

    auto x_norm = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true}).build(ctx, input, weights.norm_feed_forward1);
    auto ff1 = engine::modules::LinearModule({config.hidden_size, config.intermediate_size, false}).build(ctx, x_norm, weights.ff1_linear1);
    ff1 = engine::modules::SiluModule().build(ctx, ff1);
    ff1 = engine::modules::LinearModule({config.intermediate_size, config.hidden_size, false}).build(ctx, ff1, weights.ff1_linear2);
    ff1 = engine::core::wrap_tensor(ggml_scale(ctx.ggml, ff1.tensor, 0.5f), ff1.shape, GGML_TYPE_F32);
    auto x = engine::core::wrap_tensor(ggml_add(ctx.ggml, input.tensor, ff1.tensor), input.shape, GGML_TYPE_F32);

    auto attn_input = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true}).build(ctx, x, weights.norm_self_att);
    const int64_t head_dim = config.hidden_size / config.heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto q = engine::modules::LinearModule({config.hidden_size, config.hidden_size, false})
                 .build(ctx, attn_input, engine::modules::LinearWeights{weights.self_attn.attention.q_weight, weights.self_attn.attention.q_bias});
    auto k = engine::modules::LinearModule({config.hidden_size, config.hidden_size, false})
                 .build(ctx, attn_input, engine::modules::LinearWeights{weights.self_attn.attention.k_weight, weights.self_attn.attention.k_bias});
    auto v = engine::modules::LinearModule({config.hidden_size, config.hidden_size, false})
                 .build(ctx, attn_input, engine::modules::LinearWeights{weights.self_attn.attention.v_weight, weights.self_attn.attention.v_bias});
    q = ai::reshape_heads(ctx, q, config.heads, head_dim);
    k = ai::reshape_heads(ctx, k, config.heads, head_dim);
    v = ai::reshape_heads(ctx, v, config.heads, head_dim);

    auto q_heads = ai::permute_tensor(ctx, q, {0, 2, 1, 3});
    auto k_current_heads = ai::permute_tensor(ctx, k, {0, 2, 1, 3});
    auto v_current_heads = ai::permute_tensor(ctx, v, {0, 2, 1, 3});
    auto k_heads = key_cache.has_value() ? engine::modules::ConcatModule({2}).build(ctx, *key_cache, k_current_heads) : k_current_heads;
    auto v_heads = value_cache.has_value() ? engine::modules::ConcatModule({2}).build(ctx, *value_cache, v_current_heads) : v_current_heads;

    auto p = ai::reshape_heads(ctx, projected_pos_emb, config.heads, head_dim);
    auto p_heads = ai::permute_tensor(ctx, p, {0, 2, 1, 3});

    auto q_u = ai::add_attention_bias(ctx, q_heads, weights.self_attn.pos_bias_u, config.heads, head_dim);
    auto q_v = ai::add_attention_bias(ctx, q_heads, weights.self_attn.pos_bias_v, config.heads, head_dim);
    auto matrix_ac = engine::modules::MatMulModule().build(ctx, q_u, ai::permute_tensor(ctx, k_heads, {0, 1, 3, 2}));
    auto matrix_bd = engine::modules::MatMulModule().build(ctx, q_v, ai::permute_tensor(ctx, p_heads, {0, 1, 3, 2}));
    matrix_bd = ai::relative_shift(ctx, matrix_bd);
    matrix_bd = engine::modules::SliceModule({3, 0, k_heads.shape.dims[2]}).build(ctx, matrix_bd);
    auto scores = engine::core::wrap_tensor(ggml_add(ctx.ggml, matrix_ac.tensor, matrix_bd.tensor), matrix_ac.shape, GGML_TYPE_F32);
    auto attn = engine::core::wrap_tensor(
        ggml_soft_max_ext(ctx.ggml, ai::ensure_contiguous_layout(ctx, scores).tensor, attention_mask.tensor, scale, 0.0f),
        scores.shape,
        GGML_TYPE_F32);
    auto context = engine::modules::MatMulModule().build(ctx, attn, v_heads);
    context = ai::permute_tensor(ctx, context, {0, 2, 1, 3});
    context = ai::ensure_contiguous_layout(ctx, context);
    context = engine::core::reshape_tensor(ctx, context, engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.hidden_size}));
    auto attn_output = engine::modules::LinearModule({config.hidden_size, config.hidden_size, false})
                           .build(ctx, context, engine::modules::LinearWeights{weights.self_attn.attention.out_weight, weights.self_attn.attention.out_bias});
    auto next_key_cache = next_time_cache_3d_axis(ctx, key_cache, k_current_heads, config.sliding_window - 1, 2);
    auto next_value_cache = next_time_cache_3d_axis(ctx, value_cache, v_current_heads, config.sliding_window - 1, 2);
    x = engine::core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, attn_output.tensor), x.shape, GGML_TYPE_F32);

    auto conv_input = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true}).build(ctx, x, weights.norm_conv);
    auto conv_outputs = build_nemotron_streaming_conv_module(ctx, conv_input, conv_cache, weights, config, keep_mask);
    x = engine::core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, conv_outputs.output.tensor), x.shape, GGML_TYPE_F32);

    auto ff2_input = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true}).build(ctx, x, weights.norm_feed_forward2);
    auto ff2 = engine::modules::LinearModule({config.hidden_size, config.intermediate_size, false}).build(ctx, ff2_input, weights.ff2_linear1);
    ff2 = engine::modules::SiluModule().build(ctx, ff2);
    ff2 = engine::modules::LinearModule({config.intermediate_size, config.hidden_size, false}).build(ctx, ff2, weights.ff2_linear2);
    ff2 = engine::core::wrap_tensor(ggml_scale(ctx.ggml, ff2.tensor, 0.5f), ff2.shape, GGML_TYPE_F32);
    x = engine::core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, ff2.tensor), x.shape, GGML_TYPE_F32);
    return {
        engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true}).build(ctx, x, weights.norm_out),
        engine::core::wrap_tensor(
            ggml_cont(ctx.ggml, next_key_cache.tensor), next_key_cache.shape, next_key_cache.type),
        engine::core::wrap_tensor(
            ggml_cont(ctx.ggml, next_value_cache.tensor), next_value_cache.shape, next_value_cache.type),
        engine::core::wrap_tensor(
            ggml_cont(ctx.ggml, conv_outputs.next_cache.tensor),
            conv_outputs.next_cache.shape,
            conv_outputs.next_cache.type),
    };
}

std::vector<float> make_relative_positional_encoding(int64_t batch, int64_t hidden, int64_t frames, int64_t max_frames) {
    if (frames > max_frames) {
        throw std::runtime_error("Nemotron ASR encoder relative position frames exceed maximum");
    }
    const int64_t pos_frames = 2 * frames - 1;
    std::vector<float> values(static_cast<size_t>(batch * pos_frames * hidden), 0.0f);
    constexpr long double kBase = 10000.0L;
    const int64_t half_hidden = hidden / 2;
    std::vector<long double> inv_freq(static_cast<size_t>(half_hidden), 0.0L);
    std::vector<long double> step_sin(static_cast<size_t>(half_hidden), 0.0L);
    std::vector<long double> step_cos(static_cast<size_t>(half_hidden), 0.0L);
    for (int64_t i = 0; i < half_hidden; ++i) {
        const long double exponent = static_cast<long double>(2 * i) / static_cast<long double>(hidden);
        inv_freq[static_cast<size_t>(i)] = 1.0L / std::pow(kBase, exponent);
        step_sin[static_cast<size_t>(i)] = std::sin(inv_freq[static_cast<size_t>(i)]);
        step_cos[static_cast<size_t>(i)] = std::cos(inv_freq[static_cast<size_t>(i)]);
    }
    for (int64_t b = 0; b < batch; ++b) {
        std::vector<long double> sin_phase(static_cast<size_t>(half_hidden), 0.0L);
        std::vector<long double> cos_phase(static_cast<size_t>(half_hidden), 0.0L);
        for (int64_t i = 0; i < half_hidden; ++i) {
            const long double phase = static_cast<long double>(frames - 1) * inv_freq[static_cast<size_t>(i)];
            sin_phase[static_cast<size_t>(i)] = std::sin(phase);
            cos_phase[static_cast<size_t>(i)] = std::cos(phase);
        }
        for (int64_t p = 0; p < pos_frames; ++p) {
            for (int64_t i = 0; i < half_hidden; ++i) {
                const size_t dst = static_cast<size_t>((b * pos_frames + p) * hidden + 2 * i);
                const size_t freq = static_cast<size_t>(i);
                values[dst] = static_cast<float>(sin_phase[freq]);
                values[dst + 1] = static_cast<float>(cos_phase[freq]);
                const long double next_sin = sin_phase[freq] * step_cos[freq] - cos_phase[freq] * step_sin[freq];
                const long double next_cos = cos_phase[freq] * step_cos[freq] + sin_phase[freq] * step_sin[freq];
                sin_phase[freq] = next_sin;
                cos_phase[freq] = next_cos;
            }
        }
    }
    return values;
}

// NeMo causal ConvSubsampling streaming cache: subsampling_factor + 1 raw frames,
// and ConformerEncoder.setup_streaming_params drops 1 + (cache - 1) / factor
// encoder frames after pre-encoding them.
int64_t pad_and_drop_cache(int64_t subsampling_factor) {
    return subsampling_factor + 1;
}

int64_t pad_and_drop_drop_frames(int64_t subsampling_factor) {
    return 1 + (pad_and_drop_cache(subsampling_factor) - 1) / subsampling_factor;
}

}  // namespace

// A speaker's pad_and_drop caches. While the speaker's latest caches live in a
// graph row (`slot` points at that row's entry), the tensors here are stale.
struct NemotronEncoderSpeakerCaches {
    ggml_context * ggml = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_tensor *> key;
    std::vector<ggml_tensor *> value;
    std::vector<ggml_tensor *> conv;
    NemotronEncoderSpeakerCaches ** slot = nullptr;
    const void * graph = nullptr;
    int64_t row = -1;

    ~NemotronEncoderSpeakerCaches() {
        if (slot != nullptr) *slot = nullptr;
        if (buffer != nullptr) ggml_backend_buffer_free(buffer);
        if (ggml != nullptr) ggml_free(ggml);
    }
};

struct NemotronEncoderRuntime::Graph {
    int64_t input_frames = 0;
    int64_t feature_dim = 0;
    int64_t encoded_frames = 0;
    int64_t hidden = 0;
    int64_t decoder_hidden = 0;
    int64_t lookahead_tokens = 0;
    int64_t prefix_frames = 0;
    bool streaming = false;
    bool first_chunk = false;
    bool raw_cache = false;
    int64_t rows = 1;
    // raw_cache: per-row 1D views of the cache inputs and next-cache outputs, so a
    // speaker's caches copy device-to-device into and out of its row.
    ggml_context * views = nullptr;
    std::vector<std::vector<ggml_tensor *>> key_in, value_in, conv_in;
    // The graph copies the new caches back into its cache inputs, so a speaker
    // whose row is unchanged between steps needs no copies at all.
    std::vector<NemotronEncoderSpeakerCaches *> resident;
    bool stream_static_inputs_valid = false;
    int64_t stream_static_prompt_id = -1;
    ggml_backend_t backend = nullptr;
    ggml_context * ggml = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_gallocr_t pos_gallocr = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_cgraph * pos_graph = nullptr;
    engine::core::TensorValue input;
    engine::core::TensorValue subsampling_cache0;
    engine::core::TensorValue subsampling_cache1;
    engine::core::TensorValue subsampling_cache2;
    engine::core::TensorValue next_subsampling_cache0;
    engine::core::TensorValue next_subsampling_cache1;
    engine::core::TensorValue next_subsampling_cache2;
    engine::core::TensorValue mask1;
    engine::core::TensorValue mask2;
    engine::core::TensorValue mask3;
    engine::core::TensorValue keep_mask;
    engine::core::TensorValue attention_mask;
    engine::core::TensorValue prompt;
    engine::core::TensorValue pos_emb;
    std::vector<engine::core::TensorValue> projected_pos_emb;
    std::vector<engine::core::TensorValue> projected_pos_emb_computed;
    std::vector<engine::core::TensorValue> attention_key_cache;
    std::vector<engine::core::TensorValue> attention_value_cache;
    std::vector<engine::core::TensorValue> conv_cache;
    std::vector<engine::core::TensorValue> next_attention_key_cache;
    std::vector<engine::core::TensorValue> next_attention_value_cache;
    std::vector<engine::core::TensorValue> next_conv_cache;
    engine::core::TensorValue output;

    ~Graph() {
        if (backend != nullptr) {
            engine::core::release_backend_graph_resources(backend, graph);
            engine::core::release_backend_graph_resources(backend, pos_graph);
        }
        if (pos_gallocr != nullptr) {
            ggml_gallocr_free(pos_gallocr);
        }
        for (auto * caches : resident) {
            if (caches != nullptr) {
                caches->slot = nullptr;
                caches->graph = nullptr;
            }
        }
        if (views != nullptr) {
            ggml_free(views);
        }
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
        if (ggml != nullptr) {
            ggml_free(ggml);
        }
    }
};

NemotronEncoderRuntime::NemotronEncoderRuntime(
    std::shared_ptr<const NemotronASRAssets> assets,
    std::shared_ptr<const NemotronWeights> weights,
    engine::core::ExecutionContext & execution_context,
    size_t graph_arena_bytes)
    : assets_(std::move(assets)),
      weights_(std::move(weights)),
      execution_context_(&execution_context),
      graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr || weights_ == nullptr) {
        throw std::runtime_error("Nemotron ASR encoder requires assets and weights");
    }
}

NemotronEncoderRuntime::~NemotronEncoderRuntime() = default;

const std::vector<float> & NemotronEncoderRuntime::relative_positional_encoding(int64_t frames) {
    auto cached = relative_positional_encoding_cache_.find(frames);
    if (cached != relative_positional_encoding_cache_.end()) {
        return cached->second;
    }
    // Bounded: each entry is (2 * frames - 1) * hidden floats -- tens of MB at
    // conversational lengths -- and the offline path now rebuilds its graph
    // whenever the request size moves, so keeping every size it has ever seen
    // would grow without bound.
    //
    // Eviction is a coarse clear rather than an LRU, which can drop the entry
    // the streaming path reuses on every chunk. That is acceptable because a
    // streaming session asks for one stable key_frames, so on its own it never
    // reaches the bound; only interleaved offline work at four different sizes
    // can evict it, and the cost is one regeneration of a chunk-sized encoding.
    constexpr size_t kMaxCachedPositionalEncodings = 4;
    if (relative_positional_encoding_cache_.size() >= kMaxCachedPositionalEncodings) {
        relative_positional_encoding_cache_.clear();
    }
    auto inserted = relative_positional_encoding_cache_.emplace(
        frames,
        make_relative_positional_encoding(1, assets_->config.encoder.hidden_size, frames, assets_->config.encoder.max_position_embeddings));
    return inserted.first->second;
}

void NemotronEncoderRuntime::ensure_graph(int64_t input_frames, int64_t feature_dim, int64_t lookahead_tokens) {
    if (input_frames <= 0 || feature_dim <= 0) {
        throw std::runtime_error("Nemotron ASR encoder graph requires positive input shape");
    }
    if (graph_ != nullptr &&
        !graph_->streaming &&
        graph_->backend == execution_context_->backend() &&
        engine::modules::asr_graph_capacity_usable(graph_->input_frames, input_frames) &&
        graph_->feature_dim == feature_dim) {
        debug::timing_log_scalar("nemotron_asr.encoder.graph_rebuild_ms", 0.0);
        debug::trace_log_scalar("nemotron_asr.encoder.graph_cache_hit", true);
        return;
    }

    const auto build_start = Clock::now();
    const auto & config = assets_->config;
    const auto & enc = config.encoder;
    const auto & weights = weights_->encoder;
    const bool streaming_graph = false;
    const int64_t k = enc.subsampling_kernel;
    const int64_t s = enc.subsampling_stride;
    const int64_t stage1_frames = causal_conv_output_dim(input_frames, k, s, streaming_graph);
    const int64_t stage1_features = causal_conv_output_dim(feature_dim, k, s, streaming_graph);
    const int64_t stage2_frames = causal_conv_output_dim(stage1_frames, k, s, streaming_graph);
    const int64_t stage2_features = causal_conv_output_dim(stage1_features, k, s, streaming_graph);
    const int64_t stage3_frames = causal_conv_output_dim(stage2_frames, k, s, streaming_graph);
    const int64_t stage3_features = causal_conv_output_dim(stage2_features, k, s, streaming_graph);
    if (stage3_features * enc.subsampling_channels != 4352) {
        throw std::runtime_error("Nemotron ASR encoder subsampling feature shape mismatch");
    }

    auto graph = std::make_unique<Graph>();
    graph->input_frames = input_frames;
    graph->feature_dim = feature_dim;
    graph->encoded_frames = stage3_frames;
    graph->hidden = enc.hidden_size;
    graph->decoder_hidden = config.decoder_hidden_size;
    graph->lookahead_tokens = lookahead_tokens;
    graph->backend = execution_context_->backend();
    ggml_init_params params{graph_arena_bytes_, nullptr, true};
    graph->ggml = ggml_init(params);
    if (graph->ggml == nullptr) {
        throw std::runtime_error("Failed to initialize Nemotron ASR encoder graph context");
    }

    engine::core::ModuleBuildContext ctx{graph->ggml, "nemotron_asr.encoder", execution_context_->backend_type()};
    graph->input = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, input_frames, feature_dim}));
    ggml_set_input(graph->input.tensor);
    graph->mask1 = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({1, stage1_frames}));
    graph->mask2 = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({1, stage2_frames}));
    graph->mask3 = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({1, stage3_frames}));
    graph->keep_mask = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({1, stage3_frames}));
    graph->attention_mask = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({stage3_frames, stage3_frames}));
    graph->prompt = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, stage3_frames, config.num_prompts}));
    graph->pos_emb = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, 2 * stage3_frames - 1, enc.hidden_size}));
    for (auto * tensor : {graph->mask1.tensor, graph->mask2.tensor, graph->mask3.tensor, graph->keep_mask.tensor, graph->attention_mask.tensor, graph->prompt.tensor, graph->pos_emb.tensor}) {
        ggml_set_input(tensor);
    }

    auto x = engine::core::reshape_tensor(ctx, graph->input, engine::core::TensorShape::from_dims({1, 1, input_frames, feature_dim}));
    x = build_causal_subsampling(ctx, x, weights.subsampling, enc, graph->mask1, graph->mask2, graph->mask3);

    x = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    x = engine::core::wrap_tensor(ggml_cont(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);
    x = engine::core::reshape_tensor(ctx, x, engine::core::TensorShape::from_dims({1, stage3_frames, enc.subsampling_channels * stage3_features}));
    x = engine::modules::LinearModule({enc.subsampling_channels * stage3_features, enc.hidden_size, true}).build(ctx, x, weights.subsampling.linear);

    graph->projected_pos_emb.reserve(static_cast<size_t>(enc.layers));
    graph->projected_pos_emb_computed.reserve(static_cast<size_t>(enc.layers));
    for (int64_t layer = 0; layer < enc.layers; ++layer) {
        graph->projected_pos_emb.push_back(engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, 2 * stage3_frames - 1, enc.hidden_size})));
        ggml_set_input(graph->projected_pos_emb.back().tensor);
        ggml_set_output(graph->projected_pos_emb.back().tensor);
        graph->projected_pos_emb_computed.push_back(
            engine::modules::LinearModule({enc.hidden_size, enc.hidden_size, false}).build(
                ctx,
                graph->pos_emb,
                {weights.layers[static_cast<size_t>(layer)].self_attn.pos_weight, std::nullopt}));
        ggml_set_output(graph->projected_pos_emb_computed.back().tensor);
    }

    for (int64_t layer = 0; layer < enc.layers; ++layer) {
        x = build_encoder_layer(
            ctx,
            x,
            graph->attention_mask,
            graph->keep_mask,
            graph->projected_pos_emb[static_cast<size_t>(layer)],
            weights.layers[static_cast<size_t>(layer)],
            enc);
    }

    x = engine::modules::ConcatModule({2}).build(ctx, x, graph->prompt);
    x = engine::modules::LinearModule({enc.hidden_size + config.num_prompts, config.prompt_intermediate_size, true})
            .build(ctx, x, weights.prompt_linear1);
    x = engine::modules::ReluModule().build(ctx, x);
    x = engine::modules::LinearModule({config.prompt_intermediate_size, enc.hidden_size, true})
            .build(ctx, x, weights.prompt_linear2);
    graph->output = engine::modules::LinearModule({enc.hidden_size, config.decoder_hidden_size, true})
                        .build(ctx, x, weights.encoder_projector);
    ggml_set_output(graph->output.tensor);

    graph->pos_graph = ggml_new_graph_custom(graph->ggml, 4096, false);
    for (const auto & projected : graph->projected_pos_emb_computed) {
        ggml_build_forward_expand(graph->pos_graph, projected.tensor);
    }
    graph->graph = ggml_new_graph_custom(graph->ggml, kEncoderGraphNodes, false);
    ggml_build_forward_expand(graph->graph, graph->output.tensor);
    const auto alloc_start = Clock::now();
    graph->gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(graph->backend));
    if (graph->gallocr == nullptr ||
        !ggml_gallocr_reserve(graph->gallocr, graph->graph) ||
        !ggml_gallocr_alloc_graph(graph->gallocr, graph->graph)) {
        throw std::runtime_error("Failed to allocate Nemotron ASR encoder graph tensors");
    }
    graph->pos_gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(graph->backend));
    if (graph->pos_gallocr == nullptr ||
        !ggml_gallocr_reserve(graph->pos_gallocr, graph->pos_graph) ||
        !ggml_gallocr_alloc_graph(graph->pos_gallocr, graph->pos_graph)) {
        throw std::runtime_error("Failed to allocate Nemotron ASR encoder position graph tensors");
    }
    debug::timing_log_scalar("nemotron_asr.encoder.graph_alloc_ms", engine::debug::elapsed_ms(alloc_start, Clock::now()));
    const auto pos_upload_start = Clock::now();
    engine::core::write_tensor_f32(
        graph->pos_emb,
        relative_positional_encoding(stage3_frames));
    debug::timing_log_scalar("nemotron_asr.encoder.pos_upload_ms", engine::debug::elapsed_ms(pos_upload_start, Clock::now()));
    const auto pos_compute_start = Clock::now();
    const auto pos_status = engine::core::compute_backend_graph(execution_context_->backend(), graph->pos_graph, nullptr, "Nemotron ASR encoder pos");
    if (pos_status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Nemotron ASR encoder position graph compute failed");
    }
    debug::timing_log_scalar("nemotron_asr.encoder.pos_compute_ms", engine::debug::elapsed_ms(pos_compute_start, Clock::now()));
    const auto pos_copy_start = Clock::now();
    for (size_t i = 0; i < graph->projected_pos_emb.size(); ++i) {
        ggml_backend_tensor_copy(graph->projected_pos_emb_computed[i].tensor, graph->projected_pos_emb[i].tensor);
    }
    debug::timing_log_scalar("nemotron_asr.encoder.pos_copy_ms", engine::debug::elapsed_ms(pos_copy_start, Clock::now()));
    graph_ = std::move(graph);
    const double build_ms = engine::debug::elapsed_ms(build_start, Clock::now());
    debug::timing_log_scalar("nemotron_asr.encoder.graph_build_ms", build_ms);
    debug::timing_log_scalar("nemotron_asr.encoder.graph_rebuild_ms", build_ms);
    debug::trace_log_scalar("nemotron_asr.encoder.graph_cache_hit", false);
    debug::trace_log_scalar("nemotron_asr.encoder.graph_input_frames", input_frames);
    debug::trace_log_scalar("nemotron_asr.encoder.graph_encoded_frames", stage3_frames);
    debug::trace_log_scalar("nemotron_asr.encoder.graph_lookahead_tokens", lookahead_tokens);
}

NemotronEncoderRuntime::Graph & NemotronEncoderRuntime::ensure_stream_graph(
    int64_t input_frames,
    int64_t feature_dim,
    int64_t lookahead_tokens,
    int64_t prefix_frames,
    bool first_chunk,
    bool raw_cache,
    int64_t rows) {
    if (input_frames <= 0 || feature_dim <= 0 || prefix_frames < 0 || rows <= 0 || (rows > 1 && !raw_cache)) {
        throw std::runtime_error("Nemotron ASR streaming encoder graph requires valid input shape");
    }
    const auto & enc = assets_->config.encoder;
    const int64_t prefix_capacity = enc.sliding_window - 1;
    for (const auto & graph_slot : stream_graphs_) {
        if (graph_slot != nullptr &&
            graph_slot->streaming &&
            graph_slot->backend == execution_context_->backend() &&
            graph_slot->input_frames == input_frames &&
            graph_slot->feature_dim == feature_dim &&
            graph_slot->lookahead_tokens == lookahead_tokens &&
            graph_slot->prefix_frames == prefix_capacity &&
            graph_slot->first_chunk == first_chunk &&
            graph_slot->raw_cache == raw_cache &&
            graph_slot->rows == rows) {
            debug::timing_log_scalar("nemotron_asr.encoder.stream.graph_rebuild_ms", 0.0);
            debug::trace_log_scalar("nemotron_asr.encoder.stream.graph_cache_hit", true);
            return *graph_slot;
        }
    }

    if (raw_cache) {
        // Row counts and final-chunk lengths vary, so bound the pad_and_drop graphs:
        // drop the oldest one that holds no speaker's caches.
        constexpr size_t kMaxRawCacheGraphs = 16;
        const auto raw_graphs = std::count_if(stream_graphs_.begin(), stream_graphs_.end(), [](const auto & slot) {
            return slot != nullptr && slot->raw_cache;
        });
        if (static_cast<size_t>(raw_graphs) >= kMaxRawCacheGraphs) {
            const auto idle = std::find_if(stream_graphs_.begin(), stream_graphs_.end(), [](const auto & slot) {
                return slot != nullptr && slot->raw_cache &&
                    std::all_of(slot->resident.begin(), slot->resident.end(), [](const auto * caches) { return caches == nullptr; });
            });
            if (idle != stream_graphs_.end()) stream_graphs_.erase(idle);
        }
    }

    const auto build_start = Clock::now();
    const auto & config = assets_->config;
    const auto & weights = weights_->encoder;
    const bool streaming_graph = true;
    const int64_t k = enc.subsampling_kernel;
    const int64_t s = enc.subsampling_stride;
    // raw_cache: NeMo pad_and_drop_preencoded runs the causal subsampling
    // statelessly over [cache | new] frames, exactly like the offline graph.
    const bool stream_subsampling = streaming_graph && !raw_cache;
    const bool first_stream_chunk = first_chunk && !raw_cache;
    auto stage_frames = [&](int64_t frames) {
        return first_stream_chunk ? (frames + 2 - k) / s + 1 : causal_conv_output_dim(frames, k, s, stream_subsampling);
    };
    const int64_t stage1_frames = stage_frames(input_frames);
    const int64_t stage1_features = causal_conv_output_dim(feature_dim, k, s, false);
    const int64_t stage2_frames = stage_frames(stage1_frames);
    const int64_t stage2_features = causal_conv_output_dim(stage1_features, k, s, false);
    const int64_t stage3_frames = stage_frames(stage2_frames);
    const int64_t stage3_features = causal_conv_output_dim(stage2_features, k, s, false);
    if (stage3_features * enc.subsampling_channels != 4352) {
        throw std::runtime_error("Nemotron ASR streaming subsampling feature shape mismatch");
    }
    const int64_t drop_frames = raw_cache ? pad_and_drop_drop_frames(enc.subsampling_factor) : 0;
    const int64_t encoded_frames = stage3_frames - drop_frames;
    if (encoded_frames <= 0) {
        throw std::runtime_error("Nemotron ASR pad_and_drop chunk is too short");
    }
    const int64_t key_frames = prefix_capacity + encoded_frames;

    auto graph = std::make_unique<Graph>();
    graph->input_frames = input_frames;
    graph->feature_dim = feature_dim;
    graph->encoded_frames = encoded_frames;
    graph->hidden = enc.hidden_size;
    graph->decoder_hidden = config.decoder_hidden_size;
    graph->lookahead_tokens = lookahead_tokens;
    graph->prefix_frames = prefix_capacity;
    graph->streaming = true;
    graph->first_chunk = first_chunk;
    graph->raw_cache = raw_cache;
    graph->rows = rows;
    graph->backend = execution_context_->backend();
    ggml_init_params params{graph_arena_bytes_, nullptr, true};
    graph->ggml = ggml_init(params);
    if (graph->ggml == nullptr) {
        throw std::runtime_error("Failed to initialize Nemotron ASR streaming encoder graph context");
    }

    engine::core::ModuleBuildContext ctx{graph->ggml, "nemotron_asr.encoder_stream", execution_context_->backend_type()};
    graph->input = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({rows, input_frames, feature_dim}));
    ggml_set_input(graph->input.tensor);
    const int64_t freq0 = feature_dim + k - 1 + s - 1;
    const int64_t freq1 = stage1_features + k - 1 + s - 1;
    const int64_t freq2 = stage2_features + k - 1 + s - 1;
    graph->subsampling_cache0 = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, 1, 1, freq0}));
    graph->subsampling_cache1 = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, enc.subsampling_channels, 1, freq1}));
    graph->subsampling_cache2 = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, enc.subsampling_channels, 1, freq2}));
    for (auto * tensor : {graph->subsampling_cache0.tensor, graph->subsampling_cache1.tensor, graph->subsampling_cache2.tensor}) {
        ggml_set_input(tensor);
        ggml_set_output(tensor);
    }
    graph->mask1 = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({rows, stage1_frames}));
    graph->mask2 = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({rows, stage2_frames}));
    graph->mask3 = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({rows, stage3_frames}));
    graph->keep_mask = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({rows, encoded_frames}));
    graph->attention_mask = engine::core::make_tensor(
        ctx, GGML_TYPE_F32,
        rows == 1 ? engine::core::TensorShape::from_dims({encoded_frames, key_frames})
                  : engine::core::TensorShape::from_dims({rows, 1, encoded_frames, key_frames}));
    graph->prompt = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({rows, encoded_frames, config.num_prompts}));
    graph->pos_emb = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, 2 * key_frames - 1, enc.hidden_size}));
    for (auto * tensor : {graph->mask1.tensor, graph->mask2.tensor, graph->mask3.tensor, graph->keep_mask.tensor, graph->attention_mask.tensor, graph->prompt.tensor, graph->pos_emb.tensor}) {
        ggml_set_input(tensor);
        ggml_set_output(tensor);
    }

    auto x = engine::core::reshape_tensor(ctx, graph->input, engine::core::TensorShape::from_dims({rows, 1, input_frames, feature_dim}));
    auto owned_copy = [&ctx](const engine::core::TensorValue & value) {
        return engine::core::wrap_tensor(
            ggml_cont(ctx.ggml, value.tensor), value.shape, value.type);
    };
    if (raw_cache) {
        x = build_causal_subsampling(ctx, x, weights.subsampling, enc, graph->mask1, graph->mask2, graph->mask3);
    } else {
    x = pad_freq_2d(ctx, x, k, s);
    graph->next_subsampling_cache0 =
        owned_copy(next_time_cache_4d(ctx, graph->subsampling_cache0, x, 1));
    ggml_set_output(graph->next_subsampling_cache0.tensor);
    x = first_chunk
        ? engine::modules::ConcatModule({2}).build(ctx, zero_time_prefix_4d(ctx, x, 1), engine::modules::ConcatModule({2}).build(ctx, graph->subsampling_cache0, x))
        : engine::modules::ConcatModule({2}).build(ctx, graph->subsampling_cache0, x);
    x = engine::modules::Conv2dModule({1, enc.subsampling_channels, k, k, static_cast<int>(s), static_cast<int>(s), 0, 0, 1, 1, true})
            .build(ctx, x, weights.subsampling.conv_in);
    x = engine::modules::ReluModule().build(ctx, engine::modules::TimeMask4dModule().build(ctx, x, graph->mask1));

    x = pad_freq_2d(ctx, x, k, s);
    graph->next_subsampling_cache1 =
        owned_copy(next_time_cache_4d(ctx, graph->subsampling_cache1, x, 1));
    ggml_set_output(graph->next_subsampling_cache1.tensor);
    x = first_chunk
        ? engine::modules::ConcatModule({2}).build(ctx, zero_time_prefix_4d(ctx, x, 1), engine::modules::ConcatModule({2}).build(ctx, graph->subsampling_cache1, x))
        : engine::modules::ConcatModule({2}).build(ctx, graph->subsampling_cache1, x);
    x = engine::modules::DepthwiseConv2dModule({enc.subsampling_channels, k, k, static_cast<int>(s), static_cast<int>(s), 0, 0, 1, 1, true})
            .build(ctx, x, {weights.subsampling.layers[0].depthwise_weight, weights.subsampling.layers[0].depthwise_bias});
    x = engine::modules::TimeMask4dModule().build(ctx, x, graph->mask2);
    x = engine::modules::Conv2dModule({enc.subsampling_channels, enc.subsampling_channels, 1, 1, 1, 1, 0, 0, 1, 1, true})
            .build(ctx, x, weights.subsampling.layers[0].pointwise);
    x = engine::modules::ReluModule().build(ctx, engine::modules::TimeMask4dModule().build(ctx, x, graph->mask2));

    x = pad_freq_2d(ctx, x, k, s);
    graph->next_subsampling_cache2 =
        owned_copy(next_time_cache_4d(ctx, graph->subsampling_cache2, x, 1));
    ggml_set_output(graph->next_subsampling_cache2.tensor);
    x = first_chunk
        ? engine::modules::ConcatModule({2}).build(ctx, zero_time_prefix_4d(ctx, x, 1), engine::modules::ConcatModule({2}).build(ctx, graph->subsampling_cache2, x))
        : engine::modules::ConcatModule({2}).build(ctx, graph->subsampling_cache2, x);
    x = engine::modules::DepthwiseConv2dModule({enc.subsampling_channels, k, k, static_cast<int>(s), static_cast<int>(s), 0, 0, 1, 1, true})
            .build(ctx, x, {weights.subsampling.layers[1].depthwise_weight, weights.subsampling.layers[1].depthwise_bias});
    x = engine::modules::TimeMask4dModule().build(ctx, x, graph->mask3);
    x = engine::modules::Conv2dModule({enc.subsampling_channels, enc.subsampling_channels, 1, 1, 1, 1, 0, 0, 1, 1, true})
            .build(ctx, x, weights.subsampling.layers[1].pointwise);
    x = engine::modules::ReluModule().build(ctx, engine::modules::TimeMask4dModule().build(ctx, x, graph->mask3));
    }

    x = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    x = engine::core::wrap_tensor(ggml_cont(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);
    x = engine::core::reshape_tensor(ctx, x, engine::core::TensorShape::from_dims({rows, stage3_frames, enc.subsampling_channels * stage3_features}));
    x = engine::modules::LinearModule({enc.subsampling_channels * stage3_features, enc.hidden_size, true}).build(ctx, x, weights.subsampling.linear);
    if (drop_frames > 0) {
        x = owned_copy(engine::modules::SliceModule({1, drop_frames, encoded_frames}).build(ctx, x));
    }

    graph->attention_key_cache.reserve(static_cast<size_t>(enc.layers));
    graph->attention_value_cache.reserve(static_cast<size_t>(enc.layers));
    graph->conv_cache.reserve(static_cast<size_t>(enc.layers));
    graph->next_attention_key_cache.reserve(static_cast<size_t>(enc.layers));
    graph->next_attention_value_cache.reserve(static_cast<size_t>(enc.layers));
    graph->next_conv_cache.reserve(static_cast<size_t>(enc.layers));
    const int64_t head_dim = enc.hidden_size / enc.heads;
    for (int64_t layer = 0; layer < enc.layers; ++layer) {
        graph->attention_key_cache.push_back(engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({rows, enc.heads, prefix_capacity, head_dim})));
        graph->attention_value_cache.push_back(engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({rows, enc.heads, prefix_capacity, head_dim})));
        ggml_set_input(graph->attention_key_cache.back().tensor);
        ggml_set_input(graph->attention_value_cache.back().tensor);
        ggml_set_output(graph->attention_key_cache.back().tensor);
        ggml_set_output(graph->attention_value_cache.back().tensor);
        graph->conv_cache.push_back(engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({rows, enc.hidden_size, enc.conv_kernel - 1})));
        ggml_set_input(graph->conv_cache.back().tensor);
        ggml_set_output(graph->conv_cache.back().tensor);
    }

    graph->projected_pos_emb.reserve(static_cast<size_t>(enc.layers));
    graph->projected_pos_emb_computed.reserve(static_cast<size_t>(enc.layers));
    for (int64_t layer = 0; layer < enc.layers; ++layer) {
        graph->projected_pos_emb.push_back(engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, 2 * key_frames - 1, enc.hidden_size})));
        ggml_set_input(graph->projected_pos_emb.back().tensor);
        ggml_set_output(graph->projected_pos_emb.back().tensor);
        graph->projected_pos_emb_computed.push_back(
            engine::modules::LinearModule({enc.hidden_size, enc.hidden_size, false}).build(
                ctx,
                graph->pos_emb,
                {weights.layers[static_cast<size_t>(layer)].self_attn.pos_weight, std::nullopt}));
        ggml_set_output(graph->projected_pos_emb_computed.back().tensor);
    }

    for (int64_t layer = 0; layer < enc.layers; ++layer) {
        auto pos = graph->projected_pos_emb[static_cast<size_t>(layer)];
        if (rows > 1) {
            pos = engine::modules::RepeatModule({engine::core::TensorShape::from_dims({rows, 2 * key_frames - 1, enc.hidden_size})})
                      .build(ctx, pos);
        }
        auto layer_out = build_projected_cache_streaming_encoder_layer(
            ctx,
            x,
            graph->attention_mask,
            graph->keep_mask,
            pos,
            graph->attention_key_cache[static_cast<size_t>(layer)],
            graph->attention_value_cache[static_cast<size_t>(layer)],
            graph->conv_cache[static_cast<size_t>(layer)],
            weights.layers[static_cast<size_t>(layer)],
            enc);
        x = layer_out.output;
        graph->next_attention_key_cache.push_back(layer_out.next_key_cache);
        graph->next_attention_value_cache.push_back(layer_out.next_value_cache);
        graph->next_conv_cache.push_back(owned_copy(layer_out.next_conv_cache));
        ggml_set_output(graph->next_attention_key_cache.back().tensor);
        ggml_set_output(graph->next_attention_value_cache.back().tensor);
        ggml_set_output(graph->next_conv_cache.back().tensor);
    }

    x = engine::modules::ConcatModule({2}).build(ctx, x, graph->prompt);
    x = engine::modules::LinearModule({enc.hidden_size + config.num_prompts, config.prompt_intermediate_size, true})
            .build(ctx, x, weights.prompt_linear1);
    x = engine::modules::ReluModule().build(ctx, x);
    x = engine::modules::LinearModule({config.prompt_intermediate_size, enc.hidden_size, true})
            .build(ctx, x, weights.prompt_linear2);
    graph->output = engine::modules::LinearModule({enc.hidden_size, config.decoder_hidden_size, true})
                        .build(ctx, x, weights.encoder_projector);

    graph->graph = ggml_new_graph_custom(graph->ggml, kEncoderGraphNodes, false);
    ggml_build_forward_expand(graph->graph, graph->output.tensor);
    if (raw_cache) {
        // Last nodes: write the next caches over the cache inputs.
        for (int64_t layer = 0; layer < enc.layers; ++layer) {
            const auto l = static_cast<size_t>(layer);
            ggml_build_forward_expand(graph->graph, ggml_cpy(ctx.ggml, graph->next_attention_key_cache[l].tensor, graph->attention_key_cache[l].tensor));
            ggml_build_forward_expand(graph->graph, ggml_cpy(ctx.ggml, graph->next_attention_value_cache[l].tensor, graph->attention_value_cache[l].tensor));
            ggml_build_forward_expand(graph->graph, ggml_cpy(ctx.ggml, graph->next_conv_cache[l].tensor, graph->conv_cache[l].tensor));
        }
    } else {
        ggml_build_forward_expand(graph->graph, graph->next_subsampling_cache0.tensor);
        ggml_build_forward_expand(graph->graph, graph->next_subsampling_cache1.tensor);
        ggml_build_forward_expand(graph->graph, graph->next_subsampling_cache2.tensor);
    }
    for (int64_t layer = 0; layer < enc.layers; ++layer) {
        ggml_build_forward_expand(graph->graph, graph->next_attention_key_cache[static_cast<size_t>(layer)].tensor);
        ggml_build_forward_expand(graph->graph, graph->next_attention_value_cache[static_cast<size_t>(layer)].tensor);
        ggml_build_forward_expand(graph->graph, graph->next_conv_cache[static_cast<size_t>(layer)].tensor);
    }
    graph->pos_graph = ggml_new_graph_custom(graph->ggml, 4096, false);
    for (const auto & projected : graph->projected_pos_emb_computed) {
        ggml_build_forward_expand(graph->pos_graph, projected.tensor);
    }
    graph->gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(graph->backend));
    if (graph->gallocr == nullptr ||
        !ggml_gallocr_reserve(graph->gallocr, graph->graph) ||
        !ggml_gallocr_alloc_graph(graph->gallocr, graph->graph)) {
        throw std::runtime_error("Failed to allocate Nemotron ASR streaming encoder graph tensors");
    }
    graph->pos_gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(graph->backend));
    if (graph->pos_gallocr == nullptr ||
        !ggml_gallocr_reserve(graph->pos_gallocr, graph->pos_graph) ||
        !ggml_gallocr_alloc_graph(graph->pos_gallocr, graph->pos_graph)) {
        throw std::runtime_error("Failed to allocate Nemotron ASR streaming encoder position graph tensors");
    }
    engine::core::write_tensor_f32(
        graph->pos_emb,
        relative_positional_encoding(key_frames));
    const auto pos_status = engine::core::compute_backend_graph(execution_context_->backend(), graph->pos_graph, nullptr, "Nemotron ASR streaming encoder pos");
    if (pos_status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Nemotron ASR streaming encoder position graph compute failed");
    }
    for (size_t i = 0; i < graph->projected_pos_emb.size(); ++i) {
        ggml_backend_tensor_copy(graph->projected_pos_emb_computed[i].tensor, graph->projected_pos_emb[i].tensor);
    }
    auto write_full_mask = [&](const engine::core::TensorValue & mask, int64_t frames) {
        engine::modules::fill_asr_keep_mask(mask_scratch_, frames, frames);
        engine::core::write_tensor_i32(mask, tile_rows(mask_scratch_, rows));
    };
    write_full_mask(graph->mask1, graph->mask1.shape.dims[1]);
    write_full_mask(graph->mask2, graph->mask2.shape.dims[1]);
    write_full_mask(graph->mask3, graph->mask3.shape.dims[1]);
    write_full_mask(graph->keep_mask, graph->encoded_frames);
    engine::modules::fill_asr_stream_attention_bias(
        attention_mask_scratch_,
        graph->encoded_frames,
        graph->prefix_frames + graph->encoded_frames,
        0,
        0,
        0,
        enc.sliding_window - 1,
        lookahead_tokens);
    engine::core::write_tensor_f32(graph->attention_mask, tile_rows(attention_mask_scratch_, rows));
    if (raw_cache) {
        const size_t per_layer = 3 * static_cast<size_t>(rows);
        ggml_init_params view_params{ggml_tensor_overhead() * (per_layer * static_cast<size_t>(enc.layers) + 1), nullptr, true};
        graph->views = ggml_init(view_params);
        if (graph->views == nullptr) throw std::runtime_error("Failed to initialize Nemotron ASR cache view context");
        auto row_views = [&](const std::vector<engine::core::TensorValue> & tensors) {
            std::vector<std::vector<ggml_tensor *>> out(tensors.size());
            for (size_t layer = 0; layer < tensors.size(); ++layer) {
                ggml_tensor * tensor = tensors[layer].tensor;
                const int64_t per_row = ggml_nelements(tensor) / rows;
                for (int64_t row = 0; row < rows; ++row) {
                    ggml_tensor * view = ggml_view_1d(
                        graph->views, tensor, per_row, static_cast<size_t>(row * per_row) * ggml_element_size(tensor));
                    if (ggml_backend_view_init(view) != GGML_STATUS_SUCCESS) {
                        throw std::runtime_error("Failed to bind Nemotron ASR cache view");
                    }
                    out[layer].push_back(view);
                }
            }
            return out;
        };
        graph->key_in = row_views(graph->attention_key_cache);
        graph->value_in = row_views(graph->attention_value_cache);
        graph->conv_in = row_views(graph->conv_cache);
        graph->resident.assign(static_cast<size_t>(rows), nullptr);
    }
    Graph * built = graph.get();
    stream_graphs_.push_back(std::move(graph));
    const double build_ms = engine::debug::elapsed_ms(build_start, Clock::now());
    debug::timing_log_scalar("nemotron_asr.encoder.stream.graph_build_ms", build_ms);
    debug::timing_log_scalar("nemotron_asr.encoder.stream.graph_rebuild_ms", build_ms);
    debug::trace_log_scalar("nemotron_asr.encoder.stream.graph_cache_hit", false);
    debug::trace_log_scalar("nemotron_asr.encoder.stream.graph_input_frames", input_frames);
    debug::trace_log_scalar("nemotron_asr.encoder.stream.graph_encoded_frames", stage3_frames);
    debug::trace_log_scalar("nemotron_asr.encoder.stream.graph_prefix_frames", prefix_frames);
    debug::trace_log_scalar("nemotron_asr.encoder.stream.graph_prefix_capacity", prefix_capacity);
    return *built;
}

void NemotronEncoderRuntime::prepare_capacity(int64_t input_frames, int64_t feature_dim, int64_t lookahead_tokens) {
    ensure_graph(input_frames, feature_dim, lookahead_tokens);
}

void NemotronEncoderRuntime::release_offline_graph() {
    graph_.reset();
}

void NemotronEncoderRuntime::prepare_streaming_capacity(int64_t feature_dim, int64_t lookahead_tokens) {
    const auto & enc = assets_->config.encoder;
    const int64_t first_frames = std::max<int64_t>(
        enc.subsampling_factor,
        1 + enc.subsampling_factor * lookahead_tokens);
    const int64_t next_frames = enc.subsampling_factor *
        std::max<int64_t>(lookahead_tokens + 1, 4);
    (void) ensure_stream_graph(first_frames, feature_dim, lookahead_tokens, 0, true);
    (void) ensure_stream_graph(next_frames, feature_dim, lookahead_tokens, 0, false);
}

NemotronEncoderStreamState NemotronEncoderRuntime::make_stream_state() const {
    NemotronEncoderStreamState state;
    return state;
}

NemotronEncodedAudio NemotronEncoderRuntime::encode(
    const NemotronFrontendFeatures & features,
    int64_t prompt_id,
    int64_t lookahead_tokens) {
    if (features.frames <= 0 || features.feature_dim <= 0) {
        throw std::runtime_error("Nemotron ASR encoder requires positive frontend shape");
    }
    if (prompt_id < 0 || prompt_id >= assets_->config.num_prompts) {
        throw std::runtime_error("Nemotron ASR prompt id is out of range");
    }
    const auto wall_start = Clock::now();
    ensure_graph(features.frames, features.feature_dim, lookahead_tokens);
    auto & graph = *graph_;
    input_scratch_.assign(static_cast<size_t>(graph.input_frames * graph.feature_dim), 0.0f);
    for (int64_t t = 0; t < features.frames; ++t) {
        for (int64_t f = 0; f < features.feature_dim; ++f) {
            input_scratch_[static_cast<size_t>(t * graph.feature_dim + f)] =
                features.values[static_cast<size_t>(t * features.feature_dim + f)];
        }
    }
    engine::core::write_tensor_f32(graph.input, input_scratch_);

    const auto & enc = assets_->config.encoder;
    const bool streaming_graph = false;
    const int64_t valid1 = std::min<int64_t>(graph.mask1.shape.dims[1], causal_conv_output_dim(features.valid_frames, enc.subsampling_kernel, enc.subsampling_stride, streaming_graph));
    const int64_t valid2 = std::min<int64_t>(graph.mask2.shape.dims[1], causal_conv_output_dim(valid1, enc.subsampling_kernel, enc.subsampling_stride, streaming_graph));
    const int64_t valid3 = std::min<int64_t>(graph.encoded_frames, causal_conv_output_dim(valid2, enc.subsampling_kernel, enc.subsampling_stride, streaming_graph));
    engine::modules::fill_asr_keep_mask(mask_scratch_, graph.mask1.shape.dims[1], valid1);
    engine::core::write_tensor_i32(graph.mask1, mask_scratch_);
    engine::modules::fill_asr_keep_mask(mask_scratch_, graph.mask2.shape.dims[1], valid2);
    engine::core::write_tensor_i32(graph.mask2, mask_scratch_);
    engine::modules::fill_asr_keep_mask(mask_scratch_, graph.mask3.shape.dims[1], valid3);
    engine::core::write_tensor_i32(graph.mask3, mask_scratch_);
    engine::modules::fill_asr_keep_mask(mask_scratch_, graph.encoded_frames, valid3);
    engine::core::write_tensor_i32(graph.keep_mask, mask_scratch_);
    engine::modules::fill_asr_chunked_attention_bias(
        attention_mask_scratch_,
        graph.encoded_frames,
        valid3,
        enc.sliding_window - 1,
        lookahead_tokens);
    engine::core::write_tensor_f32(graph.attention_mask, attention_mask_scratch_);

    prompt_scratch_ = prompt_one_hot(graph.encoded_frames, assets_->config.num_prompts, prompt_id);
    engine::core::write_tensor_f32(graph.prompt, prompt_scratch_);

    engine::core::set_backend_threads(execution_context_->backend(), execution_context_->config().threads);
    const auto compute_start = Clock::now();
    const auto status = engine::core::compute_backend_graph(execution_context_->backend(), graph.graph, nullptr, "Nemotron ASR encoder");
    ggml_backend_synchronize(execution_context_->backend());
    debug::timing_log_scalar("nemotron_asr.encoder.graph.compute_ms", engine::debug::elapsed_ms(compute_start, Clock::now()));
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Nemotron ASR encoder graph compute failed");
    }
    engine::core::read_tensor_f32_into(graph.output.tensor, output_scratch_);
    if (valid3 < graph.encoded_frames) {
        for (int64_t row = valid3; row < graph.encoded_frames; ++row) {
            std::fill_n(
                output_scratch_.begin() + static_cast<std::ptrdiff_t>(row * graph.decoder_hidden),
                static_cast<std::ptrdiff_t>(graph.decoder_hidden),
                0.0f);
        }
    }
    NemotronEncodedAudio out;
    out.values = output_scratch_;
    out.frames = graph.encoded_frames;
    out.valid_frames = valid3;
    out.hidden_size = graph.decoder_hidden;
    debug::timing_log_scalar("nemotron_asr.encoder_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    debug::trace_log_scalar("nemotron_asr.encoder.valid_frames", valid3);
    return out;
}

NemotronEncodedAudio NemotronEncoderRuntime::encode_stream_chunk(
    const NemotronFrontendFeatures & features,
    int64_t prompt_id,
    int64_t lookahead_tokens,
    NemotronEncoderStreamState & state) {
    if (features.frames <= 0 || features.feature_dim <= 0) {
        throw std::runtime_error("Nemotron ASR streaming encoder requires positive frontend shape");
    }
    if (prompt_id < 0 || prompt_id >= assets_->config.num_prompts) {
        throw std::runtime_error("Nemotron ASR prompt id is out of range");
    }
    const auto wall_start = Clock::now();
    auto & graph = ensure_stream_graph(features.frames, features.feature_dim, lookahead_tokens, state.attention_cached_frames, state.first_chunk);
    const auto & enc = assets_->config.encoder;
    Graph * backend_cache_source = nullptr;
    if (state.backend_cache_valid) {
        for (const auto & graph_slot : stream_graphs_) {
            if (graph_slot.get() == state.backend_cache_owner) {
                backend_cache_source = graph_slot.get();
                break;
            }
        }
        if (backend_cache_source == nullptr) {
            throw std::runtime_error("Nemotron ASR streaming backend cache owner is no longer available");
        }
    }
    const bool use_same_backend_cache = backend_cache_source == &graph;
    const bool use_cross_backend_cache = backend_cache_source != nullptr && backend_cache_source != &graph;

    input_scratch_.assign(static_cast<size_t>(graph.input_frames * graph.feature_dim), 0.0f);
    for (int64_t t = 0; t < features.frames; ++t) {
        for (int64_t f = 0; f < features.feature_dim; ++f) {
            input_scratch_[static_cast<size_t>(t * graph.feature_dim + f)] =
                features.values[static_cast<size_t>(t * features.feature_dim + f)];
        }
    }
    engine::core::write_tensor_f32(graph.input, input_scratch_);

    auto write_zeros = [](const engine::core::TensorValue & tensor) {
        const size_t elements = static_cast<size_t>(tensor.shape.num_elements());
        std::vector<float> zeros(elements, 0.0f);
        engine::core::write_tensor_f32(tensor, zeros);
    };
    const auto cache_transfer_start = Clock::now();
    if (use_cross_backend_cache) {
        const auto backend = execution_context_->backend();
        if (ggml_nelements(backend_cache_source->next_subsampling_cache0.tensor) != ggml_nelements(graph.subsampling_cache0.tensor) ||
            ggml_nelements(backend_cache_source->next_subsampling_cache1.tensor) != ggml_nelements(graph.subsampling_cache1.tensor) ||
            ggml_nelements(backend_cache_source->next_subsampling_cache2.tensor) != ggml_nelements(graph.subsampling_cache2.tensor)) {
            throw std::runtime_error("Nemotron ASR streaming subsampling backend cache shape mismatch");
        }
        ggml_backend_tensor_copy_async(backend, backend, backend_cache_source->next_subsampling_cache0.tensor, graph.subsampling_cache0.tensor);
        ggml_backend_tensor_copy_async(backend, backend, backend_cache_source->next_subsampling_cache1.tensor, graph.subsampling_cache1.tensor);
        ggml_backend_tensor_copy_async(backend, backend, backend_cache_source->next_subsampling_cache2.tensor, graph.subsampling_cache2.tensor);
    } else if (!use_same_backend_cache) {
        if (!state.first_chunk) {
            throw std::runtime_error("Nemotron ASR streaming cache is missing between chunks");
        }
        write_zeros(graph.subsampling_cache0);
        write_zeros(graph.subsampling_cache1);
        write_zeros(graph.subsampling_cache2);
    }

    if (!graph.stream_static_inputs_valid || graph.stream_static_prompt_id != prompt_id) {
        prompt_scratch_ = prompt_one_hot(graph.encoded_frames, assets_->config.num_prompts, prompt_id);
        engine::core::write_tensor_f32(graph.prompt, prompt_scratch_);
        graph.stream_static_inputs_valid = true;
        graph.stream_static_prompt_id = prompt_id;
    }

    for (size_t layer = 0; layer < graph.conv_cache.size(); ++layer) {
        if (use_cross_backend_cache) {
            const auto backend = execution_context_->backend();
            if (layer >= backend_cache_source->next_conv_cache.size() ||
                ggml_nelements(backend_cache_source->next_conv_cache[layer].tensor) != ggml_nelements(graph.conv_cache[layer].tensor)) {
                throw std::runtime_error("Nemotron ASR streaming convolution backend cache shape mismatch");
            }
            ggml_backend_tensor_copy_async(backend, backend, backend_cache_source->next_conv_cache[layer].tensor, graph.conv_cache[layer].tensor);
        } else if (!use_same_backend_cache) {
            if (!state.first_chunk) {
                throw std::runtime_error("Nemotron ASR streaming convolution cache is missing between chunks");
            }
            write_zeros(graph.conv_cache[layer]);
        }
    }
    if (!use_cross_backend_cache && !use_same_backend_cache && !state.first_chunk) {
        throw std::runtime_error("Nemotron ASR streaming attention cache is missing between chunks");
    }
    if (use_cross_backend_cache &&
        (backend_cache_source->next_attention_key_cache.size() != graph.attention_key_cache.size() ||
         backend_cache_source->next_attention_value_cache.size() != graph.attention_value_cache.size())) {
        throw std::runtime_error("Nemotron ASR streaming attention cache layer count mismatch");
    }
    for (size_t layer = 0; layer < graph.attention_key_cache.size(); ++layer) {
        if (use_cross_backend_cache) {
            const auto backend = execution_context_->backend();
            if (ggml_nelements(backend_cache_source->next_attention_key_cache[layer].tensor) != ggml_nelements(graph.attention_key_cache[layer].tensor) ||
                ggml_nelements(backend_cache_source->next_attention_value_cache[layer].tensor) != ggml_nelements(graph.attention_value_cache[layer].tensor)) {
                throw std::runtime_error("Nemotron ASR streaming attention backend cache shape mismatch");
            }
            ggml_backend_tensor_copy_async(
                backend,
                backend,
                backend_cache_source->next_attention_key_cache[layer].tensor,
                graph.attention_key_cache[layer].tensor);
            ggml_backend_tensor_copy_async(
                backend,
                backend,
                backend_cache_source->next_attention_value_cache[layer].tensor,
                graph.attention_value_cache[layer].tensor);
        } else if (!use_same_backend_cache) {
            write_zeros(graph.attention_key_cache[layer]);
            write_zeros(graph.attention_value_cache[layer]);
        }
    }
    engine::modules::fill_asr_stream_attention_bias(
        attention_mask_scratch_,
        graph.encoded_frames,
        graph.prefix_frames + graph.encoded_frames,
        state.attention_cached_frames,
        state.attention_seen_frames,
        state.attention_seen_frames - state.attention_cached_frames,
        enc.sliding_window - 1,
        lookahead_tokens);
    engine::core::write_tensor_f32(graph.attention_mask, attention_mask_scratch_);
    debug::timing_log_scalar(
        "nemotron_asr.encoder.stream.cache_transfer_ms",
        engine::debug::elapsed_ms(cache_transfer_start, Clock::now()));

    engine::core::set_backend_threads(execution_context_->backend(), execution_context_->config().threads);
    const auto compute_start = Clock::now();
    const auto status = engine::core::compute_backend_graph(execution_context_->backend(), graph.graph, nullptr, "Nemotron ASR streaming encoder");
    ggml_backend_synchronize(execution_context_->backend());
    debug::timing_log_scalar("nemotron_asr.encoder.stream.graph.compute_ms", engine::debug::elapsed_ms(compute_start, Clock::now()));
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Nemotron ASR streaming encoder graph compute failed");
    }

    engine::core::read_tensor_f32_into(graph.output.tensor, output_scratch_);
    state.attention_seen_frames += graph.encoded_frames;
    state.attention_cached_frames = std::min<int64_t>(enc.sliding_window - 1, state.attention_seen_frames);
    state.backend_cache_valid = true;
    state.backend_cache_owner = static_cast<const void *>(&graph);
    const auto backend = execution_context_->backend();
    ggml_backend_tensor_copy_async(backend, backend, graph.next_subsampling_cache0.tensor, graph.subsampling_cache0.tensor);
    ggml_backend_tensor_copy_async(backend, backend, graph.next_subsampling_cache1.tensor, graph.subsampling_cache1.tensor);
    ggml_backend_tensor_copy_async(backend, backend, graph.next_subsampling_cache2.tensor, graph.subsampling_cache2.tensor);
    for (int64_t layer = 0; layer < enc.layers; ++layer) {
        ggml_backend_tensor_copy_async(
            backend,
            backend,
            graph.next_attention_key_cache[static_cast<size_t>(layer)].tensor,
            graph.attention_key_cache[static_cast<size_t>(layer)].tensor);
        ggml_backend_tensor_copy_async(
            backend,
            backend,
            graph.next_attention_value_cache[static_cast<size_t>(layer)].tensor,
            graph.attention_value_cache[static_cast<size_t>(layer)].tensor);
        ggml_backend_tensor_copy_async(
            backend,
            backend,
            graph.next_conv_cache[static_cast<size_t>(layer)].tensor,
            graph.conv_cache[static_cast<size_t>(layer)].tensor);
    }

    NemotronEncodedAudio out;
    out.values = output_scratch_;
    out.frames = graph.encoded_frames;
    out.valid_frames = graph.encoded_frames;
    out.hidden_size = graph.decoder_hidden;
    state.first_chunk = false;
    debug::timing_log_scalar("nemotron_asr.encoder.stream_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    debug::trace_log_scalar("nemotron_asr.encoder.stream.valid_frames", out.valid_frames);
    debug::trace_log_scalar("nemotron_asr.encoder.stream.attention_seen_frames", state.attention_seen_frames);
    debug::trace_log_scalar("nemotron_asr.encoder.stream.attention_cached_frames", state.attention_cached_frames);
    return out;
}

int64_t NemotronEncoderRuntime::pad_and_drop_cache_frames() const {
    return pad_and_drop_cache(assets_->config.encoder.subsampling_factor);
}

std::vector<NemotronEncodedAudio> NemotronEncoderRuntime::encode_pad_and_drop_batch(
    const std::vector<const NemotronFrontendFeatures *> & features,
    int64_t prompt_id,
    int64_t lookahead_tokens,
    const std::vector<NemotronEncoderSpeakerState *> & states) {
    const auto rows = static_cast<int64_t>(features.size());
    if (rows <= 0 || states.size() != features.size()) {
        throw std::runtime_error("Nemotron ASR pad_and_drop batch needs one state per row");
    }
    const int64_t frames = features.front()->frames;
    const int64_t dim = features.front()->feature_dim;
    for (const auto * row : features) {
        if (row->frames != frames || row->feature_dim != dim || frames <= pad_and_drop_cache_frames() || dim <= 0) {
            throw std::runtime_error("Nemotron ASR pad_and_drop rows need equal cache-plus-new frame counts");
        }
    }
    if (prompt_id < 0 || prompt_id >= assets_->config.num_prompts) {
        throw std::runtime_error("Nemotron ASR prompt id is out of range");
    }
    const auto wall_start = Clock::now();
    const auto & enc = assets_->config.encoder;
    auto & graph = ensure_stream_graph(frames, dim, lookahead_tokens, 0, false, true, rows);
    const auto backend = execution_context_->backend();

    input_scratch_.clear();
    for (const auto * row : features) {
        input_scratch_.insert(input_scratch_.end(), row->values.begin(),
                              row->values.begin() + static_cast<std::ptrdiff_t>(frames * dim));
    }
    engine::core::write_tensor_f32(graph.input, input_scratch_);
    if (!graph.stream_static_inputs_valid || graph.stream_static_prompt_id != prompt_id) {
        prompt_scratch_ = prompt_one_hot(rows * graph.encoded_frames, assets_->config.num_prompts, prompt_id);
        engine::core::write_tensor_f32(graph.prompt, prompt_scratch_);
        graph.stream_static_inputs_valid = true;
        graph.stream_static_prompt_id = prompt_id;
    }

    std::vector<float> masks;
    for (int64_t row = 0; row < rows; ++row) {
        auto & state = *states[static_cast<size_t>(row)];
        if (state.caches == nullptr) {
            // A new speaker starts from zero caches (NeMo get_initial_cache_state).
            auto caches = std::make_shared<NemotronEncoderSpeakerCaches>();
            ggml_init_params params{ggml_tensor_overhead() * static_cast<size_t>(3 * enc.layers + 1), nullptr, true};
            caches->ggml = ggml_init(params);
            if (caches->ggml == nullptr) throw std::runtime_error("Failed to initialize Nemotron ASR speaker cache context");
            for (int64_t layer = 0; layer < enc.layers; ++layer) {
                const auto l = static_cast<size_t>(layer);
                caches->key.push_back(ggml_new_tensor_1d(caches->ggml, GGML_TYPE_F32, ggml_nelements(graph.key_in[l][0])));
                caches->value.push_back(ggml_new_tensor_1d(caches->ggml, GGML_TYPE_F32, ggml_nelements(graph.value_in[l][0])));
                caches->conv.push_back(ggml_new_tensor_1d(caches->ggml, GGML_TYPE_F32, ggml_nelements(graph.conv_in[l][0])));
            }
            caches->buffer = ggml_backend_alloc_ctx_tensors(caches->ggml, backend);
            if (caches->buffer == nullptr) throw std::runtime_error("Failed to allocate Nemotron ASR speaker caches");
            ggml_backend_buffer_clear(caches->buffer, 0);
            state.caches = std::move(caches);
        }
        engine::modules::fill_asr_stream_attention_bias(
            attention_mask_scratch_,
            graph.encoded_frames,
            graph.prefix_frames + graph.encoded_frames,
            state.attention_cached_frames,
            state.attention_seen_frames,
            state.attention_seen_frames - state.attention_cached_frames,
            enc.sliding_window - 1,
            lookahead_tokens);
        masks.insert(masks.end(), attention_mask_scratch_.begin(), attention_mask_scratch_.end());
    }
    engine::core::write_tensor_f32(graph.attention_mask, masks);

    // Put each speaker's latest caches into its row, moving only what changed.
    auto copy_row = [&](Graph & owner, int64_t row, NemotronEncoderSpeakerCaches & caches, bool to_speaker) {
        const auto r = static_cast<size_t>(row);
        for (int64_t layer = 0; layer < enc.layers; ++layer) {
            const auto l = static_cast<size_t>(layer);
            for (auto [speaker, rows_of] : {std::pair{&caches.key, &owner.key_in}, std::pair{&caches.value, &owner.value_in},
                                            std::pair{&caches.conv, &owner.conv_in}}) {
                ggml_tensor * mine = (*speaker)[l];
                ggml_tensor * graph_row = (*rows_of)[l][r];
                ggml_backend_tensor_copy(to_speaker ? graph_row : mine, to_speaker ? mine : graph_row);
            }
        }
    };
    auto flush = [&](NemotronEncoderSpeakerCaches & caches) {
        if (caches.slot == nullptr) return;
        copy_row(*static_cast<Graph *>(const_cast<void *>(caches.graph)), caches.row, caches, true);
        *caches.slot = nullptr;
        caches.slot = nullptr;
        caches.graph = nullptr;
        caches.row = -1;
    };
    for (int64_t row = 0; row < rows; ++row) {
        auto * occupant = graph.resident[static_cast<size_t>(row)];
        if (occupant != nullptr && occupant != states[static_cast<size_t>(row)]->caches.get()) flush(*occupant);
    }
    for (int64_t row = 0; row < rows; ++row) {
        auto & caches = *states[static_cast<size_t>(row)]->caches;
        if (caches.graph == &graph && caches.row == row) continue;
        flush(caches);
        copy_row(graph, row, caches, false);
        graph.resident[static_cast<size_t>(row)] = &caches;
        caches.slot = &graph.resident[static_cast<size_t>(row)];
        caches.graph = &graph;
        caches.row = row;
    }

    engine::core::set_backend_threads(backend, execution_context_->config().threads);
    const auto status = engine::core::compute_backend_graph(backend, graph.graph, nullptr, "Nemotron ASR pad_and_drop encoder");
    ggml_backend_synchronize(backend);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Nemotron ASR pad_and_drop encoder graph compute failed");
    }
    engine::core::read_tensor_f32_into(graph.output.tensor, output_scratch_);

    std::vector<NemotronEncodedAudio> out(static_cast<size_t>(rows));
    const size_t row_values = static_cast<size_t>(graph.encoded_frames * graph.decoder_hidden);
    for (int64_t row = 0; row < rows; ++row) {
        auto & state = *states[static_cast<size_t>(row)];
        state.attention_seen_frames += graph.encoded_frames;
        state.attention_cached_frames = std::min<int64_t>(enc.sliding_window - 1, state.attention_seen_frames);
        auto & encoded = out[static_cast<size_t>(row)];
        encoded.values.assign(output_scratch_.begin() + static_cast<std::ptrdiff_t>(row * row_values),
                              output_scratch_.begin() + static_cast<std::ptrdiff_t>((row + 1) * row_values));
        encoded.frames = graph.encoded_frames;
        encoded.valid_frames = graph.encoded_frames;
        encoded.hidden_size = graph.decoder_hidden;
    }
    debug::timing_log_scalar("nemotron_asr.encoder.pad_and_drop_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    debug::trace_log_scalar("nemotron_asr.encoder.pad_and_drop_rows", rows);
    return out;
}

}  // namespace engine::models::nemotron_asr
