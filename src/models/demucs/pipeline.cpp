#include "engine/models/demucs/pipeline.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/cross_attention.h"
#include "engine/framework/modules/attention/self_attention.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/models/demucs/frontend.h"
#include "engine/models/demucs/postprocess.h"
#include "../common/constant_tensor_cache.h"

#include <ggml-backend.h>
#include <ggml.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::demucs {
namespace {

namespace assets_ns = engine::assets;
namespace modules = engine::modules;
namespace binding = engine::modules::binding;

using LinearWeights = modules::LinearWeights;
using Conv1dWeights = modules::Conv1dWeights;
using Conv2dWeights = modules::Conv2dWeights;
using ConvTranspose1dWeights = modules::ConvTranspose1dWeights;

struct ConvTranspose2dWeights {
    core::TensorValue weight;
    std::optional<core::TensorValue> bias;
};

struct DConvResidualWeights {
    Conv1dWeights conv1;
    modules::NormWeights norm1;
    Conv1dWeights conv2;
    modules::NormWeights norm2;
    modules::LayerScaleWeights scale;
    int dilation = 1;
};

struct DConvWeights {
    std::vector<DConvResidualWeights> layers;
};

struct HEncLayerWeights {
    Conv1dWeights conv1d;
    Conv2dWeights conv2d;
    bool freq = false;
    bool rewrite = false;
    bool dconv = false;
    bool norm = false;
    int context = 0;
    int stride = 1;
    int pad = 0;
    modules::NormWeights norm1;
    modules::NormWeights norm2;
    DConvWeights dconv_weights;
    Conv1dWeights rewrite1d;
    Conv2dWeights rewrite2d;
};

struct HDecLayerWeights {
    ConvTranspose1dWeights conv_tr_1d;
    ConvTranspose2dWeights conv_tr_freq;
    bool freq = false;
    bool empty = false;
    bool last = false;
    bool rewrite = false;
    bool dconv = false;
    bool norm = false;
    int pad = 0;
    modules::NormWeights norm1;
    modules::NormWeights norm2;
    DConvWeights dconv_weights;
    Conv1dWeights rewrite1d;
    Conv2dWeights rewrite2d;
};

struct TransformerSelfLayerWeights {
    modules::NormWeights norm1;
    core::TensorValue in_proj_weight;
    core::TensorValue in_proj_bias;
    LinearWeights out_proj;
    modules::NormWeights norm2;
    LinearWeights linear1;
    LinearWeights linear2;
    modules::LayerScaleWeights gamma1;
    modules::LayerScaleWeights gamma2;
    modules::NormWeights norm_out;
};

struct TransformerCrossLayerWeights {
    modules::NormWeights norm1;
    modules::NormWeights norm2;
    core::TensorValue in_proj_weight;
    core::TensorValue in_proj_bias;
    LinearWeights out_proj;
    modules::NormWeights norm3;
    LinearWeights linear1;
    LinearWeights linear2;
    modules::LayerScaleWeights gamma1;
    modules::LayerScaleWeights gamma2;
    modules::NormWeights norm_out;
};

struct TransformerLayerWeights {
    bool cross = false;
    TransformerSelfLayerWeights self;
    TransformerCrossLayerWeights cross_weights;
};

struct CrossTransformerWeights {
    modules::NormWeights norm_in_freq;
    modules::NormWeights norm_in_time;
    std::vector<TransformerLayerWeights> freq_layers;
    std::vector<TransformerLayerWeights> time_layers;
};

struct HTDemucsWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    std::vector<HEncLayerWeights> encoder;
    std::vector<HEncLayerWeights> tencoder;
    std::vector<HDecLayerWeights> decoder;
    std::vector<HDecLayerWeights> tdecoder;
    std::optional<std::vector<float>> freq_embedding_values;
    std::optional<Conv1dWeights> channel_upsampler;
    std::optional<Conv1dWeights> channel_downsampler;
    std::optional<Conv1dWeights> channel_upsampler_t;
    std::optional<Conv1dWeights> channel_downsampler_t;
    std::optional<CrossTransformerWeights> transformer;
};

core::TensorValue ensure_contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return core::ensure_backend_addressable_layout(ctx, value);
}

ConvTranspose2dWeights load_conv_transpose2d(
    core::BackendWeightStore & store,
    const assets_ns::TensorSource & source,
    const std::string & prefix,
    assets_ns::TensorStorageType storage_type,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    bool use_bias) {
    ConvTranspose2dWeights weights;
    weights.weight = store.load_tensor_as_shape(
        source,
        prefix + ".weight",
        storage_type,
        {in_channels, out_channels, kernel_size, 1},
        core::TensorShape::from_dims({in_channels, out_channels, 1, kernel_size}));
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    }
    return weights;
}

core::TensorValue conv_transpose2d_weight(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & weight) {
    auto contiguous = ensure_contiguous(ctx, weight);
    if (contiguous.type == GGML_TYPE_F32 || contiguous.type == GGML_TYPE_F16) {
        return contiguous;
    }
    if (contiguous.type == GGML_TYPE_BF16 && ctx.backend_type == core::BackendType::Cpu) {
        return core::wrap_tensor(ggml_cast(ctx.ggml, contiguous.tensor, GGML_TYPE_F16), contiguous.shape, GGML_TYPE_F16);
    }
    if (contiguous.type == GGML_TYPE_BF16 || ggml_is_quantized(contiguous.type)) {
        return core::wrap_tensor(ggml_cast(ctx.ggml, contiguous.tensor, GGML_TYPE_F32), contiguous.shape, GGML_TYPE_F32);
    }
    throw std::runtime_error(
        std::string("HTDemucs ConvTranspose2d does not support weight type: ") + ggml_type_name(contiguous.type));
}

modules::LayerScaleWeights load_layer_scale(
    core::BackendWeightStore & store,
    const assets_ns::TensorSource & source,
    const std::string & key,
    int64_t hidden_size) {
    return {store.load_f32_tensor(source, key, {hidden_size})};
}

core::TensorValue group_norm_affine(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t groups,
    float eps,
    const modules::NormWeights & weights) {
    auto output = core::wrap_tensor(ggml_group_norm(ctx.ggml, input.tensor, groups, eps), input.shape, GGML_TYPE_F32);
    if (weights.weight.has_value()) {
        core::TensorShape weight_shape = {};
        weight_shape.rank = input.shape.rank;
        for (size_t i = 0; i < weight_shape.rank; ++i) {
            weight_shape.dims[i] = 1;
        }
        weight_shape.dims[1] = input.shape.dims[1];
        auto weight_view = core::reshape_tensor(ctx, *weights.weight, weight_shape);
        output = core::wrap_tensor(ggml_mul(ctx.ggml, output.tensor, weight_view.tensor), output.shape, GGML_TYPE_F32);
    }
    if (weights.bias.has_value()) {
        core::TensorShape bias_shape = {};
        bias_shape.rank = input.shape.rank;
        for (size_t i = 0; i < bias_shape.rank; ++i) {
            bias_shape.dims[i] = 1;
        }
        bias_shape.dims[1] = input.shape.dims[1];
        auto bias_view = core::reshape_tensor(ctx, *weights.bias, bias_shape);
        output = core::wrap_tensor(ggml_add(ctx.ggml, output.tensor, bias_view.tensor), output.shape, GGML_TYPE_F32);
    }
    return output;
}

core::TensorValue group_norm_freq_independent(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t groups,
    float eps,
    const modules::NormWeights & weights) {
    if (groups != 1) {
        throw std::runtime_error("HTDemucs frequency DConv normalization currently supports only groups=1");
    }
    auto x = modules::TransposeModule({{0, 2, 1, 3}, input.shape.rank}).build(ctx, input);
    x = ensure_contiguous(ctx, x);
    const int64_t batch = x.shape.dims[0];
    const int64_t freqs = x.shape.dims[1];
    const int64_t channels = x.shape.dims[2];
    const int64_t frames = x.shape.dims[3];
    auto flat = core::reshape_tensor(
        ctx,
        x,
        core::TensorShape::from_dims({batch * freqs, 1, channels * frames}));
    auto normalized_flat = modules::LayerNormModule({channels * frames, eps, false, false}).build(ctx, flat, {});
    auto normalized = core::reshape_tensor(
        ctx,
        normalized_flat,
        core::TensorShape::from_dims({batch, freqs, channels, frames}));
    normalized = modules::TransposeModule({{0, 2, 1, 3}, normalized.shape.rank}).build(ctx, normalized);

    if (weights.weight.has_value()) {
        auto weight_view = core::reshape_tensor(ctx, *weights.weight, core::TensorShape::from_dims({1, channels, 1, 1}));
        normalized = core::wrap_tensor(ggml_mul(ctx.ggml, normalized.tensor, weight_view.tensor), normalized.shape, GGML_TYPE_F32);
    }
    if (weights.bias.has_value()) {
        auto bias_view = core::reshape_tensor(ctx, *weights.bias, core::TensorShape::from_dims({1, channels, 1, 1}));
        normalized = core::wrap_tensor(ggml_add(ctx.ggml, normalized.tensor, bias_view.tensor), normalized.shape, GGML_TYPE_F32);
    }
    return normalized;
}

core::TensorValue group_norm_sequence_exact(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    float eps,
    const modules::NormWeights & weights) {
    auto flat = core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, input),
        core::TensorShape::from_dims({input.shape.dims[0], 1, input.shape.dims[1] * input.shape.dims[2]}));
    auto normalized_flat = modules::LayerNormModule(
        {input.shape.dims[1] * input.shape.dims[2], eps, false, false}).build(ctx, flat, {});
    auto normalized = core::reshape_tensor(ctx, normalized_flat, input.shape);

    if (weights.weight.has_value()) {
        auto weight_view = core::reshape_tensor(ctx, *weights.weight, core::TensorShape::from_dims({1, 1, input.shape.dims[2]}));
        normalized = core::wrap_tensor(ggml_mul(ctx.ggml, normalized.tensor, weight_view.tensor), input.shape, GGML_TYPE_F32);
    }
    if (weights.bias.has_value()) {
        auto bias_view = core::reshape_tensor(ctx, *weights.bias, core::TensorShape::from_dims({1, 1, input.shape.dims[2]}));
        normalized = core::wrap_tensor(ggml_add(ctx.ggml, normalized.tensor, bias_view.tensor), input.shape, GGML_TYPE_F32);
    }
    return normalized;
}

core::TensorValue glu_channels(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    const int64_t channels = input.shape.dims[1];
    if (channels % 2 != 0) {
        throw std::runtime_error("HTDemucs channel GLU requires an even channel dimension");
    }
    const int64_t half = channels / 2;
    auto a = modules::SliceModule({1, 0, half}).build(ctx, input);
    auto b = modules::SliceModule({1, half, half}).build(ctx, input);
    b = modules::SigmoidModule{}.build(ctx, b);
    return modules::MulModule{}.build(ctx, a, b);
}

core::TensorValue layer_scale_channel_first(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::LayerScaleWeights & weights) {
    core::TensorShape scale_shape = {};
    scale_shape.rank = input.shape.rank;
    for (size_t i = 0; i < scale_shape.rank; ++i) {
        scale_shape.dims[i] = 1;
    }
    scale_shape.dims[1] = input.shape.dims[1];
    auto scale_view = core::reshape_tensor(ctx, weights.scale, scale_shape);
    return core::wrap_tensor(ggml_mul(ctx.ggml, input.tensor, scale_view.tensor), input.shape, GGML_TYPE_F32);
}

core::TensorValue layer_scale_channel_last(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::LayerScaleWeights & weights) {
    auto scale_view = core::reshape_tensor(
        ctx,
        weights.scale,
        core::TensorShape::from_dims({1, 1, input.shape.dims[2]}));
    return core::wrap_tensor(ggml_mul(ctx.ggml, input.tensor, scale_view.tensor), input.shape, GGML_TYPE_F32);
}

std::vector<float> precompute_frequency_embedding(
    const std::vector<float> & source_values,
    int64_t freqs,
    int64_t channels,
    float freq_scale,
    float embedding_scale) {
    if (source_values.size() != static_cast<size_t>(freqs * channels)) {
        throw std::runtime_error("HTDemucs frequency embedding size mismatch");
    }
    std::vector<float> out(static_cast<size_t>(freqs * channels));
    const float scale = freq_scale * embedding_scale;
    for (int64_t channel = 0; channel < channels; ++channel) {
        const size_t dst_base = static_cast<size_t>(channel * freqs);
        for (int64_t freq = 0; freq < freqs; ++freq) {
            out[dst_base + static_cast<size_t>(freq)] =
                source_values[static_cast<size_t>(freq * channels + channel)] * scale;
        }
    }
    return out;
}

core::TensorValue flatten_freq_time_python_order(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input) {
    auto x = modules::TransposeModule({{0, 1, 3, 2}, input.shape.rank}).build(ctx, input);
    x = ensure_contiguous(ctx, x);
    return core::reshape_tensor(
        ctx,
        x,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], input.shape.dims[2] * input.shape.dims[3]}));
}

core::TensorValue unflatten_freq_time_python_order(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t freqs,
    int64_t frames) {
    auto x = core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, input),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], frames, freqs}));
    return modules::TransposeModule({{0, 1, 3, 2}, x.shape.rank}).build(ctx, x);
}

LinearWeights packed_attention_projection(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & in_proj_weight,
    const core::TensorValue & in_proj_bias,
    int64_t hidden_size,
    int projection_index) {
    if (in_proj_weight.shape.rank != 2 || in_proj_weight.shape.dims[0] != hidden_size * 3
        || in_proj_weight.shape.dims[1] != hidden_size) {
        throw std::runtime_error("HTDemucs packed attention weight must have shape [3 * hidden_size, hidden_size]");
    }
    if (in_proj_bias.shape.rank != 1 || in_proj_bias.shape.dims[0] != hidden_size * 3) {
        throw std::runtime_error("HTDemucs packed attention bias must have shape [3 * hidden_size]");
    }
    if (projection_index < 0 || projection_index > 2) {
        throw std::runtime_error("HTDemucs packed attention projection index is out of range");
    }
    const auto view_shape = core::TensorShape::from_dims({hidden_size, hidden_size});
    const size_t row_stride = in_proj_weight.tensor->nb[1];
    const size_t weight_offset = static_cast<size_t>(projection_index) * static_cast<size_t>(hidden_size) * row_stride;
    const auto weight = core::wrap_tensor(
        ggml_view_2d(ctx.ggml, in_proj_weight.tensor, hidden_size, hidden_size, row_stride, weight_offset),
        view_shape,
        in_proj_weight.type);
    const size_t bias_offset = static_cast<size_t>(projection_index) * static_cast<size_t>(hidden_size) * in_proj_bias.tensor->nb[0];
    const auto bias = core::wrap_tensor(
        ggml_view_1d(ctx.ggml, in_proj_bias.tensor, hidden_size, bias_offset),
        core::TensorShape::from_dims({hidden_size}),
        in_proj_bias.type);
    return {weight, bias};
}

LinearWeights packed_attention_projection_range(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & in_proj_weight,
    const core::TensorValue & in_proj_bias,
    int64_t hidden_size,
    int projection_offset,
    int projection_count) {
    if (projection_offset < 0 || projection_count <= 0 || projection_offset + projection_count > 3) {
        throw std::runtime_error("HTDemucs packed attention projection range is out of range");
    }
    const int64_t out_hidden = hidden_size * projection_count;
    const auto view_shape = core::TensorShape::from_dims({out_hidden, hidden_size});
    const size_t row_stride = in_proj_weight.tensor->nb[1];
    const size_t weight_offset = static_cast<size_t>(projection_offset) * static_cast<size_t>(hidden_size) * row_stride;
    const auto weight = core::wrap_tensor(
        ggml_view_2d(ctx.ggml, in_proj_weight.tensor, hidden_size, out_hidden, row_stride, weight_offset),
        view_shape,
        in_proj_weight.type);
    const size_t bias_offset =
        static_cast<size_t>(projection_offset) * static_cast<size_t>(hidden_size) * in_proj_bias.tensor->nb[0];
    const auto bias = core::wrap_tensor(
        ggml_view_1d(ctx.ggml, in_proj_bias.tensor, out_hidden, bias_offset),
        core::TensorShape::from_dims({out_hidden}),
        in_proj_bias.type);
    return {weight, bias};
}

core::TensorValue build_self_attention_flash(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const TransformerSelfLayerWeights & weights,
    const HTDemucsConfig & config) {
    const int64_t hidden = input.shape.dims[2];
    const int64_t head_dim = hidden / config.transformer_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const modules::LinearModule qkv_proj(
        binding::linear_config(hidden, hidden * 3, true));
    const modules::LinearModule out_proj(
        binding::linear_config(hidden, hidden, true));
    const modules::MatMulModule matmul;

    auto qkv = qkv_proj.build(
        ctx,
        input,
        binding::linear_data(
            ctx,
            weights.in_proj_weight,
            std::optional<core::TensorValue>(weights.in_proj_bias)));
    auto q = modules::SliceModule({2, 0, hidden}).build(ctx, qkv);
    auto k = modules::SliceModule({2, hidden, hidden}).build(ctx, qkv);
    auto v = modules::SliceModule({2, 2 * hidden, hidden}).build(ctx, qkv);
    q = ensure_contiguous(ctx, q);
    k = ensure_contiguous(ctx, k);
    v = ensure_contiguous(ctx, v);

    q = core::reshape_tensor(ctx, q, core::TensorShape::from_dims({
        input.shape.dims[0], input.shape.dims[1], config.transformer_heads, head_dim}));
    k = core::reshape_tensor(ctx, k, core::TensorShape::from_dims({
        input.shape.dims[0], input.shape.dims[1], config.transformer_heads, head_dim}));
    v = core::reshape_tensor(ctx, v, core::TensorShape::from_dims({
        input.shape.dims[0], input.shape.dims[1], config.transformer_heads, head_dim}));

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    core::TensorValue context;
    if (ctx.backend_type == core::BackendType::Cuda) {
        q_heads = ensure_contiguous(ctx, q_heads);
        k_heads = ensure_contiguous(ctx, k_heads);
        v_heads = ensure_contiguous(ctx, v_heads);
        auto * flash = ggml_flash_attn_ext(
            ctx.ggml,
            q_heads.tensor,
            k_heads.tensor,
            v_heads.tensor,
            nullptr,
            scale,
            0.0f,
            0.0f);
        ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
        context = core::wrap_tensor(
            flash,
            core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.transformer_heads, head_dim}),
            GGML_TYPE_F32);
    } else {
        auto k_transposed = modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads);
        auto scores = matmul.build(ctx, q_heads, k_transposed);
        scores = core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, scale), scores.shape, GGML_TYPE_F32);
        auto attn = core::wrap_tensor(
            ggml_soft_max(ctx.ggml, ensure_contiguous(ctx, scores).tensor),
            scores.shape,
            GGML_TYPE_F32);
        context = matmul.build(ctx, attn, v_heads);
        context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
        context = ensure_contiguous(ctx, context);
    }
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], hidden}));
    return out_proj.build(
        ctx,
        context,
        binding::linear_data(
            ctx,
            weights.out_proj.weight,
            weights.out_proj.bias));
}

core::TensorValue build_cross_attention_flash(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & query,
    const core::TensorValue & memory,
    const TransformerCrossLayerWeights & weights,
    const HTDemucsConfig & config) {
    const int64_t hidden = query.shape.dims[2];
    const int64_t head_dim = hidden / config.transformer_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto q_weights = packed_attention_projection(ctx, weights.in_proj_weight, weights.in_proj_bias, hidden, 0);
    auto kv_weights = packed_attention_projection_range(ctx, weights.in_proj_weight, weights.in_proj_bias, hidden, 1, 2);
    const modules::LinearModule q_proj(
        binding::linear_config(hidden, hidden, true));
    const modules::LinearModule kv_proj(
        binding::linear_config(hidden, hidden * 2, true));
    const modules::LinearModule out_proj(
        binding::linear_config(hidden, hidden, true));
    const modules::MatMulModule matmul;

    auto q = q_proj.build(
        ctx,
        query,
        binding::linear_data(ctx, q_weights.weight, q_weights.bias));
    auto kv = kv_proj.build(
        ctx,
        memory,
        binding::linear_data(ctx, kv_weights.weight, kv_weights.bias));
    auto k = modules::SliceModule({2, 0, hidden}).build(ctx, kv);
    auto v = modules::SliceModule({2, hidden, hidden}).build(ctx, kv);
    q = ensure_contiguous(ctx, q);
    k = ensure_contiguous(ctx, k);
    v = ensure_contiguous(ctx, v);

    q = core::reshape_tensor(ctx, q, core::TensorShape::from_dims({
        query.shape.dims[0], query.shape.dims[1], config.transformer_heads, head_dim}));
    k = core::reshape_tensor(ctx, k, core::TensorShape::from_dims({
        memory.shape.dims[0], memory.shape.dims[1], config.transformer_heads, head_dim}));
    v = core::reshape_tensor(ctx, v, core::TensorShape::from_dims({
        memory.shape.dims[0], memory.shape.dims[1], config.transformer_heads, head_dim}));

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    core::TensorValue context;
    if (ctx.backend_type == core::BackendType::Cuda) {
        q_heads = ensure_contiguous(ctx, q_heads);
        k_heads = ensure_contiguous(ctx, k_heads);
        v_heads = ensure_contiguous(ctx, v_heads);
        auto * flash = ggml_flash_attn_ext(
            ctx.ggml,
            q_heads.tensor,
            k_heads.tensor,
            v_heads.tensor,
            nullptr,
            scale,
            0.0f,
            0.0f);
        ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
        context = core::wrap_tensor(
            flash,
            core::TensorShape::from_dims({query.shape.dims[0], query.shape.dims[1], config.transformer_heads, head_dim}),
            GGML_TYPE_F32);
    } else {
        auto k_transposed = modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads);
        auto scores = matmul.build(ctx, q_heads, k_transposed);
        scores = core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, scale), scores.shape, GGML_TYPE_F32);
        auto attn = core::wrap_tensor(
            ggml_soft_max(ctx.ggml, ensure_contiguous(ctx, scores).tensor),
            scores.shape,
            GGML_TYPE_F32);
        context = matmul.build(ctx, attn, v_heads);
        context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
        context = ensure_contiguous(ctx, context);
    }
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({query.shape.dims[0], query.shape.dims[1], hidden}));
    return out_proj.build(
        ctx,
        context,
        binding::linear_data(
            ctx,
            weights.out_proj.weight,
            weights.out_proj.bias));
}

core::TensorValue build_feed_forward(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const LinearWeights & fc1,
    const LinearWeights & fc2,
    const HTDemucsConfig & config) {
    const int64_t hidden = input.shape.dims[2];
    const int64_t intermediate = static_cast<int64_t>(std::llround(static_cast<double>(hidden) * config.transformer_hidden_scale));
    return modules::FeedForwardModule({
        hidden,
        intermediate,
        true,
        config.transformer_gelu ? modules::GeluApproximation::ExactErf : modules::GeluApproximation::Tanh,
    }).build(ctx, input, {fc1.weight, fc1.bias, fc2.weight, fc2.bias});
}

core::TensorValue build_transformer_self_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const TransformerSelfLayerWeights & weights,
    const HTDemucsConfig & config) {
    auto x = input;
    auto attn_in = modules::LayerNormModule({x.shape.dims[2], 1e-5f, true, true}).build(ctx, x, weights.norm1);
    auto attn = build_self_attention_flash(ctx, attn_in, weights, config);
    attn = layer_scale_channel_last(ctx, attn, weights.gamma1);
    x = modules::AddModule{}.build(ctx, x, attn);
    auto ff_in = modules::LayerNormModule({x.shape.dims[2], 1e-5f, true, true}).build(ctx, x, weights.norm2);
    auto ff = build_feed_forward(ctx, ff_in, weights.linear1, weights.linear2, config);
    ff = layer_scale_channel_last(ctx, ff, weights.gamma2);
    x = modules::AddModule{}.build(ctx, x, ff);
    if (config.transformer_norm_first && config.transformer_norm_out) {
        return group_norm_sequence_exact(ctx, x, 1e-5f, weights.norm_out);
    }
    return x;
}

core::TensorValue build_transformer_cross_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & query,
    const core::TensorValue & memory,
    const TransformerCrossLayerWeights & weights,
    const HTDemucsConfig & config) {
    auto x = query;
    auto q_in = modules::LayerNormModule({x.shape.dims[2], 1e-5f, true, true}).build(ctx, x, weights.norm1);
    auto m_in = modules::LayerNormModule({memory.shape.dims[2], 1e-5f, true, true}).build(ctx, memory, weights.norm2);
    auto attn = build_cross_attention_flash(ctx, q_in, m_in, weights, config);
    attn = layer_scale_channel_last(ctx, attn, weights.gamma1);
    x = modules::AddModule{}.build(ctx, x, attn);
    auto ff_in = modules::LayerNormModule({x.shape.dims[2], 1e-5f, true, true}).build(ctx, x, weights.norm3);
    auto ff = build_feed_forward(ctx, ff_in, weights.linear1, weights.linear2, config);
    ff = layer_scale_channel_last(ctx, ff, weights.gamma2);
    x = modules::AddModule{}.build(ctx, x, ff);
    if (config.transformer_norm_first && config.transformer_norm_out) {
        return group_norm_sequence_exact(ctx, x, 1e-5f, weights.norm_out);
    }
    return x;
}

core::TensorValue build_dconv(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const DConvWeights & weights) {
    auto x = input;
    for (size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
        const auto & layer = weights.layers[layer_index];
        auto y = modules::Conv1dModule({
            x.shape.dims[1],
            layer.conv1.weight.shape.dims[0],
            layer.conv1.weight.shape.dims[2],
            1,
            layer.dilation * static_cast<int>(layer.conv1.weight.shape.dims[2] / 2),
            layer.dilation,
            layer.conv1.bias.has_value(),
        }).build(ctx, x, binding::conv1d_data(ctx, layer.conv1.weight, layer.conv1.bias));
        y = group_norm_affine(ctx, y, 1, 1e-5f, layer.norm1);
        y = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, y);
        y = modules::Conv1dModule({
            y.shape.dims[1],
            layer.conv2.weight.shape.dims[0],
            1,
            1,
            0,
            1,
            layer.conv2.bias.has_value(),
        }).build(ctx, y, binding::conv1d_data(ctx, layer.conv2.weight, layer.conv2.bias));
        y = group_norm_affine(ctx, y, 1, 1e-5f, layer.norm2);
        y = glu_channels(ctx, y);
        y = layer_scale_channel_first(ctx, y, layer.scale);
        x = modules::AddModule{}.build(ctx, x, y);
    }
    return x;
}

core::TensorValue build_dconv_freq(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const DConvWeights & weights) {
    auto x = input;
    for (size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
        const auto & layer = weights.layers[layer_index];
        auto conv1_weight = core::reshape_tensor(
            ctx,
            layer.conv1.weight,
            core::TensorShape::from_dims({
                layer.conv1.weight.shape.dims[0],
                layer.conv1.weight.shape.dims[1],
                1,
                layer.conv1.weight.shape.dims[2],
            }));
        auto y = modules::Conv2dModule({
            x.shape.dims[1],
            conv1_weight.shape.dims[0],
            conv1_weight.shape.dims[2],
            conv1_weight.shape.dims[3],
            1,
            1,
            0,
            layer.dilation * static_cast<int>(layer.conv1.weight.shape.dims[2] / 2),
            1,
            layer.dilation,
            layer.conv1.bias.has_value(),
        }).build(ctx, x, binding::conv2d_data(ctx, conv1_weight, layer.conv1.bias));
        y = group_norm_freq_independent(ctx, y, 1, 1e-5f, layer.norm1);
        y = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, y);

        auto conv2_weight = core::reshape_tensor(
            ctx,
            layer.conv2.weight,
            core::TensorShape::from_dims({
                layer.conv2.weight.shape.dims[0],
                layer.conv2.weight.shape.dims[1],
                1,
                layer.conv2.weight.shape.dims[2],
            }));
        y = modules::Conv2dModule({
            y.shape.dims[1],
            conv2_weight.shape.dims[0],
            conv2_weight.shape.dims[2],
            conv2_weight.shape.dims[3],
            1,
            1,
            0,
            0,
            1,
            1,
            layer.conv2.bias.has_value(),
        }).build(ctx, y, binding::conv2d_data(ctx, conv2_weight, layer.conv2.bias));
        y = group_norm_freq_independent(ctx, y, 1, 1e-5f, layer.norm2);
        y = glu_channels(ctx, y);
        y = layer_scale_channel_first(ctx, y, layer.scale);
        x = modules::AddModule{}.build(ctx, x, y);
    }
    return x;
}

core::TensorValue build_henc_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::optional<core::TensorValue> & inject,
    const HEncLayerWeights & weights,
    const HTDemucsConfig & config) {
    (void) config;
    auto padded_input = input;
    if (!weights.freq) {
        const int64_t remainder = input.shape.dims[2] % weights.stride;
        if (remainder != 0) {
            const int64_t extra = weights.stride - remainder;
            auto tail = modules::SliceModule({2, 0, extra}).build(ctx, input);
            tail = core::wrap_tensor(ggml_scale(ctx.ggml, ensure_contiguous(ctx, tail).tensor, 0.0f), tail.shape, GGML_TYPE_F32);
            padded_input = modules::ConcatModule({2}).build(ctx, input, tail);
        }
    }
    auto y = weights.freq
        ? modules::Conv2dModule({
              padded_input.shape.dims[1],
              weights.conv2d.weight.shape.dims[0],
              weights.conv2d.weight.shape.dims[2],
              weights.conv2d.weight.shape.dims[3],
              weights.stride,
              1,
              weights.pad,
              0,
              1,
              1,
              weights.conv2d.bias.has_value(),
          }).build(ctx, padded_input, binding::conv2d_data(ctx, weights.conv2d.weight, weights.conv2d.bias))
        : modules::Conv1dModule({
              padded_input.shape.dims[1],
              weights.conv1d.weight.shape.dims[0],
              weights.conv1d.weight.shape.dims[2],
              weights.stride,
              weights.pad,
              1,
              weights.conv1d.bias.has_value(),
          }).build(ctx, padded_input, binding::conv1d_data(ctx, weights.conv1d.weight, weights.conv1d.bias));
    if (inject.has_value()) {
        y = modules::AddModule{}.build(ctx, y, *inject);
    }
    if (weights.norm) {
        y = group_norm_affine(ctx, y, config.norm_groups, 1e-5f, weights.norm1);
    }
    y = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, y);
    if (weights.dconv) {
        if (weights.freq) {
            y = build_dconv_freq(ctx, y, weights.dconv_weights);
        } else {
            y = build_dconv(ctx, y, weights.dconv_weights);
        }
    }
    if (weights.rewrite) {
        y = weights.freq
            ? modules::Conv2dModule({
                  y.shape.dims[1],
                  weights.rewrite2d.weight.shape.dims[0],
                  weights.rewrite2d.weight.shape.dims[2],
                  weights.rewrite2d.weight.shape.dims[3],
                  1,
                  1,
                  weights.context,
                  weights.context,
                  1,
                  1,
                  weights.rewrite2d.bias.has_value(),
              }).build(ctx, y, binding::conv2d_data(ctx, weights.rewrite2d.weight, weights.rewrite2d.bias))
            : modules::Conv1dModule({
                  y.shape.dims[1],
                  weights.rewrite1d.weight.shape.dims[0],
                  weights.rewrite1d.weight.shape.dims[2],
                  1,
                  weights.context,
                  1,
                  weights.rewrite1d.bias.has_value(),
              }).build(ctx, y, binding::conv1d_data(ctx, weights.rewrite1d.weight, weights.rewrite1d.bias));
        if (weights.norm) {
            y = group_norm_affine(ctx, y, config.norm_groups, 1e-5f, weights.norm2);
        }
        y = glu_channels(ctx, y);
    }
    return y;
}

core::TensorValue build_freq_conv_transpose_2d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ConvTranspose2dWeights & weights,
    int stride) {
    core::validate_rank_between(input, 4, 4, "input");
    const int64_t batch = input.shape.dims[0];
    const int64_t channels = input.shape.dims[1];
    const int64_t freqs = input.shape.dims[2];
    const int64_t frames = input.shape.dims[3];
    if (weights.weight.shape.dims[0] != channels) {
        throw std::runtime_error("HTDemucs frequency transposed-conv input channel mismatch");
    }
    const auto batch_time_shape = core::TensorShape::from_dims({batch, frames, channels, freqs});
    auto * batch_time_tensor = ggml_permute(ctx.ggml, input.tensor, 2, 0, 1, 3);
    const auto batch_time_dims = core::to_ggml_dims(batch_time_shape);
    auto x = core::wrap_tensor(
        ggml_cont_4d(
            ctx.ggml,
            batch_time_tensor,
            batch_time_dims[0],
            batch_time_dims[1],
            batch_time_dims[2],
            batch_time_dims[3]),
        batch_time_shape,
        input.type);
    x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({batch * frames, channels, 1, freqs}));

    const int64_t output_freqs = (freqs - 1) * stride + weights.weight.shape.dims[3];
    auto output = core::wrap_tensor(
        ggml_conv_transpose_2d_p0(
            ctx.ggml,
            conv_transpose2d_weight(ctx, weights.weight).tensor,
            x.tensor,
            stride),
        core::TensorShape::from_dims({batch * frames, weights.weight.shape.dims[1], 1, output_freqs}),
        GGML_TYPE_F32);
    if (weights.bias.has_value()) {
        const auto bias_view = core::reshape_tensor(
            ctx,
            *weights.bias,
            core::TensorShape::from_dims({1, weights.weight.shape.dims[1], 1, 1}));
        output = core::wrap_tensor(ggml_add(ctx.ggml, output.tensor, bias_view.tensor), output.shape, GGML_TYPE_F32);
    }
    output = core::reshape_tensor(
        ctx,
        output,
        core::TensorShape::from_dims({batch, frames, weights.weight.shape.dims[1], output_freqs}));
    const auto output_shape = core::TensorShape::from_dims({batch, weights.weight.shape.dims[1], output_freqs, frames});
    const auto output_dims = core::to_ggml_dims(output_shape);
    auto * output_tensor = ggml_permute(ctx.ggml, output.tensor, 1, 2, 0, 3);
    return core::wrap_tensor(
        ggml_cont_4d(
            ctx.ggml,
            output_tensor,
            output_dims[0],
            output_dims[1],
            output_dims[2],
            output_dims[3]),
        output_shape,
        GGML_TYPE_F32);
}

struct HDecOutputs {
    core::TensorValue output;
    core::TensorValue pre;
};

HDecOutputs build_hdec_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::optional<core::TensorValue> & skip,
    int64_t output_length,
    const HDecLayerWeights & weights,
    const HTDemucsConfig & config) {
    auto y = input;
    if (!weights.empty) {
        y = modules::AddModule{}.build(ctx, y, *skip);
        if (weights.rewrite) {
            y = weights.freq
                ? modules::Conv2dModule({
                      y.shape.dims[1],
                      weights.rewrite2d.weight.shape.dims[0],
                      weights.rewrite2d.weight.shape.dims[2],
                      weights.rewrite2d.weight.shape.dims[3],
                      1,
                      1,
                      config.context,
                      config.context,
                      1,
                      1,
                      weights.rewrite2d.bias.has_value(),
                  }).build(ctx, y, binding::conv2d_data(ctx, weights.rewrite2d.weight, weights.rewrite2d.bias))
                : modules::Conv1dModule({
                      y.shape.dims[1],
                      weights.rewrite1d.weight.shape.dims[0],
                      weights.rewrite1d.weight.shape.dims[2],
                      1,
                      config.context,
                      1,
                      weights.rewrite1d.bias.has_value(),
                  }).build(ctx, y, binding::conv1d_data(ctx, weights.rewrite1d.weight, weights.rewrite1d.bias));
            if (weights.norm) {
                y = group_norm_affine(ctx, y, config.norm_groups, 1e-5f, weights.norm1);
            }
            y = glu_channels(ctx, y);
        }
        if (weights.dconv) {
            if (weights.freq) {
                y = build_dconv_freq(ctx, y, weights.dconv_weights);
            } else {
                y = build_dconv(ctx, y, weights.dconv_weights);
            }
        }
    }

    auto z = weights.freq
        ? build_freq_conv_transpose_2d(ctx, y, weights.conv_tr_freq, config.stride)
        : modules::ConvTranspose1dModule({
              y.shape.dims[1],
              weights.conv_tr_1d.weight.shape.dims[1],
              weights.conv_tr_1d.weight.shape.dims[2],
              config.stride,
              0,
              1,
              weights.conv_tr_1d.bias.has_value(),
          }).build(ctx, y, binding::conv_transpose1d_data(ctx, weights.conv_tr_1d.weight, weights.conv_tr_1d.bias));
    if (weights.norm) {
        z = group_norm_affine(ctx, z, config.norm_groups, 1e-5f, weights.norm2);
    }
    if (weights.freq) {
        z = modules::SliceModule({2, weights.pad, z.shape.dims[2] - 2 * weights.pad}).build(ctx, z);
    } else {
        z = modules::SliceModule({2, weights.pad, output_length}).build(ctx, z);
    }
    if (!weights.last) {
        z = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, z);
    }
    return {z, y};
}

HTDemucsWeights load_weights(
    const HTDemucsSubmodelAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets_ns::TensorStorageType storage_type) {
    validate_demucs_weight_storage_type(storage_type);
    const auto & config = assets.config;
    const auto & source = *assets.tensor_source;
    HTDemucsWeights out;
    out.store = std::make_shared<core::BackendWeightStore>(
        backend, backend_type, "htdemucs.weights", 1024ull * 1024ull * 1024ull);

    int chin = config.audio_channels;
    int chin_z = config.input_freq_channels;
    int chout = config.channels;
    int chout_z = config.channels;
    int freqs = config.stft_freq_bins;
    for (int index = 0; index < config.depth; ++index) {
        const auto decoder_storage_type =
            storage_type == assets_ns::TensorStorageType::F16 && index == 0
                ? assets_ns::TensorStorageType::F32
                : storage_type;
        const bool freq = freqs > 1;
        const bool norm = index >= config.norm_starts;
        bool last_freq = false;
        int ker = config.kernel_size;
        if (freq && freqs <= config.kernel_size) {
            ker = freqs;
            last_freq = true;
        }

        HEncLayerWeights enc;
        enc.freq = true;
        enc.rewrite = config.rewrite;
        enc.dconv = (config.dconv_mode & 1) != 0;
        enc.norm = norm;
        enc.context = config.context_enc;
        enc.stride = config.stride;
        enc.pad = last_freq ? 0 : ker / 4;
        enc.conv2d = binding::conv2d_from_source(*out.store, source, "encoder." + std::to_string(index) + ".conv", storage_type, chout_z, chin_z, ker, 1, true);
        if (enc.norm) {
            enc.norm1 = binding::norm_from_source(*out.store, source, "encoder." + std::to_string(index) + ".norm1", chout_z);
        }
        if (enc.rewrite) {
            enc.rewrite2d = binding::conv2d_from_source(*out.store, source, "encoder." + std::to_string(index) + ".rewrite", storage_type, chout_z * 2, chout_z, 1 + 2 * config.context_enc, 1 + 2 * config.context_enc, true);
            if (enc.norm) {
                enc.norm2 = binding::norm_from_source(*out.store, source, "encoder." + std::to_string(index) + ".norm2", chout_z * 2);
            }
        }
        if (enc.dconv) {
            const int hidden = std::max<int>(1, chout_z / config.dconv_comp);
            for (int d = 0; d < config.dconv_depth; ++d) {
                DConvResidualWeights layer;
                const int dilation = 1 << d;
                layer.conv1 = binding::conv1d_from_source(
                    *out.store, source, "encoder." + std::to_string(index) + ".dconv.layers." + std::to_string(d) + ".0",
                    storage_type, hidden, chout_z, 3, true);
                layer.norm1 = binding::norm_from_source(*out.store, source, "encoder." + std::to_string(index) + ".dconv.layers." + std::to_string(d) + ".1", hidden);
                layer.conv2 = binding::conv1d_from_source(
                    *out.store, source, "encoder." + std::to_string(index) + ".dconv.layers." + std::to_string(d) + ".3",
                    storage_type, chout_z * 2, hidden, 1, true);
                layer.norm2 = binding::norm_from_source(*out.store, source, "encoder." + std::to_string(index) + ".dconv.layers." + std::to_string(d) + ".4", chout_z * 2);
                layer.scale = load_layer_scale(*out.store, source, "encoder." + std::to_string(index) + ".dconv.layers." + std::to_string(d) + ".6.scale", chout_z);
                layer.dilation = dilation;
                enc.dconv_weights.layers.push_back(std::move(layer));
            }
        }
        out.encoder.push_back(std::move(enc));

        HEncLayerWeights tenc;
        tenc.freq = false;
        tenc.rewrite = config.rewrite;
        tenc.dconv = (config.dconv_mode & 1) != 0;
        tenc.norm = norm;
        tenc.context = config.context_enc;
        tenc.stride = config.stride;
        tenc.pad = config.kernel_size / 4;
        tenc.conv1d = binding::conv1d_from_source(*out.store, source, "tencoder." + std::to_string(index) + ".conv", storage_type, chout, chin, config.kernel_size, true);
        if (tenc.norm) {
            tenc.norm1 = binding::norm_from_source(*out.store, source, "tencoder." + std::to_string(index) + ".norm1", chout);
        }
        if (tenc.rewrite) {
            tenc.rewrite1d = binding::conv1d_from_source(*out.store, source, "tencoder." + std::to_string(index) + ".rewrite", storage_type, chout * 2, chout, 1 + 2 * config.context_enc, true);
            if (tenc.norm) {
                tenc.norm2 = binding::norm_from_source(*out.store, source, "tencoder." + std::to_string(index) + ".norm2", chout * 2);
            }
        }
        if (tenc.dconv) {
            const int hidden = std::max<int>(1, chout / config.dconv_comp);
            for (int d = 0; d < config.dconv_depth; ++d) {
                DConvResidualWeights layer;
                layer.conv1 = binding::conv1d_from_source(
                    *out.store, source, "tencoder." + std::to_string(index) + ".dconv.layers." + std::to_string(d) + ".0",
                    storage_type, hidden, chout, 3, true);
                layer.norm1 = binding::norm_from_source(*out.store, source, "tencoder." + std::to_string(index) + ".dconv.layers." + std::to_string(d) + ".1", hidden);
                layer.conv2 = binding::conv1d_from_source(
                    *out.store, source, "tencoder." + std::to_string(index) + ".dconv.layers." + std::to_string(d) + ".3",
                    storage_type, chout * 2, hidden, 1, true);
                layer.norm2 = binding::norm_from_source(*out.store, source, "tencoder." + std::to_string(index) + ".dconv.layers." + std::to_string(d) + ".4", chout * 2);
                layer.scale = load_layer_scale(*out.store, source, "tencoder." + std::to_string(index) + ".dconv.layers." + std::to_string(d) + ".6.scale", chout);
                layer.dilation = 1 << d;
                tenc.dconv_weights.layers.push_back(std::move(layer));
            }
        }
        out.tencoder.push_back(std::move(tenc));

        if (index == 0) {
            chin = config.output_time_channels;
            chin_z = config.output_freq_channels;
        }

        const int dec_index = config.depth - 1 - index;
        const std::string dec_prefix = "decoder." + std::to_string(dec_index);
        const std::string tdec_prefix = "tdecoder." + std::to_string(dec_index);

        HDecLayerWeights dec;
        dec.freq = true;
        dec.last = index == 0;
        dec.rewrite = config.rewrite;
        dec.dconv = (config.dconv_mode & 2) != 0;
        dec.norm = norm;
        dec.pad = ker / 4;
        dec.conv_tr_freq = load_conv_transpose2d(
            *out.store, source, dec_prefix + ".conv_tr", decoder_storage_type, chout_z, chin_z, ker, true);
        if (dec.norm) {
            dec.norm2 = binding::norm_from_source(*out.store, source, dec_prefix + ".norm2", chin_z);
        }
        if (dec.rewrite) {
            dec.rewrite2d = binding::conv2d_from_source(*out.store, source, dec_prefix + ".rewrite", decoder_storage_type, chout_z * 2, chout_z, 1 + 2 * config.context, 1 + 2 * config.context, true);
            if (dec.norm) {
                dec.norm1 = binding::norm_from_source(*out.store, source, dec_prefix + ".norm1", chout_z * 2);
            }
        }
        if (dec.dconv) {
            const int hidden = std::max<int>(1, chout_z / config.dconv_comp);
            for (int d = 0; d < config.dconv_depth; ++d) {
                DConvResidualWeights layer;
                layer.conv1 = binding::conv1d_from_source(
                    *out.store, source, dec_prefix + ".dconv.layers." + std::to_string(d) + ".0",
                    decoder_storage_type, hidden, chout_z, 3, true);
                layer.norm1 = binding::norm_from_source(*out.store, source, dec_prefix + ".dconv.layers." + std::to_string(d) + ".1", hidden);
                layer.conv2 = binding::conv1d_from_source(
                    *out.store, source, dec_prefix + ".dconv.layers." + std::to_string(d) + ".3",
                    decoder_storage_type, chout_z * 2, hidden, 1, true);
                layer.norm2 = binding::norm_from_source(*out.store, source, dec_prefix + ".dconv.layers." + std::to_string(d) + ".4", chout_z * 2);
                layer.scale = load_layer_scale(*out.store, source, dec_prefix + ".dconv.layers." + std::to_string(d) + ".6.scale", chout_z);
                layer.dilation = 1 << d;
                dec.dconv_weights.layers.push_back(std::move(layer));
            }
        }
        out.decoder.insert(out.decoder.begin(), std::move(dec));

        HDecLayerWeights tdec;
        tdec.freq = false;
        tdec.last = index == 0;
        tdec.rewrite = config.rewrite;
        tdec.dconv = (config.dconv_mode & 2) != 0;
        tdec.norm = norm;
        tdec.pad = config.kernel_size / 4;
        tdec.conv_tr_1d = binding::conv_transpose1d_from_source(
            *out.store, source, tdec_prefix + ".conv_tr", decoder_storage_type, chout, chin, config.kernel_size, true);
        if (tdec.norm) {
            tdec.norm2 = binding::norm_from_source(*out.store, source, tdec_prefix + ".norm2", chin);
        }
        if (tdec.rewrite) {
            tdec.rewrite1d = binding::conv1d_from_source(*out.store, source, tdec_prefix + ".rewrite", decoder_storage_type, chout * 2, chout, 1 + 2 * config.context, true);
            if (tdec.norm) {
                tdec.norm1 = binding::norm_from_source(*out.store, source, tdec_prefix + ".norm1", chout * 2);
            }
        }
        if (tdec.dconv) {
            const int hidden = std::max<int>(1, chout / config.dconv_comp);
            for (int d = 0; d < config.dconv_depth; ++d) {
                DConvResidualWeights layer;
                layer.conv1 = binding::conv1d_from_source(
                    *out.store, source, tdec_prefix + ".dconv.layers." + std::to_string(d) + ".0",
                    decoder_storage_type, hidden, chout, 3, true);
                layer.norm1 = binding::norm_from_source(*out.store, source, tdec_prefix + ".dconv.layers." + std::to_string(d) + ".1", hidden);
                layer.conv2 = binding::conv1d_from_source(
                    *out.store, source, tdec_prefix + ".dconv.layers." + std::to_string(d) + ".3",
                    decoder_storage_type, chout * 2, hidden, 1, true);
                layer.norm2 = binding::norm_from_source(*out.store, source, tdec_prefix + ".dconv.layers." + std::to_string(d) + ".4", chout * 2);
                layer.scale = load_layer_scale(*out.store, source, tdec_prefix + ".dconv.layers." + std::to_string(d) + ".6.scale", chout);
                layer.dilation = 1 << d;
                tdec.dconv_weights.layers.push_back(std::move(layer));
            }
        }
        out.tdecoder.insert(out.tdecoder.begin(), std::move(tdec));

        chin = chout;
        chin_z = chout_z;
        chout *= config.growth;
        chout_z *= config.growth;
        if (freq) {
            freqs = freqs <= config.kernel_size ? 1 : freqs / config.stride;
        }
    }

    out.freq_embedding_values = precompute_frequency_embedding(
        source.require_f32(
            "freq_emb.embedding.weight",
            {config.stft_freq_bins / config.stride, config.channels}),
        config.stft_freq_bins / config.stride,
        config.channels,
        config.freq_emb_scale,
        config.embedding_scale);

    const int transformer_in_channels = config.channels * static_cast<int>(std::pow(config.growth, config.depth - 1));
    out.channel_upsampler = binding::conv1d_from_source(*out.store, source, "channel_upsampler", storage_type, config.bottom_channels, transformer_in_channels, 1, true);
    out.channel_downsampler = binding::conv1d_from_source(*out.store, source, "channel_downsampler", storage_type, transformer_in_channels, config.bottom_channels, 1, true);
    out.channel_upsampler_t = binding::conv1d_from_source(*out.store, source, "channel_upsampler_t", storage_type, config.bottom_channels, transformer_in_channels, 1, true);
    out.channel_downsampler_t = binding::conv1d_from_source(*out.store, source, "channel_downsampler_t", storage_type, transformer_in_channels, config.bottom_channels, 1, true);

    if (config.transformer_layers > 0) {
        CrossTransformerWeights tr;
        tr.norm_in_freq = binding::norm_from_source(*out.store, source, "crosstransformer.norm_in", config.bottom_channels);
        tr.norm_in_time = binding::norm_from_source(*out.store, source, "crosstransformer.norm_in_t", config.bottom_channels);
        for (int i = 0; i < config.transformer_layers; ++i) {
            const bool cross = (i % 2) != (config.transformer_cross_first ? 1 : 0);
            if (!cross) {
                TransformerLayerWeights layer;
                layer.cross = false;
                const std::string prefix = "crosstransformer.layers." + std::to_string(i);
                layer.self.norm1 = binding::norm_from_source(*out.store, source, prefix + ".norm1", config.bottom_channels);
                layer.self.in_proj_weight = out.store->load_tensor(source, prefix + ".self_attn.in_proj_weight", storage_type, {config.bottom_channels * 3, config.bottom_channels});
                layer.self.in_proj_bias = out.store->load_f32_tensor(source, prefix + ".self_attn.in_proj_bias", {config.bottom_channels * 3});
                layer.self.out_proj = binding::linear_from_source(*out.store, source, prefix + ".self_attn.out_proj", storage_type, config.bottom_channels, config.bottom_channels, true);
                layer.self.norm2 = binding::norm_from_source(*out.store, source, prefix + ".norm2", config.bottom_channels);
                layer.self.linear1 = binding::linear_from_source(*out.store, source, prefix + ".linear1", storage_type, static_cast<int64_t>(config.bottom_channels * config.transformer_hidden_scale), config.bottom_channels, true);
                layer.self.linear2 = binding::linear_from_source(*out.store, source, prefix + ".linear2", storage_type, config.bottom_channels, static_cast<int64_t>(config.bottom_channels * config.transformer_hidden_scale), true);
                layer.self.gamma1 = load_layer_scale(*out.store, source, prefix + ".gamma_1.scale", config.bottom_channels);
                layer.self.gamma2 = load_layer_scale(*out.store, source, prefix + ".gamma_2.scale", config.bottom_channels);
                layer.self.norm_out = binding::norm_from_source(*out.store, source, prefix + ".norm_out", config.bottom_channels);
                tr.freq_layers.push_back(std::move(layer));

                TransformerLayerWeights layer_t;
                layer_t.cross = false;
                const std::string prefix_t = "crosstransformer.layers_t." + std::to_string(i);
                layer_t.self.norm1 = binding::norm_from_source(*out.store, source, prefix_t + ".norm1", config.bottom_channels);
                layer_t.self.in_proj_weight = out.store->load_tensor(source, prefix_t + ".self_attn.in_proj_weight", storage_type, {config.bottom_channels * 3, config.bottom_channels});
                layer_t.self.in_proj_bias = out.store->load_f32_tensor(source, prefix_t + ".self_attn.in_proj_bias", {config.bottom_channels * 3});
                layer_t.self.out_proj = binding::linear_from_source(*out.store, source, prefix_t + ".self_attn.out_proj", storage_type, config.bottom_channels, config.bottom_channels, true);
                layer_t.self.norm2 = binding::norm_from_source(*out.store, source, prefix_t + ".norm2", config.bottom_channels);
                layer_t.self.linear1 = binding::linear_from_source(*out.store, source, prefix_t + ".linear1", storage_type, static_cast<int64_t>(config.bottom_channels * config.transformer_hidden_scale), config.bottom_channels, true);
                layer_t.self.linear2 = binding::linear_from_source(*out.store, source, prefix_t + ".linear2", storage_type, config.bottom_channels, static_cast<int64_t>(config.bottom_channels * config.transformer_hidden_scale), true);
                layer_t.self.gamma1 = load_layer_scale(*out.store, source, prefix_t + ".gamma_1.scale", config.bottom_channels);
                layer_t.self.gamma2 = load_layer_scale(*out.store, source, prefix_t + ".gamma_2.scale", config.bottom_channels);
                layer_t.self.norm_out = binding::norm_from_source(*out.store, source, prefix_t + ".norm_out", config.bottom_channels);
                tr.time_layers.push_back(std::move(layer_t));
            } else {
                TransformerLayerWeights layer;
                layer.cross = true;
                const std::string prefix = "crosstransformer.layers." + std::to_string(i);
                layer.cross_weights.norm1 = binding::norm_from_source(*out.store, source, prefix + ".norm1", config.bottom_channels);
                layer.cross_weights.norm2 = binding::norm_from_source(*out.store, source, prefix + ".norm2", config.bottom_channels);
                layer.cross_weights.in_proj_weight = out.store->load_tensor(source, prefix + ".cross_attn.in_proj_weight", storage_type, {config.bottom_channels * 3, config.bottom_channels});
                layer.cross_weights.in_proj_bias = out.store->load_f32_tensor(source, prefix + ".cross_attn.in_proj_bias", {config.bottom_channels * 3});
                layer.cross_weights.out_proj = binding::linear_from_source(*out.store, source, prefix + ".cross_attn.out_proj", storage_type, config.bottom_channels, config.bottom_channels, true);
                layer.cross_weights.norm3 = binding::norm_from_source(*out.store, source, prefix + ".norm3", config.bottom_channels);
                layer.cross_weights.linear1 = binding::linear_from_source(*out.store, source, prefix + ".linear1", storage_type, static_cast<int64_t>(config.bottom_channels * config.transformer_hidden_scale), config.bottom_channels, true);
                layer.cross_weights.linear2 = binding::linear_from_source(*out.store, source, prefix + ".linear2", storage_type, config.bottom_channels, static_cast<int64_t>(config.bottom_channels * config.transformer_hidden_scale), true);
                layer.cross_weights.gamma1 = load_layer_scale(*out.store, source, prefix + ".gamma_1.scale", config.bottom_channels);
                layer.cross_weights.gamma2 = load_layer_scale(*out.store, source, prefix + ".gamma_2.scale", config.bottom_channels);
                layer.cross_weights.norm_out = binding::norm_from_source(*out.store, source, prefix + ".norm_out", config.bottom_channels);
                tr.freq_layers.push_back(std::move(layer));

                TransformerLayerWeights layer_t;
                layer_t.cross = true;
                const std::string prefix_t = "crosstransformer.layers_t." + std::to_string(i);
                layer_t.cross_weights.norm1 = binding::norm_from_source(*out.store, source, prefix_t + ".norm1", config.bottom_channels);
                layer_t.cross_weights.norm2 = binding::norm_from_source(*out.store, source, prefix_t + ".norm2", config.bottom_channels);
                layer_t.cross_weights.in_proj_weight = out.store->load_tensor(source, prefix_t + ".cross_attn.in_proj_weight", storage_type, {config.bottom_channels * 3, config.bottom_channels});
                layer_t.cross_weights.in_proj_bias = out.store->load_f32_tensor(source, prefix_t + ".cross_attn.in_proj_bias", {config.bottom_channels * 3});
                layer_t.cross_weights.out_proj = binding::linear_from_source(*out.store, source, prefix_t + ".cross_attn.out_proj", storage_type, config.bottom_channels, config.bottom_channels, true);
                layer_t.cross_weights.norm3 = binding::norm_from_source(*out.store, source, prefix_t + ".norm3", config.bottom_channels);
                layer_t.cross_weights.linear1 = binding::linear_from_source(*out.store, source, prefix_t + ".linear1", storage_type, static_cast<int64_t>(config.bottom_channels * config.transformer_hidden_scale), config.bottom_channels, true);
                layer_t.cross_weights.linear2 = binding::linear_from_source(*out.store, source, prefix_t + ".linear2", storage_type, config.bottom_channels, static_cast<int64_t>(config.bottom_channels * config.transformer_hidden_scale), true);
                layer_t.cross_weights.gamma1 = load_layer_scale(*out.store, source, prefix_t + ".gamma_1.scale", config.bottom_channels);
                layer_t.cross_weights.gamma2 = load_layer_scale(*out.store, source, prefix_t + ".gamma_2.scale", config.bottom_channels);
                layer_t.cross_weights.norm_out = binding::norm_from_source(*out.store, source, prefix_t + ".norm_out", config.bottom_channels);
                tr.time_layers.push_back(std::move(layer_t));
            }
        }
        out.transformer = std::move(tr);
    }

    out.store->upload();
    return out;
}

std::vector<float> create_1d_sin_embedding(int length, int dim, float max_period) {
    if (dim % 2 != 0) {
        throw std::runtime_error("HTDemucs 1D sinusoidal embedding requires an even hidden size");
    }
    const int half = dim / 2;
    std::vector<float> out(static_cast<size_t>(length * dim), 0.0f);
    for (int t = 0; t < length; ++t) {
        for (int i = 0; i < half; ++i) {
            const float phase = static_cast<float>(t) / std::pow(max_period, static_cast<float>(i) / static_cast<float>(half - 1));
            out[static_cast<size_t>(t * dim + i)] = std::cos(phase);
            out[static_cast<size_t>(t * dim + half + i)] = std::sin(phase);
        }
    }
    return out;
}

std::vector<float> create_2d_sin_embedding(int dim, int height, int width, float max_period) {
    if (dim % 4 != 0) {
        throw std::runtime_error("HTDemucs 2D sinusoidal embedding requires hidden size divisible by 4");
    }
    std::vector<float> out(static_cast<size_t>(height * width * dim), 0.0f);
    const int half = dim / 2;
    const int quarter = half / 2;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            const size_t base = static_cast<size_t>((w * height + h) * dim);
            for (int i = 0; i < quarter; ++i) {
                const float div = std::exp(-std::log(max_period) * (2.0f * i) / static_cast<float>(half));
                out[base + 2 * i] = std::sin(w * div);
                out[base + 2 * i + 1] = std::cos(w * div);
                out[base + half + 2 * i] = std::sin(h * div);
                out[base + half + 2 * i + 1] = std::cos(h * div);
            }
        }
    }
    return out;
}

struct FixedShapeGraph {
    virtual ~FixedShapeGraph() {
        reset_graph_storage();
    }

protected:
    void reset_graph_storage() {
        if (graph_ != nullptr) {
            engine::core::release_backend_graph_resources(backend_, graph_);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        ctx_.reset();
        outputs_.clear();
    }

    FixedShapeGraph(ggml_backend_t backend, int threads)
        : backend_(backend),
          threads_(std::max(1, threads)) {
    }

    void init_context(size_t bytes) {
        ggml_init_params params{bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize HTDemucs ggml context");
        }
    }

    core::ModuleBuildContext make_build_context(core::ExecutionContext & execution_context, const char * name) {
        return core::ModuleBuildContext{ctx_.get(), name, execution_context.backend_type()};
    }

    void finalize_graph(size_t max_nodes) {
        graph_ = ggml_new_graph_custom(ctx_.get(), max_nodes, false);
        for (auto * output : outputs_) {
            ggml_set_output(output);
            ggml_build_forward_expand(graph_, output);
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate HTDemucs graph buffer");
        }
    }

    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
    struct GgmlContextDeleter {
        void operator()(ggml_context * ctx) const noexcept {
            if (ctx != nullptr) {
                ggml_free(ctx);
            }
        }
    };
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    std::vector<ggml_tensor *> outputs_;
};

class HTDemucsGraph final : public FixedShapeGraph {
public:
    struct Timing {
        double rebuild_ms = 0.0;
        double upload_ms = 0.0;
        double compute_ms = 0.0;
        double readback_ms = 0.0;
    };

    HTDemucsGraph(
        std::shared_ptr<const HTDemucsSubmodelAssets> assets,
        core::ExecutionContext & execution_context,
        assets_ns::TensorStorageType storage_type)
        : FixedShapeGraph(execution_context.backend(), execution_context.config().threads),
          backend_type_(execution_context.backend_type()) {
        if (assets == nullptr) {
            throw std::runtime_error("HTDemucs graph requires submodel assets");
        }
        base_config_ = assets->config;
        weights_ = load_weights(*assets, execution_context.backend(), execution_context.backend_type(), storage_type);
        rebuild();
    }

    void rebuild() {
        const auto rebuild_start = std::chrono::steady_clock::now();
        if (ctx_ != nullptr && graph_ != nullptr) {
            last_timing_.rebuild_ms = 0.0;
            return;
        }
        reset_graph_storage();
        config_ = base_config_;
        config_.stft_frames = static_cast<int>(std::ceil(
            static_cast<double>(config_.segment_samples) / static_cast<double>(config_.hop_length)));

        init_context(768ull * 1024ull * 1024ull);
        constants_ = std::make_unique<common::ConstantTensorCache>(
            backend_,
            threads_,
            "htdemucs.constants",
            128ull * 1024ull * 1024ull);
        auto ctx = core::ModuleBuildContext{ctx_.get(), "htdemucs.graph", backend_type_};
        constants_->begin_graph();

        freq_input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({
            1,
            config_.input_freq_channels,
            config_.stft_freq_bins,
            config_.stft_frames,
        })).tensor;
        time_input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({
            1,
            config_.audio_channels,
            config_.segment_samples,
        })).tensor;

        auto x = core::wrap_tensor(
            freq_input_,
            core::TensorShape::from_dims({1, config_.input_freq_channels, config_.stft_freq_bins, config_.stft_frames}),
            GGML_TYPE_F32);
        auto xt = core::wrap_tensor(
            time_input_,
            core::TensorShape::from_dims({1, config_.audio_channels, config_.segment_samples}),
            GGML_TYPE_F32);

        std::vector<core::TensorValue> saved;
        std::vector<core::TensorValue> saved_t;
        std::vector<int64_t> lengths_t;
        for (int idx = 0; idx < config_.depth; ++idx) {
            lengths_t.push_back(xt.shape.dims[2]);
            xt = build_henc_layer(ctx, xt, std::nullopt, weights_.tencoder[static_cast<size_t>(idx)], config_);
            saved_t.push_back(xt);
            x = build_henc_layer(
                ctx,
                x,
                std::nullopt,
                weights_.encoder[static_cast<size_t>(idx)],
                config_);
            if (idx == 0 && weights_.freq_embedding_values.has_value()) {
                auto freq_emb = constants_->make_f32(
                    core::TensorShape::from_dims({1, x.shape.dims[1], x.shape.dims[2], 1}),
                    *weights_.freq_embedding_values);
                x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, freq_emb.tensor), x.shape, GGML_TYPE_F32);
            }
            saved.push_back(x);
        }

        const int64_t freq_channels = x.shape.dims[1];
        const int64_t time_channels = xt.shape.dims[1];
        if (!weights_.channel_upsampler.has_value() || !weights_.transformer.has_value()) {
            throw std::runtime_error("HTDemucs graph requires transformer weights");
        }
        auto x_flat = flatten_freq_time_python_order(ctx, x);
        x_flat = modules::Conv1dModule({
            freq_channels,
            config_.bottom_channels,
            1,
            1,
            0,
            1,
            weights_.channel_upsampler->bias.has_value(),
        }).build(ctx, x_flat, binding::conv1d_data(ctx, weights_.channel_upsampler->weight, weights_.channel_upsampler->bias));
        x = unflatten_freq_time_python_order(ctx, x_flat, saved.back().shape.dims[2], saved.back().shape.dims[3]);
        xt = modules::Conv1dModule({
            time_channels,
            config_.bottom_channels,
            1,
            1,
            0,
            1,
            weights_.channel_upsampler_t->bias.has_value(),
        }).build(ctx, xt, binding::conv1d_data(ctx, weights_.channel_upsampler_t->weight, weights_.channel_upsampler_t->bias));

        const auto freq_pos_host = create_2d_sin_embedding(config_.bottom_channels, static_cast<int>(x.shape.dims[2]), static_cast<int>(x.shape.dims[3]), config_.transformer_max_period);
        auto freq_pos = constants_->make_f32(
            core::TensorShape::from_dims({1, x.shape.dims[2] * x.shape.dims[3], config_.bottom_channels}),
            freq_pos_host);
        const auto time_pos_host = create_1d_sin_embedding(static_cast<int>(xt.shape.dims[2]), config_.bottom_channels, config_.transformer_max_period);
        auto time_pos = constants_->make_f32(
            core::TensorShape::from_dims({1, xt.shape.dims[2], config_.bottom_channels}),
            time_pos_host);

        auto xf = modules::TransposeModule({{0, 3, 2, 1}, x.shape.rank}).build(ctx, x);
        xf = ensure_contiguous(ctx, xf);
        xf = core::reshape_tensor(ctx, xf, core::TensorShape::from_dims({1, x.shape.dims[2] * x.shape.dims[3], config_.bottom_channels}));
        xf = modules::LayerNormModule({config_.bottom_channels, 1e-5f, true, true}).build(ctx, xf, weights_.transformer->norm_in_freq);
        xf = modules::AddModule{}.build(ctx, xf, core::wrap_tensor(ggml_scale(ctx.ggml, freq_pos.tensor, config_.transformer_weight_pos_embed), freq_pos.shape, GGML_TYPE_F32));

        auto xtf = modules::TransposeModule({{0, 2, 1}, xt.shape.rank}).build(ctx, xt);
        xtf = modules::LayerNormModule({config_.bottom_channels, 1e-5f, true, true}).build(ctx, xtf, weights_.transformer->norm_in_time);
        xtf = modules::AddModule{}.build(ctx, xtf, core::wrap_tensor(ggml_scale(ctx.ggml, time_pos.tensor, config_.transformer_weight_pos_embed), time_pos.shape, GGML_TYPE_F32));

        for (int i = 0; i < config_.transformer_layers; ++i) {
            const auto & f_layer = weights_.transformer->freq_layers[static_cast<size_t>(i)];
            const auto & t_layer = weights_.transformer->time_layers[static_cast<size_t>(i)];
            if (!f_layer.cross) {
                xf = build_transformer_self_layer(ctx, xf, f_layer.self, config_);
                xtf = build_transformer_self_layer(ctx, xtf, t_layer.self, config_);
            } else {
                auto old_xf = xf;
                xf = build_transformer_cross_layer(ctx, xf, xtf, f_layer.cross_weights, config_);
                xtf = build_transformer_cross_layer(ctx, xtf, old_xf, t_layer.cross_weights, config_);
            }
        }

        xf = core::reshape_tensor(ctx, ensure_contiguous(ctx, xf), core::TensorShape::from_dims({1, x.shape.dims[3], x.shape.dims[2], config_.bottom_channels}));
        xf = modules::TransposeModule({{0, 3, 2, 1}, xf.shape.rank}).build(ctx, xf);
        xtf = modules::TransposeModule({{0, 2, 1}, xtf.shape.rank}).build(ctx, xtf);
        xf = flatten_freq_time_python_order(ctx, xf);
        xf = modules::Conv1dModule({
            config_.bottom_channels,
            freq_channels,
            1,
            1,
            0,
            1,
            weights_.channel_downsampler->bias.has_value(),
        }).build(ctx, xf, binding::conv1d_data(ctx, weights_.channel_downsampler->weight, weights_.channel_downsampler->bias));
        x = unflatten_freq_time_python_order(ctx, xf, saved.back().shape.dims[2], saved.back().shape.dims[3]);

        xt = modules::Conv1dModule({
            config_.bottom_channels,
            time_channels,
            1,
            1,
            0,
            1,
            weights_.channel_downsampler_t->bias.has_value(),
        }).build(ctx, xtf, binding::conv1d_data(ctx, weights_.channel_downsampler_t->weight, weights_.channel_downsampler_t->bias));

        for (int idx = 0; idx < config_.depth; ++idx) {
            auto skip = saved.back();
            saved.pop_back();
            auto freq_dec = build_hdec_layer(ctx, x, skip, 0, weights_.decoder[static_cast<size_t>(idx)], config_);
            x = freq_dec.output;
            auto tskip = saved_t.back();
            saved_t.pop_back();
            const auto time_dec = build_hdec_layer(ctx, xt, tskip, lengths_t.back(), weights_.tdecoder[static_cast<size_t>(idx)], config_);
            lengths_t.pop_back();
            xt = time_dec.output;
        }

        freq_output_ = ensure_contiguous(ctx, x).tensor;
        time_output_ = ensure_contiguous(ctx, xt).tensor;
        outputs_ = {
            freq_output_,
            time_output_,
        };
        constants_->finish_graph();
        constants_->ensure_uploaded();
        finalize_graph(262144);
        const auto rebuild_end = std::chrono::steady_clock::now();
        last_timing_.rebuild_ms = debug::elapsed_ms(rebuild_start, rebuild_end);
    }

    ~HTDemucsGraph() override = default;

    void run(
        const std::vector<float> & freq_input,
        const std::vector<float> & time_input,
        std::vector<float> & freq_output,
        std::vector<float> & time_output) {
        const size_t freq_elements = static_cast<size_t>(
            config_.input_freq_channels * config_.stft_freq_bins * config_.stft_frames);
        const size_t time_elements = static_cast<size_t>(
            config_.audio_channels * config_.segment_samples);
        if (freq_input.size() != freq_elements || time_input.size() != time_elements) {
            throw std::runtime_error("HTDemucs graph input size mismatch");
        }
        rebuild();
        const auto upload_start = std::chrono::steady_clock::now();
        ggml_backend_tensor_set(freq_input_, freq_input.data(), 0, freq_input.size() * sizeof(float));
        ggml_backend_tensor_set(time_input_, time_input.data(), 0, time_input.size() * sizeof(float));
        const auto upload_end = std::chrono::steady_clock::now();
        const auto compute_start = upload_end;
        core::set_backend_threads(backend_, threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        const auto compute_end = std::chrono::steady_clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("HTDemucs graph compute failed");
        }
        const auto readback_start = compute_end;
        if (freq_output.size() != static_cast<size_t>(
                config_.output_freq_channels * config_.stft_freq_bins * config_.stft_frames) ||
            time_output.size() != static_cast<size_t>(
                config_.output_time_channels * config_.segment_samples)) {
            throw std::runtime_error("HTDemucs graph output buffer size mismatch");
        }
        ggml_backend_tensor_get(freq_output_, freq_output.data(), 0, freq_output.size() * sizeof(float));
        ggml_backend_tensor_get(time_output_, time_output.data(), 0, time_output.size() * sizeof(float));
        const auto readback_end = std::chrono::steady_clock::now();
        last_timing_.upload_ms = debug::elapsed_ms(upload_start, upload_end);
        last_timing_.compute_ms = debug::elapsed_ms(compute_start, compute_end);
        last_timing_.readback_ms = debug::elapsed_ms(readback_start, readback_end);
    }

    const Timing & last_timing() const noexcept { return last_timing_; }

private:
    core::BackendType backend_type_ = core::BackendType::Cpu;
    HTDemucsConfig base_config_;
    HTDemucsConfig config_;
    HTDemucsWeights weights_;
    std::unique_ptr<common::ConstantTensorCache> constants_;
    ggml_tensor * freq_input_ = nullptr;
    ggml_tensor * time_input_ = nullptr;
    ggml_tensor * freq_output_ = nullptr;
    ggml_tensor * time_output_ = nullptr;
    Timing last_timing_;
};

}  // namespace

class HTDemucsPipeline::Impl {
public:
    HTDemucsConfig config;
    size_t fft_threads = 1;
    std::unique_ptr<HTDemucsGraph> graph;
    HTDemucsFrontend frontend;
    HTDemucsPostprocessor postprocessor;
    std::vector<float> freq_output;
    std::vector<float> time_output;
    std::vector<float> chunk_output;
    HTDemucsChunkTiming last_timing;

    Impl(const HTDemucsConfig & graph_config, size_t graph_fft_threads)
        : config(graph_config),
          fft_threads(graph_fft_threads),
          frontend(config, fft_threads),
          postprocessor(config, fft_threads, frontend.stft_window()) {
        freq_output.resize(static_cast<size_t>(
            config.output_freq_channels * config.stft_freq_bins * config.stft_frames));
        time_output.resize(static_cast<size_t>(
            config.output_time_channels * config.segment_samples));
        chunk_output.resize(static_cast<size_t>(
            config.sources.size() * config.audio_channels * config.segment_samples));
    }
};

HTDemucsPipeline::HTDemucsPipeline(
    std::shared_ptr<const HTDemucsSubmodelAssets> assets,
    core::ExecutionContext & execution_context,
    assets::TensorStorageType weight_storage_type)
    : assets_(std::move(assets)) {
    validate_demucs_weight_storage_type(weight_storage_type);
    if (assets_ == nullptr) {
        throw std::runtime_error("HTDemucs pipeline requires submodel assets");
    }
    impl_ = std::make_unique<Impl>(
        assets_->config,
        static_cast<size_t>(std::max(1, execution_context.config().threads)));
    impl_->graph = std::make_unique<HTDemucsGraph>(assets_, execution_context, weight_storage_type);
}

HTDemucsPipeline::~HTDemucsPipeline() = default;

const HTDemucsConfig & HTDemucsPipeline::config() const noexcept {
    return impl_->config;
}

const std::vector<float> & HTDemucsPipeline::separate_chunk(std::vector<float> & chunk_planar) {
    const auto frontend_start = std::chrono::steady_clock::now();
    impl_->frontend.prepare_chunk(chunk_planar);
    const auto frontend_end = std::chrono::steady_clock::now();

    const auto graph_start = std::chrono::steady_clock::now();
    impl_->graph->run(
        impl_->frontend.freq_input(),
        impl_->frontend.time_input(),
        impl_->freq_output,
        impl_->time_output);
    const auto graph_end = std::chrono::steady_clock::now();
    const auto postprocess_start = std::chrono::steady_clock::now();
    impl_->postprocessor.combine_chunk_into(
        impl_->freq_output.data(),
        impl_->time_output.data(),
        impl_->frontend.input_samples(),
        impl_->frontend.freq_mean(),
        impl_->frontend.freq_std(),
        impl_->frontend.time_mean(),
        impl_->frontend.time_std(),
        impl_->chunk_output.data());
    const auto postprocess_end = std::chrono::steady_clock::now();
    const auto & graph_timing = impl_->graph->last_timing();

    impl_->last_timing.frontend_ms = debug::elapsed_ms(frontend_start, frontend_end);
    impl_->last_timing.graph_ms = debug::elapsed_ms(graph_start, graph_end);
    impl_->last_timing.postprocess_ms = debug::elapsed_ms(postprocess_start, postprocess_end);
    impl_->last_timing.graph_rebuild_ms = graph_timing.rebuild_ms;
    return impl_->chunk_output;
}

const HTDemucsChunkTiming & HTDemucsPipeline::last_timing() const noexcept {
    return impl_->last_timing;
}

}  // namespace engine::models::demucs
