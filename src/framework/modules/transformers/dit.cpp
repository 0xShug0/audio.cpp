#include "engine/framework/modules/transformers/dit.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/projected_grouped_self_attention.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/packed_linear_weights.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::modules {
namespace {
using core::TensorShape;
using core::TensorValue;

std::string expand_layer(std::string name, int64_t layer) {
    const std::string token = "{layer}";
    const auto at = name.find(token);
    if (at != std::string::npos) name.replace(at, token.size(), std::to_string(layer));
    return name;
}

LinearWeights load_projection(core::BackendWeightStore & store, const assets::TensorSource & source,
    const DitLinearNames & names, assets::TensorStorageType storage, int64_t layer, bool autocast, core::BackendType backend) {
    if (names.weight.empty()) return {};
    const auto weight_name = expand_layer(names.weight, layer);
    const auto shape = source.require_metadata(weight_name).shape;
    if (shape.size() != 2 && !(shape.size() == 3 && shape[2] == 1))
        throw std::runtime_error("DiT projection must be a matrix or a kernel-one convolution: " + weight_name);
    if (autocast && storage == assets::TensorStorageType::Native &&
        backend == core::BackendType::Vulkan)
        storage = assets::resolve_tensor_storage_type(source, weight_name, storage);
    LinearWeights out;
    out.weight = shape.size() == 2
        ? store.load_tensor(source, weight_name, storage, shape)
        : store.load_tensor_as_shape(source, weight_name, storage, shape, TensorShape::from_dims({shape[0], shape[1]}));
    if (!names.bias.empty()) {
        auto bias = source.require_f32(expand_layer(names.bias, layer), {shape[0]});
        if (autocast) core::round_f32_to_bf16_in_place(bias);
        out.bias = store.make_f32(TensorShape::from_dims({shape[0]}), std::move(bias));
    }
    return out;
}

NormWeights load_norm(core::BackendWeightStore & store, const assets::TensorSource & source,
    const DitNormNames & names, int64_t layer) {
    NormWeights out;
    if (!names.weight.empty()) {
        const auto name = expand_layer(names.weight, layer);
        out.weight = store.load_f32_tensor(source, name, source.require_metadata(name).shape);
    }
    if (!names.bias.empty()) {
        const auto name = expand_layer(names.bias, layer);
        out.bias = store.load_f32_tensor(source, name, source.require_metadata(name).shape);
    }
    return out;
}

DitAttentionWeights load_attention(core::BackendWeightStore & store, const assets::TensorSource & source,
    const DitAttentionNames & names, assets::TensorStorageType storage, int64_t layer, bool autocast,
    core::BackendType backend, bool self) {
    DitAttentionWeights out;
    out.output = load_projection(store, source, names.output, storage, layer, autocast, backend);
    out.query_norm = load_norm(store, source, names.query_norm, layer);
    out.key_norm = load_norm(store, source, names.key_norm, layer);
    out.qkv = load_projection(store, source, names.qkv, storage, layer, autocast, backend);
    out.kv = load_projection(store, source, names.kv, storage, layer, autocast, backend);
    if (self && (autocast || !names.qkv_parts.empty()) && names.qkv.weight.empty()) {
        const auto parts = names.qkv_parts.empty() ? std::vector<DitLinearNames>{names.query, names.key, names.value} : names.qkv_parts;
        const auto q = expand_layer(parts.front().weight, layer);
        const auto shape = source.require_metadata(q).shape;
        std::vector<PackedLinearSource> projections;
        std::vector<float> bias;
        for (const auto & projection : parts) {
            const auto weight = expand_layer(projection.weight, layer);
            const auto width = source.require_metadata(weight).shape[0];
            projections.push_back({weight, std::nullopt, width});
            if (!projection.bias.empty()) {
                const auto values = source.require_f32(expand_layer(projection.bias, layer), {width});
                bias.insert(bias.end(), values.begin(), values.end());
            }
        }
        auto packed_storage = storage;
        if (backend == core::BackendType::Vulkan && storage == assets::TensorStorageType::Native)
            packed_storage = assets::resolve_tensor_storage_type(source, q, storage);
        if (!autocast && storage == assets::TensorStorageType::Native &&
            (backend == core::BackendType::Vulkan || backend == core::BackendType::Metal) &&
            assets::resolve_tensor_storage_type(source, q, storage) == assets::TensorStorageType::BF16)
            packed_storage = assets::TensorStorageType::F16;
        out.qkv = PackedLinearWeightsBuilder({shape[1], projections, false, true}).build(store, source, packed_storage);
        if (!bias.empty()) {
            if (autocast) core::round_f32_to_bf16_in_place(bias);
            const auto width = static_cast<int64_t>(bias.size());
            out.qkv.bias = store.make_f32(TensorShape::from_dims({width}), std::move(bias));
        }
    } else {
        out.query = load_projection(store, source, names.query, storage, layer, autocast, backend);
        out.key = load_projection(store, source, names.key, storage, layer, autocast, backend);
        out.value = load_projection(store, source, names.value, storage, layer, autocast, backend);
    }
    return out;
}

TensorValue project(core::ModuleBuildContext & ctx, TensorValue input, LinearWeights weights,
    const DitConfig & config, bool autocast) {
    LinearConfig linear{weights.weight.shape.dims[1], weights.weight.shape.dims[0], weights.bias.has_value(), config.projection_precision};
    if (autocast && config.projection_type == DitProjectionType::Bf16Autocast) {
        if (!ggml_is_quantized(weights.weight.type) && weights.weight.type != GGML_TYPE_BF16)
            weights.weight = core::wrap_tensor(ggml_cast(ctx.ggml, weights.weight.tensor, GGML_TYPE_BF16), weights.weight.shape, GGML_TYPE_BF16);
        if (ggml_is_quantized(weights.weight.type) || ctx.backend_type == core::BackendType::Metal) {
            input = Bf16RoundModule().build(ctx, input);
            linear.vulkan_f32_activation_lowering = ctx.backend_type == core::BackendType::Vulkan;
        } else if (ctx.backend_type != core::BackendType::Cuda && input.type != GGML_TYPE_BF16) {
            input = core::wrap_tensor(ggml_cast(ctx.ggml, input.tensor, GGML_TYPE_BF16), input.shape, GGML_TYPE_BF16);
        }
    }
    return LinearModule(linear).build(ctx, input, weights);
}

TensorValue mlp(core::ModuleBuildContext & ctx, TensorValue input,
    const std::vector<LinearWeights> & weights, const DitConfig & config, bool autocast, DitEmbeddingActivation activation = DitEmbeddingActivation::Silu) {
    for (size_t i = 0; i < weights.size(); ++i) {
        if (i) input = activation == DitEmbeddingActivation::Silu ? SiluModule().build(ctx, input) : GeluModule({config.gelu_approximation}).build(ctx, input);
        input = project(ctx, input, weights[i], config, autocast);
    }
    return input;
}

TensorValue build_condition(core::ModuleBuildContext & ctx, const DitForwardInputs & in,
    const DitWeights & weights, const DitConfig & config) {
    auto value = in.time_condition.valid() ? in.time_condition : mlp(ctx, in.time, weights.time_embedding, config, false);
    if (in.global.valid()) {
        auto global = mlp(ctx, in.global, weights.global_embedding, config, true);
        if (global.shape.dims[0] != value.shape.dims[0]) global = RepeatModule({value.shape}).build(ctx, global);
        value = AddModule().build(ctx, global, value);
    }
    if (in.duration_condition.valid() || in.duration.valid()) {
        auto duration = in.duration_condition.valid() ? in.duration_condition : mlp(ctx, in.duration, weights.duration_embedding, config, false);
        value = AddModule().build(ctx, value, duration);
    }
    return value;
}

TensorValue norm(core::ModuleBuildContext & ctx, const TensorValue & input,
    const NormWeights & weights, const DitConfig & config) {
    NormConfig c{config.hidden_size, config.norm_eps, weights.weight.has_value(), weights.bias.has_value()};
    return config.normalization == DitNormalization::Rms
        ? RMSNormModule(c).build(ctx, input, weights) : LayerNormModule(c).build(ctx, input, weights);
}

TensorValue heads(core::ModuleBuildContext & ctx, const TensorValue & input, int64_t count, int64_t width,
    bool preserve_views = false) {
    if (input.shape.rank == 4) return input;
    if (!preserve_views) return core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, input),
        TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], count, width}));
    auto * view = ggml_view_4d(ctx.ggml, input.tensor, width, count, input.shape.dims[1], input.shape.dims[0],
        width * input.tensor->nb[0], input.tensor->nb[1], input.tensor->nb[2], 0);
    return core::wrap_tensor(view, TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], count, width}), input.type);
}

TensorValue attention(core::ModuleBuildContext & ctx, const TensorValue & input, const TensorValue & context,
    const TensorValue & positions, const TensorValue & value_mask, const DitAttentionWeights & weights,
    const DitAttentionConfig & c, const DitConfig & config, const DitContextKV * cache = nullptr,
    const TensorValue & attention_mask = {}, const DitContextKV * prefix = nullptr,
    std::optional<ggml_type> cache_type = std::nullopt, DitContextKV * current = nullptr) {
    TensorValue q, k, v, q_diff, k_diff;
    // Preserve packed projection strides only for the view-preserving lowering.
    // The ordinary flash and explicit paths retain canonical projected heads.
    const bool preserve_views = c.lowering == ScaledDotProductAttentionLowering::FlashPreserveViews;
    const bool self = !context.valid() && cache == nullptr;
    if (self && !prefix && !current && !attention_mask.valid() && !c.differential && c.lowering != ScaledDotProductAttentionLowering::Explicit && weights.query.weight.valid() && c.qk_norm == DitQKNorm::None && c.rotary.dimensions == c.head_dim) {
        ProjectedGroupedSelfAttentionConfig a;
        a.hidden_size = config.hidden_size; a.attention_heads = c.heads; a.kv_heads = c.kv_heads;
        a.head_dim = c.head_dim; a.use_bias = weights.query.bias.has_value();
        a.projection_precision = config.projection_precision; a.attention_precision = c.precision;
        a.use_rope = c.rotary.dimensions > 0; a.apply_rope_to_projected_prefix = true;
        a.rope_type = c.rotary.mode; a.rope_theta = c.rotary.theta; a.local_rope_theta = c.rotary.theta;
        a.lowering = GroupedQueryAttentionLowering::FlashGroupedViewKV;
        if (!value_mask.valid() && config.projection_type == DitProjectionType::Native)
            return ProjectedGroupedSelfAttentionModule(a).build(ctx, input, positions,
                {weights.query, weights.key, weights.value, weights.output, {}, {}}, 0);
    }
    if (weights.qkv.weight.valid()) {
        const auto packed = project(ctx, input, weights.qkv, config, true);
        const auto width = c.heads * c.head_dim, kv_width = c.kv_heads * c.head_dim;
        q = SliceModule({2, 0, width}).build(ctx, packed);
        k = SliceModule({2, width, kv_width}).build(ctx, packed);
        v = SliceModule({2, width + kv_width, kv_width}).build(ctx, packed);
        if (c.differential) {
            q_diff = SliceModule({2, width + 2 * kv_width, width}).build(ctx, packed);
            k_diff = SliceModule({2, 2 * width + 2 * kv_width, kv_width}).build(ctx, packed);
        }
    } else {
        q = project(ctx, input, weights.query, config, true);
        if (c.differential) {
            q_diff = SliceModule({2, c.heads * c.head_dim, c.heads * c.head_dim}).build(ctx, q);
            q = SliceModule({2, 0, c.heads * c.head_dim}).build(ctx, q);
        }
        if (cache == nullptr) {
            const auto memory = self ? input : context;
            if (weights.kv.weight.valid()) {
                const auto packed = project(ctx, memory, weights.kv, config, true);
                const auto width = c.kv_heads * c.head_dim;
                k = SliceModule({2, 0, width}).build(ctx, packed);
                v = SliceModule({2, c.differential ? 2 * width : width, width}).build(ctx, packed);
                if (c.differential) k_diff = SliceModule({2, width, width}).build(ctx, packed);
            } else {
                k = project(ctx, memory, weights.key, config, true);
                v = project(ctx, memory, weights.value, config, true);
            }
        }
    }
    const auto normalize = [&](TensorValue x, const NormWeights & w, int64_t nheads) {
        if (c.qk_norm == DitQKNorm::None) return x;
        if (c.qk_norm == DitQKNorm::Heads) x = heads(ctx, x, nheads, c.head_dim, preserve_views);
        return RMSNormModule({x.shape.last_dim(), c.norm_eps, w.weight.has_value(), false,
            config.projection_type == DitProjectionType::Bf16Autocast}).build(ctx, x, w);
    };
    q = normalize(q, weights.query_norm, c.heads);
    q = heads(ctx, q, c.heads, c.head_dim, preserve_views);
    if (self && c.rotary.dimensions > 0) q = RoPEModule(c.rotary).build(ctx, q, positions);
    q = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    if (cache) { k = cache->key; v = cache->value; k_diff = cache->differential_key; }
    else {
        k = normalize(k, weights.key_norm, c.kv_heads);
        k = heads(ctx, k, c.kv_heads, c.head_dim, preserve_views);
        v = heads(ctx, v, c.kv_heads, c.head_dim, preserve_views);
        if (self && c.rotary.dimensions > 0) k = RoPEModule(c.rotary).build(ctx, k, positions);
        k = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
        v = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
    }
    if (self && value_mask.valid()) v = MulModule().build(ctx, v, RepeatModule({v.shape}).build(ctx, value_mask));
    if (self && cache_type) {
        for (auto * value : {&k, &v}) if (value->type != *cache_type)
            *value = core::wrap_tensor(ggml_cast(ctx.ggml, value->tensor, *cache_type), value->shape, *cache_type);
    }
    if (current) *current = {k, v, {}};
    if (prefix) {
        k = ConcatModule({2}).build(ctx, prefix->key, k);
        v = ConcatModule({2}).build(ctx, prefix->value, v);
    }
    const auto repeat_kv_heads = [&](TensorValue x) {
        if (c.kv_heads == c.heads) return x;
        const auto b = x.shape.dims[0], n = x.shape.dims[2];
        x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), TensorShape::from_dims({b, c.kv_heads, 1, n * c.head_dim}));
        x = RepeatModule({TensorShape::from_dims({b, c.kv_heads, c.heads / c.kv_heads, n * c.head_dim})}).build(ctx, x);
        return core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), TensorShape::from_dims({b, c.heads, n, c.head_dim}));
    };
    k = repeat_kv_heads(k);
    v = repeat_kv_heads(v);
    auto result = ScaledDotProductAttentionModule({c.head_dim, c.lowering, c.precision, AttentionCausality::NonCausal}).build(ctx, q, k, v,
        attention_mask.valid() ? std::optional<TensorValue>(attention_mask) : std::nullopt);
    if (c.differential) {
        q_diff = heads(ctx, normalize(q_diff, weights.query_norm, c.heads), c.heads, c.head_dim, preserve_views);
        if (self && c.rotary.dimensions > 0) q_diff = RoPEModule(c.rotary).build(ctx, q_diff, positions);
        q_diff = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q_diff);
        if (!cache) {
            k_diff = heads(ctx, normalize(k_diff, weights.key_norm, c.kv_heads), c.kv_heads, c.head_dim, preserve_views);
            if (self && c.rotary.dimensions > 0) k_diff = RoPEModule(c.rotary).build(ctx, k_diff, positions);
            k_diff = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k_diff);
        }
        if (self && cache_type && k_diff.type != *cache_type)
            k_diff = core::wrap_tensor(ggml_cast(ctx.ggml, k_diff.tensor, *cache_type), k_diff.shape, *cache_type);
        if (current) current->differential_key = k_diff;
        if (prefix) k_diff = ConcatModule({2}).build(ctx, prefix->differential_key, k_diff);
        k_diff = repeat_kv_heads(k_diff);
        const auto secondary = ScaledDotProductAttentionModule({c.head_dim, c.lowering, c.precision, AttentionCausality::NonCausal}).build(ctx, q_diff, k_diff, v,
            attention_mask.valid() ? std::optional<TensorValue>(attention_mask) : std::nullopt);
        result = core::wrap_tensor(ggml_sub(ctx.ggml, result.tensor, secondary.tensor), result.shape);
    }
    result = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, result),
        TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], c.heads * c.head_dim}));
    return project(ctx, result, weights.output, config, true);
}

TensorValue position_convolution(core::ModuleBuildContext & ctx, TensorValue input,
    const std::vector<Conv1dWeights> & weights, const DitConfig & config) {
    auto x = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
    const auto channels = config.hidden_size, groups = config.position_groups;
    for (const auto & w : weights) {
        x = core::wrap_tensor(ggml_pad_ext(ctx.ggml, x.tensor, config.position_kernel - 1, 0, 0, 0, 0, 0, 0, 0),
            TensorShape::from_dims({x.shape.dims[0], channels, x.shape.dims[2] + config.position_kernel - 1}));
        TensorValue output;
        const auto contiguous = core::ensure_backend_addressable_layout(ctx, x);
        for (int64_t group = 0; group < groups; ++group) {
            const auto in = SliceModule({1, group * channels / groups, channels / groups}).build(ctx, contiguous);
            Conv1dWeights part{SliceModule({0, group * channels / groups, channels / groups}).build(ctx, w.weight), std::nullopt};
            if (w.bias) part.bias = SliceModule({0, group * channels / groups, channels / groups}).build(ctx, *w.bias);
            const auto y = Conv1dModule({channels / groups, channels / groups, config.position_kernel, 1, 0, 1, part.bias.has_value()}).build(ctx, in, part);
            output = output.valid() ? ConcatModule({1}).build(ctx, output, y) : y;
        }
        x = core::ensure_backend_addressable_layout(ctx, output);
        const auto softplus = core::wrap_tensor(ggml_softplus(ctx.ggml, x.tensor), x.shape);
        const auto tanh = core::wrap_tensor(ggml_tanh(ctx.ggml, softplus.tensor), x.shape);
        x = MulModule().build(ctx, x, tanh);
    }
    return AddModule().build(ctx, input, core::ensure_backend_addressable_layout(ctx, TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x)));
}
} // namespace

DitWeights load_dit_weights(core::BackendWeightStore & store, const assets::TensorSource & source,
    const DitConfig & config, const DitWeightNames & names, assets::TensorStorageType storage, core::BackendType backend) {
    const bool autocast = config.projection_type == DitProjectionType::Bf16Autocast;
    const auto scalar_storage = autocast ? assets::TensorStorageType::F32 : storage;
    DitWeights out;
    out.input = load_projection(store, source, names.input, storage, 0, autocast, backend);
    out.output = load_projection(store, source, names.output, storage, 0, autocast, backend);
    out.preprocess = load_projection(store, source, names.preprocess, storage, 0, autocast, backend);
    out.postprocess = load_projection(store, source, names.postprocess, storage, 0, autocast, backend);
    for (const auto & name : names.duration_embedding) out.duration_embedding.push_back(load_projection(store, source, name, scalar_storage, 0, false, backend));
    for (const auto & name : names.time_embedding) out.time_embedding.push_back(load_projection(store, source, name, scalar_storage, 0, false, backend));
    for (const auto & name : names.context_embedding) out.context_embedding.push_back(load_projection(store, source, name, storage, 0, autocast, backend));
    for (const auto & name : names.global_embedding) out.global_embedding.push_back(load_projection(store, source, name,storage, 0, autocast, backend));
    for (const auto & name : names.global_modulation) out.global_modulation.push_back(load_projection(store, source, name, storage, 0, autocast, backend));
    if (!names.memory_tokens.empty()) out.memory_tokens = store.load_f32_tensor(source, names.memory_tokens, {config.memory_tokens, config.hidden_size});
    out.time_modulation = load_projection(store, source, names.time_modulation, scalar_storage, 0, false, backend);
    out.final_modulation = load_projection(store, source, names.final_modulation, scalar_storage, 0, false, backend);
    if (!names.final_modulation_table.empty()) out.final_modulation_table = store.load_f32_tensor(source, names.final_modulation_table, {1, 2, config.hidden_size});
    for (const auto & name : names.position_convolutions) {
        out.position_convolutions.push_back({store.load_tensor(source, name.weight, storage,
            {config.hidden_size, config.hidden_size / config.position_groups, config.position_kernel}),
            name.bias.empty() ? std::nullopt : std::optional<TensorValue>(store.load_f32_tensor(source, name.bias, {config.hidden_size}))});
    }
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        DitLayerWeights w;
        w.self_norm = load_norm(store, source, names.layer.self_norm, layer);
        auto cross_names = names.layer.cross_attention;
        const bool fold_query = autocast && storage == assets::TensorStorageType::Native &&
            !cross_names.query.weight.empty() && !cross_names.query.bias.empty() &&
            !names.layer.cross_norm.weight.empty() && !names.layer.cross_norm.bias.empty() &&
            assets::resolve_tensor_storage_type(source, expand_layer(cross_names.query.weight, layer), storage) == assets::TensorStorageType::BF16;
        if (fold_query) cross_names.query = {};
        else w.cross_norm = load_norm(store, source, names.layer.cross_norm, layer);
        w.feed_forward_norm = load_norm(store, source, names.layer.feed_forward_norm, layer);
        w.self_attention = load_attention(store, source, names.layer.self_attention, storage, layer, autocast, backend, true);
        w.cross_attention = load_attention(store, source, cross_names, storage, layer, autocast, backend, false);
        // Normalize a fixed affine norm into its sole autocast query projection.
        // The operation is load-time and preserves the established BF16 binding.
        if (fold_query) {
            const auto q_name = expand_layer(names.layer.cross_attention.query.weight, layer);
            const auto bias_name = expand_layer(names.layer.cross_attention.query.bias, layer);
            auto matrix = source.require_f32(q_name, {config.hidden_size, config.hidden_size});
            auto bias = source.require_f32(bias_name, {config.hidden_size});
            const auto scale = source.require_f32(expand_layer(names.layer.cross_norm.weight, layer), {config.hidden_size});
            const auto shift = source.require_f32(expand_layer(names.layer.cross_norm.bias, layer), {config.hidden_size});
            core::round_f32_to_bf16_in_place(matrix); core::round_f32_to_bf16_in_place(bias);
            for (int64_t row = 0; row < config.hidden_size; ++row) {
                float offset = bias[static_cast<size_t>(row)];
                for (int64_t col = 0; col < config.hidden_size; ++col) {
                    auto & value = matrix[static_cast<size_t>(row * config.hidden_size + col)];
                    offset += value * shift[static_cast<size_t>(col)]; value *= scale[static_cast<size_t>(col)];
                }
                bias[static_cast<size_t>(row)] = offset;
            }
            std::vector<ggml_bf16_t> folded(matrix.size());
            ggml_fp32_to_bf16_row(matrix.data(), folded.data(), static_cast<int64_t>(matrix.size()));
            w.cross_attention.query = {store.make_tensor(TensorShape::from_dims({config.hidden_size, config.hidden_size}), GGML_TYPE_BF16,
                folded.data(), folded.size() * sizeof(ggml_bf16_t)), store.make_f32(TensorShape::from_dims({config.hidden_size}), std::move(bias))};
            w.cross_norm = {};
        }

        if (config.feed_forward == DitFeedForward::Gelu) {
            FeedForwardConfig ff{config.hidden_size, config.intermediate_size,
                !names.layer.feed_forward_in.bias.empty(), config.gelu_approximation, config.projection_precision};
            ff.precision_policy = autocast ? FeedForwardPrecisionPolicy::Bf16Autocast : FeedForwardPrecisionPolicy::Native;
            const auto parameters = binding::feed_forward_from_source(store, source,
                {expand_layer(names.layer.feed_forward_in.weight, layer), expand_layer(names.layer.feed_forward_in.bias, layer),
                 expand_layer(names.layer.feed_forward_out.weight, layer), expand_layer(names.layer.feed_forward_out.bias, layer)},
                ff, storage, backend);
            w.feed_forward_in = {parameters.fc1_weight, parameters.fc1_bias};
            w.feed_forward_out = {parameters.fc2_weight, parameters.fc2_bias};
        } else {
            w.feed_forward_in = load_projection(store, source, names.layer.feed_forward_in, storage, layer, autocast, backend);
            w.feed_forward_out = load_projection(store, source, names.layer.feed_forward_out, storage, layer, autocast, backend);
        }
        for (const auto & name : names.layer.local_embedding) w.local_embedding.push_back(load_projection(store, source, name, storage, layer, autocast, backend));
        w.modulation = load_projection(store, source, names.layer.modulation, scalar_storage, layer, false, backend);
        if (!names.layer.modulation_table.empty()) {
            const auto name = expand_layer(names.layer.modulation_table, layer);
            w.modulation_table = store.make_f32(TensorShape::from_dims({1, 6, config.hidden_size}),
                source.require_f32(name, source.require_metadata(name).shape));
        }
        out.layers.push_back(std::move(w));
    }
    out.final_shift_plane = names.final_shift_plane;
    out.final_scale_plane = names.final_scale_plane;
    out.modulation_planes = names.modulation_planes;
    return out;
}

DitModule::DitModule(DitConfig config) : config_(std::move(config)) {
    if (config_.hidden_size <= 0 || config_.input_channels <= 0 || config_.output_channels <= 0 ||
        config_.layers <= 0 || config_.intermediate_size <= 0 || config_.norm_eps <= 0)
        throw std::invalid_argument("DiT dimensions and normalization epsilon must be positive");
    for (const auto * a : {&config_.self_attention, &config_.cross_attention}) {
        if (a == &config_.cross_attention && !config_.context_dim) continue;
        if (a->heads <= 0 || a->kv_heads <= 0 || a->head_dim <= 0 || a->heads % a->kv_heads || a->heads * a->head_dim != config_.hidden_size)
            throw std::invalid_argument("DiT attention dimensions are inconsistent");
    }
    if (config_.position_kernel && (config_.position_groups <= 0 || config_.hidden_size % config_.position_groups))
        throw std::invalid_argument("DiT positional convolution groups must divide hidden size");
}

std::vector<DitContextKV> DitModule::build_context(core::ModuleBuildContext & ctx,
    const TensorValue & context, const DitWeights & weights) const {
    const auto embedded = mlp(ctx, context, weights.context_embedding, config_, true, config_.context_activation);
    std::vector<DitContextKV> out;
    for (const auto & layer : weights.layers) {
        const auto & w = layer.cross_attention;
        TensorValue key, value, differential_key;
        const auto & c = config_.cross_attention;
        if (w.kv.weight.valid()) {
            auto packed = project(ctx, embedded, w.kv, config_, true);
            const auto width = c.kv_heads * c.head_dim;
            key = SliceModule({2, 0, width}).build(ctx, packed);
            value = SliceModule({2, c.differential ? 2 * width : width, width}).build(ctx, packed);
            if (c.differential) differential_key = SliceModule({2, width, width}).build(ctx, packed);
        } else {
            key = project(ctx, embedded, w.key, config_, true); value = project(ctx, embedded, w.value, config_, true);
        }
        if (c.qk_norm != DitQKNorm::None) {
            if (c.qk_norm == DitQKNorm::Heads) key = heads(ctx, key, c.kv_heads, c.head_dim);
            key = RMSNormModule({key.shape.last_dim(), c.norm_eps, w.key_norm.weight.has_value(), false}).build(ctx, key, w.key_norm);
        }
        key = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, heads(ctx, key, c.kv_heads, c.head_dim));
        value = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, heads(ctx, value, c.kv_heads, c.head_dim));
        if (c.differential) {
            if (c.qk_norm == DitQKNorm::Heads) differential_key = heads(ctx, differential_key, c.kv_heads, c.head_dim);
            if (c.qk_norm != DitQKNorm::None) differential_key = RMSNormModule({differential_key.shape.last_dim(), c.norm_eps, w.key_norm.weight.has_value(), false}).build(ctx, differential_key, w.key_norm);
            differential_key = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, heads(ctx, differential_key, c.kv_heads, c.head_dim));
        }
        out.push_back({key, value, differential_key});
    }
    return out;
}

TensorValue DitModule::build(core::ModuleBuildContext & ctx, const DitForwardInputs & in, const DitWeights & weights, ggml_cgraph * graph) const {
    const auto & c = config_;
    if (weights.layers.size() != static_cast<size_t>(c.layers)) throw std::invalid_argument("DiT layer count mismatch");
    auto input = in.latent;
    for (const auto & condition : in.frame_conditions) input = ConcatModule({2}).build(ctx, input, condition);
    for (const auto & condition : in.vector_conditions) {
        auto value = core::reshape_tensor(ctx, condition, TensorShape::from_dims({input.shape.dims[0], 1, condition.shape.last_dim()}));
        value = RepeatModule({TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], value.shape.last_dim()})}).build(ctx, value);
        input = ConcatModule({2}).build(ctx, input, value);
    }
    input = core::ensure_backend_addressable_layout(ctx, input);
    if (weights.preprocess.weight.valid()) input = AddModule().build(ctx, project(ctx, input, weights.preprocess, c, true), input);
    auto x = project(ctx, input, weights.input, c, true);
    if (!weights.position_convolutions.empty()) x = position_convolution(ctx, core::ensure_backend_addressable_layout(ctx, x), weights.position_convolutions, c);
    if (weights.memory_tokens.valid()) {
        auto prefix = core::reshape_tensor(ctx, weights.memory_tokens, TensorShape::from_dims({1, c.memory_tokens, c.hidden_size}));
        prefix = RepeatModule({TensorShape::from_dims({x.shape.dims[0], c.memory_tokens, c.hidden_size})}).build(ctx, prefix);
        x = ConcatModule({1}).build(ctx, prefix, x);
    }
    TensorValue time;
    if (!in.modulations.valid()) {
        time = build_condition(ctx, in, weights, c);
        time = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, time), TensorShape::from_dims({time.shape.dims[0], 1, c.hidden_size}));
    }
    TensorValue shared;
    if (!weights.global_modulation.empty()) {
        shared = mlp(ctx, time, weights.global_modulation, c, false);
        shared = core::reshape_tensor(ctx, shared, TensorShape::from_dims({shared.shape.dims[0], 6, c.hidden_size}));
    }
    if (weights.time_modulation.weight.valid()) {
        shared = project(ctx, SiluModule().build(ctx, time), weights.time_modulation, c, false);
        shared = core::reshape_tensor(ctx, shared, TensorShape::from_dims({shared.shape.dims[0], 6, c.hidden_size}));
    }
    const auto context = in.context.valid() ? mlp(ctx, in.context, weights.context_embedding, c, true, c.context_activation) : TensorValue{};
    for (size_t i = 0; i < weights.layers.size(); ++i) {
        const auto & w = weights.layers[i];
        TensorValue modulation;
        if (in.modulations.valid()) {
            modulation = SliceModule({1, static_cast<int64_t>(i) * 6 * c.hidden_size, 6 * c.hidden_size}).build(ctx, in.modulations);
            modulation = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, modulation), TensorShape::from_dims({modulation.shape.dims[0], 6, c.hidden_size}));
        } else if (c.modulation == DitModulation::PerLayer) {
            modulation = project(ctx, SiluModule().build(ctx, time), w.modulation, c, false);
            modulation = core::reshape_tensor(ctx, modulation, TensorShape::from_dims({modulation.shape.dims[0], 6, c.hidden_size}));
        } else if (c.modulation == DitModulation::SharedTable) {
            auto table = w.modulation_table;
            if (table.shape.dims[0] != shared.shape.dims[0]) table = RepeatModule({shared.shape}).build(ctx, table);
            modulation = AddModule().build(ctx, table, shared);
        }
        const auto plane = [&](int index) {
            auto value = SliceModule({1, weights.modulation_planes.at(static_cast<size_t>(index)), 1}).build(ctx, modulation);
            return core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, value), value.shape);
        };
        const auto normalized = [&](TensorValue value, const NormWeights & n, int shift, int scale) {
            value = norm(ctx, value, n, c);
            if (modulation.valid()) value = AdaptiveLayerNormModule({c.hidden_size, c.norm_eps, AdaptiveLayerNormMode::Affine}).build(ctx, value, {plane(shift), plane(scale)});
            return value;
        };
        DitContextKV current;
        const auto * cache = in.self_cache ? &*in.self_cache : nullptr;
        const auto * prefix = cache && !cache->prefix.empty() ? &cache->prefix.at(i) : nullptr;
        auto y = attention(ctx, normalized(x, w.self_norm, 0, 1), {}, in.positions, in.value_mask,
            w.self_attention, c.self_attention, c, nullptr, in.attention_mask, prefix,
            cache ? cache->storage_type : std::nullopt, cache ? &current : nullptr);
        const auto gate = [&](int index) {
            auto value = plane(index);
            if (c.residual_gate == DitResidualGate::SigmoidOneMinus)
                value = SigmoidModule().build(ctx, ScaleBiasModule({-1, 1}).build(ctx, value));
            return value;
        };
        if (modulation.valid()) y = MulModule(true).build(ctx, y, gate(2));
        x = AddModule().build(ctx, x, y);
        if (c.context_dim) {
            auto query = norm(ctx, x, w.cross_norm, c);
            y = attention(ctx, query, context, {}, {}, w.cross_attention, c.cross_attention, c,
                in.context_kv.empty() ? nullptr : &in.context_kv.at(i));
            x = AddModule().build(ctx, x, y);
        }
        if (!w.local_embedding.empty()) {
            auto local = mlp(ctx, in.local_condition, w.local_embedding, c, false);
            if (c.memory_tokens) {
                const auto prefix = SliceModule({1, 0, c.memory_tokens}).build(ctx, x);
                auto zero = core::wrap_tensor(ggml_sub(ctx.ggml, prefix.tensor, prefix.tensor), prefix.shape);
                local = ConcatModule({1}).build(ctx, zero, local);
            }
            x = AddModule().build(ctx, x, local);
        }
        y = normalized(x, w.feed_forward_norm, 3, 4);
        const auto b = y.shape.dims[0], t = y.shape.dims[1];
        y = core::reshape_tensor(ctx, y, TensorShape::from_dims({b * t, c.hidden_size}));
        if (c.feed_forward == DitFeedForward::SwiGlu) {
            GatedFeedForwardConfig ff{c.hidden_size, c.intermediate_size,
                w.feed_forward_in.bias.has_value(), GatedFeedForwardActivation::Silu,
                c.gelu_approximation, c.projection_precision};
            y = GatedFeedForwardModule(ff).build(ctx, y,
                {{}, {}, w.feed_forward_out, w.feed_forward_in});
        } else {
            FeedForwardConfig ff{c.hidden_size, c.intermediate_size,
                w.feed_forward_in.bias.has_value(), c.gelu_approximation, c.projection_precision};
            ff.precision_policy = c.projection_type == DitProjectionType::Bf16Autocast
                ? FeedForwardPrecisionPolicy::Bf16Autocast : FeedForwardPrecisionPolicy::Native;
            y = FeedForwardModule(ff).build(ctx, y, {w.feed_forward_in.weight, w.feed_forward_in.bias,
                w.feed_forward_out.weight, w.feed_forward_out.bias});
        }
        y = core::reshape_tensor(ctx, y, TensorShape::from_dims({b, t, c.hidden_size}));
        if (modulation.valid()) y = MulModule(true).build(ctx, y, gate(5));
        x = AddModule().build(ctx, x, y);
        if (cache) {
            if (!graph || cache->append_tokens <= 0) throw std::invalid_argument("DiT cache append requires an allocation graph and positive token count");
            const auto & destination = cache->destination.at(i);
            for (const auto & pair : {std::pair<TensorValue, TensorValue>{current.key, destination.key}, {current.value, destination.value}, {current.differential_key, destination.differential_key}}) {
                if (!pair.first.valid()) continue;
                auto rows = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, pair.first);
                rows = SliceModule({1, 0, cache->append_tokens}).build(ctx, rows);
                rows = core::ensure_backend_addressable_layout(ctx, rows);
                if (rows.type != GGML_TYPE_F32) rows = core::wrap_tensor(ggml_cast(ctx.ggml, rows.tensor, GGML_TYPE_F32), rows.shape);
                rows = core::reshape_tensor(ctx, rows, TensorShape::from_dims({1, rows.shape.dims[0] * rows.shape.dims[1], rows.shape.dims[2], rows.shape.dims[3]}));
                const auto updated = FastKVSetRowsModule({FastKVSetRowsMode::BackendViewOptimized}).build_block(ctx, pair.second, rows, cache->slots);
                ggml_build_forward_expand(graph, updated.tensor);
            }
        }
    }
    if (c.memory_tokens) x = SliceModule({1, c.memory_tokens, in.latent.shape.dims[1]}).build(ctx, x);
    if (in.output_tokens > 0) x = SliceModule({1, in.output_start, in.output_tokens}).build(ctx, x);
    if (weights.final_modulation.weight.valid() || weights.final_modulation_table.valid()) {
        auto final = in.modulations.valid()
            ? SliceModule({1, c.layers * 6 * c.hidden_size, 2 * c.hidden_size}).build(ctx, in.modulations)
            : weights.final_modulation.weight.valid() ? project(ctx, SiluModule().build(ctx, time), weights.final_modulation, c, false) : time;
        if (weights.final_modulation_table.valid()) {
            final = core::reshape_tensor(ctx, final, TensorShape::from_dims({final.shape.dims[0], 1, c.hidden_size}));
            const auto table = RepeatModule({TensorShape::from_dims({final.shape.dims[0], 2, c.hidden_size})}).build(ctx, weights.final_modulation_table);
            final = AddModule(false, true).build(ctx, table, final);
        } else final = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, final), TensorShape::from_dims({final.shape.dims[0], 2, c.hidden_size}));
        auto scale = SliceModule({1, weights.final_scale_plane, 1}).build(ctx, final);
        // A singleton-batch slice can be logically contiguous while retaining
        // the parent plane stride, which SCALE cannot consume.
        scale = core::wrap_tensor(ggml_cont(ctx.ggml, scale.tensor), scale.shape);
        x = AdaptiveLayerNormModule({c.hidden_size, c.norm_eps, AdaptiveLayerNormMode::LayerAffine}).build(ctx, x,
            {SliceModule({1, weights.final_shift_plane, 1}).build(ctx, final), scale});
    }
    x = project(ctx, x, weights.output, c, true);
    if (weights.postprocess.weight.valid()) x = AddModule().build(ctx, x, project(ctx, x, weights.postprocess, c, true));
    return x;
}
TensorValue DitModule::build_modulations(core::ModuleBuildContext & ctx,
    const DitForwardInputs & in, const DitWeights & weights) const {
    const auto & c = config_;
    if (c.modulation != DitModulation::PerLayer) throw std::invalid_argument("DiT modulation banks require per-layer conditioning");
    const auto condition = build_condition(ctx, in, weights, c);
    const auto activated = SiluModule().build(ctx, condition);
    TensorValue output;
    for (const auto & layer : weights.layers) {
        const auto row = project(ctx, activated, layer.modulation, c, false);
        output = output.valid() ? ConcatModule({1}).build(ctx, output, row) : row;
    }
    return ConcatModule({1}).build(ctx, output, project(ctx, activated, weights.final_modulation, c, false));
}
} // namespace engine::modules
