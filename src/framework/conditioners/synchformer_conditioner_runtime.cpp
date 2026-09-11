#include "engine/framework/conditioners/synchformer_conditioner_runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/attention/grouped_query_attention.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/attention/self_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::conditioners {
namespace {

namespace binding = engine::modules::binding;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

std::string tensor_name(const SynchformerConfig & config, const std::string & name) {
    if (config.tensor_prefix.empty()) {
        return name;
    }
    return config.tensor_prefix + "." + name;
}

void validate_config(const SynchformerConfig & config) {
    if (config.image_channels <= 0 || config.image_size <= 0 || config.frames <= 0 ||
        config.patch_size <= 0 || config.temporal_patch_size <= 0 || config.hidden_size <= 0 ||
        config.layers <= 0 || config.heads <= 0 || config.intermediate_size <= 0) {
        throw std::runtime_error("Synchformer config dimensions must be valid");
    }
    if (config.hidden_size % config.heads != 0) {
        throw std::runtime_error("Synchformer hidden size must be divisible by head count");
    }
    if (config.image_size % config.patch_size != 0 || config.frames % config.temporal_patch_size != 0) {
        throw std::runtime_error("Synchformer patch sizes must divide image/frame dimensions");
    }
}

struct SynchformerBlockWeights {
    modules::NormWeights norm1;
    modules::NormWeights norm2;
    modules::NormWeights norm3;
    modules::AttentionWeights space_attention;
    modules::AttentionWeights time_attention;
    modules::FeedForwardWeights mlp;
};

struct SynchformerSpatialAggregatorWeights {
    core::TensorValue cls_token;
    modules::NormWeights norm1;
    modules::NormWeights norm2;
    modules::AttentionWeights attention;
    modules::FeedForwardWeights mlp;
};

struct SynchformerWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::Conv3dWeights patch_embedding;
    core::TensorValue cls_token;
    core::TensorValue position;
    std::vector<SynchformerBlockWeights> blocks;
    modules::NormWeights norm;
    SynchformerSpatialAggregatorWeights spatial_aggregator;
};

modules::AttentionWeights load_qkv_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const SynchformerConfig & config,
    const std::string & prefix,
    const SynchformerRuntimeOptions & options) {
    modules::AttentionWeights weights;
    weights.qkv_weight = store.load_tensor(
        source,
        tensor_name(config, prefix + ".qkv.weight"),
        options.weight_storage_type,
        {3 * config.hidden_size, config.hidden_size});
    weights.qkv_bias = store.load_f32_tensor(
        source,
        tensor_name(config, prefix + ".qkv.bias"),
        {3 * config.hidden_size});
    weights.out_weight = store.load_tensor(
        source,
        tensor_name(config, prefix + ".proj.weight"),
        options.weight_storage_type,
        {config.hidden_size, config.hidden_size});
    weights.out_bias = store.load_f32_tensor(
        source,
        tensor_name(config, prefix + ".proj.bias"),
        {config.hidden_size});
    return weights;
}

modules::AttentionWeights load_in_proj_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const SynchformerConfig & config,
    const std::string & prefix,
    const SynchformerRuntimeOptions & options) {
    modules::AttentionWeights weights;
    weights.qkv_weight = store.load_tensor(
        source,
        tensor_name(config, prefix + ".in_proj_weight"),
        options.weight_storage_type,
        {3 * config.hidden_size, config.hidden_size});
    weights.qkv_bias = store.load_f32_tensor(
        source,
        tensor_name(config, prefix + ".in_proj_bias"),
        {3 * config.hidden_size});
    weights.out_weight = store.load_tensor(
        source,
        tensor_name(config, prefix + ".out_proj.weight"),
        options.weight_storage_type,
        {config.hidden_size, config.hidden_size});
    weights.out_bias = store.load_f32_tensor(
        source,
        tensor_name(config, prefix + ".out_proj.bias"),
        {config.hidden_size});
    return weights;
}

SynchformerBlockWeights load_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const SynchformerConfig & config,
    int64_t layer,
    const SynchformerRuntimeOptions & options) {
    const std::string prefix = "blocks." + std::to_string(layer);
    return {
        binding::norm_from_source(store, source, tensor_name(config, prefix + ".norm1"), config.hidden_size),
        binding::norm_from_source(store, source, tensor_name(config, prefix + ".norm2"), config.hidden_size),
        binding::norm_from_source(store, source, tensor_name(config, prefix + ".norm3"), config.hidden_size),
        load_qkv_attention(store, source, config, prefix + ".attn", options),
        load_qkv_attention(store, source, config, prefix + ".timeattn", options),
        {
            store.load_tensor(
                source,
                tensor_name(config, prefix + ".mlp.fc1.weight"),
                options.weight_storage_type,
                {config.intermediate_size, config.hidden_size}),
            store.load_f32_tensor(source, tensor_name(config, prefix + ".mlp.fc1.bias"), {config.intermediate_size}),
            store.load_tensor(
                source,
                tensor_name(config, prefix + ".mlp.fc2.weight"),
                options.weight_storage_type,
                {config.hidden_size, config.intermediate_size}),
            store.load_f32_tensor(source, tensor_name(config, prefix + ".mlp.fc2.bias"), {config.hidden_size}),
        },
    };
}

std::vector<float> total_position_embedding(const assets::TensorSource & source, const SynchformerConfig & config) {
    const int64_t grid = config.image_size / config.patch_size;
    const int64_t spatial_tokens = grid * grid;
    const int64_t temporal_tokens = config.frames / config.temporal_patch_size;
    const auto pos = source.require_f32(
        tensor_name(config, "pos_embed"),
        {1, spatial_tokens + 1, config.hidden_size});
    const auto temp = source.require_f32(
        tensor_name(config, "temp_embed"),
        {1, temporal_tokens, config.hidden_size});
    std::vector<float> out(static_cast<size_t>((1 + temporal_tokens * spatial_tokens) * config.hidden_size));
    for (int64_t h = 0; h < config.hidden_size; ++h) {
        out[static_cast<size_t>(h)] = pos[static_cast<size_t>(h)];
    }
    for (int64_t t = 0; t < temporal_tokens; ++t) {
        for (int64_t n = 0; n < spatial_tokens; ++n) {
            for (int64_t h = 0; h < config.hidden_size; ++h) {
                const int64_t dst = (1 + t * spatial_tokens + n) * config.hidden_size + h;
                const int64_t pos_index = (1 + n) * config.hidden_size + h;
                const int64_t temp_index = t * config.hidden_size + h;
                out[static_cast<size_t>(dst)] =
                    pos[static_cast<size_t>(pos_index)] + temp[static_cast<size_t>(temp_index)];
            }
        }
    }
    return out;
}

SynchformerWeights load_weights(
    const assets::TensorSource & source,
    const SynchformerConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    const SynchformerRuntimeOptions & options) {
    SynchformerWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "framework.synchformer.weights",
        options.weight_context_bytes);
    weights.patch_embedding = {
        weights.store->load_tensor_as_shape(
            source,
            tensor_name(config, "patch_embed_3d.proj.weight"),
            options.weight_storage_type,
            {
                config.hidden_size * config.image_channels,
                config.temporal_patch_size,
                config.patch_size,
                config.patch_size,
            },
            core::TensorShape::from_dims({
                config.hidden_size * config.image_channels,
                config.temporal_patch_size,
                config.patch_size,
                config.patch_size,
            })),
        weights.store->load_f32_tensor(
            source,
            tensor_name(config, "patch_embed_3d.proj.bias"),
            {config.hidden_size})};
    weights.cls_token = weights.store->load_f32_tensor(
        source,
        tensor_name(config, "cls_token"),
        {1, 1, config.hidden_size});
    weights.position = weights.store->make_f32(
        core::TensorShape::from_dims({
            1,
            1 + (config.frames / config.temporal_patch_size) * (config.image_size / config.patch_size) * (config.image_size / config.patch_size),
            config.hidden_size,
        }),
        total_position_embedding(source, config));
    weights.blocks.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        weights.blocks.push_back(load_block(*weights.store, source, config, layer, options));
    }
    weights.norm = binding::norm_from_source(
        *weights.store,
        source,
        tensor_name(config, "norm"),
        config.hidden_size);
    weights.spatial_aggregator.cls_token = weights.store->load_f32_tensor(
        source,
        tensor_name(config, "spatial_attn_agg.cls_token"),
        {1, 1, config.hidden_size});
    weights.spatial_aggregator.norm1 = binding::norm_from_source(
        *weights.store,
        source,
        tensor_name(config, "spatial_attn_agg.norm1"),
        config.hidden_size);
    weights.spatial_aggregator.norm2 = binding::norm_from_source(
        *weights.store,
        source,
        tensor_name(config, "spatial_attn_agg.norm2"),
        config.hidden_size);
    weights.spatial_aggregator.attention = load_in_proj_attention(
        *weights.store,
        source,
        config,
        "spatial_attn_agg.self_attn",
        options);
    weights.spatial_aggregator.mlp = {
        weights.store->load_tensor(
            source,
            tensor_name(config, "spatial_attn_agg.linear1.weight"),
            options.weight_storage_type,
            {config.intermediate_size, config.hidden_size}),
        weights.store->load_f32_tensor(
            source,
            tensor_name(config, "spatial_attn_agg.linear1.bias"),
            {config.intermediate_size}),
        weights.store->load_tensor(
            source,
            tensor_name(config, "spatial_attn_agg.linear2.weight"),
            options.weight_storage_type,
            {config.hidden_size, config.intermediate_size}),
        weights.store->load_f32_tensor(
            source,
            tensor_name(config, "spatial_attn_agg.linear2.bias"),
            {config.hidden_size}),
    };
    weights.store->upload();
    return weights;
}

struct HeadTriplet {
    core::TensorValue q;
    core::TensorValue k;
    core::TensorValue v;
};

HeadTriplet project_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::AttentionWeights & weights,
    const SynchformerConfig & config,
    const SynchformerRuntimeOptions & options) {
    auto qkv = modules::LinearModule({
        config.hidden_size,
        3 * config.hidden_size,
        true,
        options.projection_precision,
    }).build(ctx, input, {*weights.qkv_weight, weights.qkv_bias});
    auto q = modules::SliceModule({2, 0, config.hidden_size}).build(ctx, qkv);
    auto k = modules::SliceModule({2, config.hidden_size, config.hidden_size}).build(ctx, qkv);
    auto v = modules::SliceModule({2, 2 * config.hidden_size, config.hidden_size}).build(ctx, qkv);
    const int64_t head_dim = config.hidden_size / config.heads;
    auto to_heads = [&](core::TensorValue value) {
        value = core::ensure_backend_addressable_layout(ctx, value);
        value = core::reshape_tensor(ctx, value, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.heads, head_dim}));
        value = modules::TransposeModule({{0, 2, 1, 3}, value.shape.rank}).build(ctx, value);
        value = core::ensure_backend_addressable_layout(ctx, value);
        return core::reshape_tensor(ctx, value, core::TensorShape::from_dims({input.shape.dims[0] * config.heads, input.shape.dims[1], head_dim}));
    };
    return {to_heads(q), to_heads(k), to_heads(v)};
}

core::TensorValue merge_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & heads,
    int64_t batch,
    int64_t tokens,
    const modules::AttentionWeights & weights,
    const SynchformerConfig & config,
    const SynchformerRuntimeOptions & options) {
    const int64_t head_dim = config.hidden_size / config.heads;
    auto merged = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, heads), core::TensorShape::from_dims({batch, config.heads, tokens, head_dim}));
    merged = modules::TransposeModule({{0, 2, 1, 3}, merged.shape.rank}).build(ctx, merged);
    merged = core::ensure_backend_addressable_layout(ctx, merged);
    merged = core::reshape_tensor(ctx, merged, core::TensorShape::from_dims({batch, tokens, config.hidden_size}));
    return modules::LinearModule({
        config.hidden_size,
        config.hidden_size,
        true,
        options.projection_precision,
    }).build(ctx, merged, {weights.out_weight, weights.out_bias});
}

core::TensorValue repeat_group_cls(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cls,
    int64_t repeats) {
    const int64_t batch_heads = cls.shape.dims[0];
    const int64_t dim = cls.shape.dims[2];
    auto reshaped = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, cls),
        core::TensorShape::from_dims({batch_heads, 1, 1, dim}));
    auto repeated = modules::RepeatModule({core::TensorShape::from_dims({batch_heads, repeats, 1, dim})}).build(ctx, reshaped);
    repeated = core::ensure_backend_addressable_layout(ctx, repeated);
    return core::reshape_tensor(ctx, repeated, core::TensorShape::from_dims({batch_heads * repeats, 1, dim}));
}

core::TensorValue attend_grouped(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q,
    const core::TensorValue & k,
    const core::TensorValue & v,
    const SynchformerConfig & config,
    const SynchformerRuntimeOptions & options) {
    const int64_t head_dim = config.hidden_size / config.heads;
    auto q4 = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, q), core::TensorShape::from_dims({q.shape.dims[0], 1, q.shape.dims[1], head_dim}));
    auto k4 = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, k), core::TensorShape::from_dims({k.shape.dims[0], 1, k.shape.dims[1], head_dim}));
    auto v4 = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, v), core::TensorShape::from_dims({v.shape.dims[0], 1, v.shape.dims[1], head_dim}));
    auto out = modules::GroupedQueryAttentionModule({
        head_dim,
        modules::GroupedQueryAttentionLowering::FlashGroupedViewKV,
        options.attention_precision,
        modules::AttentionCausality::NonCausal,
    }).build(ctx, q4, k4, v4);
    out = core::ensure_backend_addressable_layout(ctx, out);
    return core::reshape_tensor(ctx, out, core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[1], head_dim}));
}

core::TensorValue divided_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::AttentionWeights & weights,
    bool temporal,
    const SynchformerConfig & config,
    const SynchformerRuntimeOptions & options) {
    const int64_t batch = input.shape.dims[0];
    const int64_t temporal_tokens = config.frames / config.temporal_patch_size;
    const int64_t grid = config.image_size / config.patch_size;
    const int64_t spatial_tokens = grid * grid;
    const int64_t patch_tokens = temporal_tokens * spatial_tokens;
    const int64_t head_dim = config.hidden_size / config.heads;
    const int64_t batch_heads = batch * config.heads;
    auto heads = project_heads(ctx, input, weights, config, options);
    auto cls_q = modules::SliceModule({1, 0, 1}).build(ctx, heads.q);
    auto cls_k = modules::SliceModule({1, 0, 1}).build(ctx, heads.k);
    auto cls_v = modules::SliceModule({1, 0, 1}).build(ctx, heads.v);
    auto cls_out = attend_grouped(ctx, cls_q, heads.k, heads.v, config, options);
    auto q = modules::SliceModule({1, 1, patch_tokens}).build(ctx, heads.q);
    auto k = modules::SliceModule({1, 1, patch_tokens}).build(ctx, heads.k);
    auto v = modules::SliceModule({1, 1, patch_tokens}).build(ctx, heads.v);
    q = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, q), core::TensorShape::from_dims({batch_heads, temporal_tokens, spatial_tokens, head_dim}));
    k = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, k), core::TensorShape::from_dims({batch_heads, temporal_tokens, spatial_tokens, head_dim}));
    v = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, v), core::TensorShape::from_dims({batch_heads, temporal_tokens, spatial_tokens, head_dim}));
    int64_t repeats = temporal_tokens;
    int64_t group_steps = spatial_tokens;
    if (temporal) {
        q = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
        k = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
        v = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
        repeats = spatial_tokens;
        group_steps = temporal_tokens;
    }
    q = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, q), core::TensorShape::from_dims({batch_heads * repeats, group_steps, head_dim}));
    k = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, k), core::TensorShape::from_dims({batch_heads * repeats, group_steps, head_dim}));
    v = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, v), core::TensorShape::from_dims({batch_heads * repeats, group_steps, head_dim}));
    auto grouped_k = modules::ConcatModule({1}).build(ctx, repeat_group_cls(ctx, cls_k, repeats), k);
    auto grouped_v = modules::ConcatModule({1}).build(ctx, repeat_group_cls(ctx, cls_v, repeats), v);
    auto grouped_out = attend_grouped(ctx, q, grouped_k, grouped_v, config, options);
    if (temporal) {
        grouped_out = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, grouped_out), core::TensorShape::from_dims({batch_heads, spatial_tokens, temporal_tokens, head_dim}));
        grouped_out = modules::TransposeModule({{0, 2, 1, 3}, grouped_out.shape.rank}).build(ctx, grouped_out);
    } else {
        grouped_out = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, grouped_out), core::TensorShape::from_dims({batch_heads, temporal_tokens, spatial_tokens, head_dim}));
    }
    grouped_out = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, grouped_out), core::TensorShape::from_dims({batch_heads, patch_tokens, head_dim}));
    auto out = modules::ConcatModule({1}).build(ctx, cls_out, grouped_out);
    return merge_heads(ctx, out, batch, input.shape.dims[1], weights, config, options);
}

core::TensorValue apply_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const SynchformerBlockWeights & weights,
    const SynchformerConfig & config,
    const SynchformerRuntimeOptions & options) {
    const modules::ResidualAddModule residual_add;
    auto time_input = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                          .build(ctx, input, weights.norm3);
    auto time_residual = residual_add.build(ctx, input, divided_attention(ctx, time_input, weights.time_attention, true, config, options));
    auto space_input = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                           .build(ctx, time_residual, weights.norm1);
    auto space_residual = residual_add.build(ctx, time_residual, divided_attention(ctx, space_input, weights.space_attention, false, config, options));
    auto mlp_input = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                         .build(ctx, space_residual, weights.norm2);
    const modules::FeedForwardModule feed_forward({
        config.hidden_size,
        config.intermediate_size,
        true,
        modules::GeluApproximation::ExactErf,
        options.projection_precision,
    });
    return residual_add.build(ctx, space_residual, feed_forward.build(ctx, mlp_input, weights.mlp));
}

core::TensorValue spatial_aggregate(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & features,
    const SynchformerSpatialAggregatorWeights & weights,
    const SynchformerConfig & config,
    const SynchformerRuntimeOptions & options) {
    const int64_t batch_segments = features.shape.dims[0];
    const int64_t temporal_tokens = config.frames / config.temporal_patch_size;
    const int64_t grid = config.image_size / config.patch_size;
    const int64_t spatial_tokens = grid * grid;
    auto x = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, features),
        core::TensorShape::from_dims({batch_segments, config.hidden_size, temporal_tokens, spatial_tokens}));
    x = modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
    x = modules::TransposeModule({{0, 1, 3, 2}, x.shape.rank}).build(ctx, x);
    x = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, x),
        core::TensorShape::from_dims({batch_segments * temporal_tokens, spatial_tokens, config.hidden_size}));
    auto cls = modules::RepeatModule({core::TensorShape::from_dims({batch_segments * temporal_tokens, 1, config.hidden_size})})
                   .build(ctx, weights.cls_token);
    x = modules::ConcatModule({1}).build(ctx, cls, x);
    const modules::ResidualAddModule residual_add;
    auto attn_input = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                          .build(ctx, x, weights.norm1);
    modules::AttentionConfig aggregator_attention_config{
        config.hidden_size,
        config.heads,
        true,
        options.projection_precision,
        options.attention_precision,
        modules::AttentionPrefixCacheLayout::SequenceHeads,
        true,
        false,
    };
    aggregator_attention_config.use_flash_attention = true;
    const modules::SelfAttentionModule attention(aggregator_attention_config);
    x = residual_add.build(ctx, x, attention.build(ctx, attn_input, weights.attention));
    auto mlp_input = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                         .build(ctx, x, weights.norm2);
    const modules::FeedForwardModule feed_forward({
        config.hidden_size,
        config.intermediate_size,
        true,
        modules::GeluApproximation::ExactErf,
        options.projection_precision,
    });
    x = residual_add.build(ctx, x, feed_forward.build(ctx, mlp_input, weights.mlp));
    x = modules::SliceModule({1, 0, 1}).build(ctx, x);
    x = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, x),
        core::TensorShape::from_dims({batch_segments, temporal_tokens, config.hidden_size}));
    return x;
}

core::TensorValue build_synchformer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & video,
    const SynchformerWeights & weights,
    const SynchformerConfig & config,
    const SynchformerRuntimeOptions & options) {
    const int64_t temporal_tokens = config.frames / config.temporal_patch_size;
    const int64_t grid = config.image_size / config.patch_size;
    const int64_t spatial_tokens = grid * grid;
    auto hidden = modules::Conv3dModule({
        config.image_channels,
        config.hidden_size,
        config.temporal_patch_size,
        config.patch_size,
        config.patch_size,
        static_cast<int>(config.temporal_patch_size),
        static_cast<int>(config.patch_size),
        static_cast<int>(config.patch_size),
        0,
        0,
        0,
        1,
        1,
        1,
        true,
    }).build(ctx, video, weights.patch_embedding);
    hidden = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, hidden), core::TensorShape::from_dims({video.shape.dims[0] / config.image_channels, config.hidden_size, temporal_tokens * spatial_tokens}));
    hidden = modules::TransposeModule({{0, 2, 1, 3}, hidden.shape.rank}).build(ctx, hidden);
    auto cls = modules::RepeatModule({core::TensorShape::from_dims({hidden.shape.dims[0], 1, config.hidden_size})})
                   .build(ctx, weights.cls_token);
    hidden = modules::ConcatModule({1}).build(ctx, cls, hidden);
    hidden = modules::AddModule().build(ctx, hidden, modules::RepeatModule({hidden.shape}).build(ctx, weights.position));
    for (const auto & block : weights.blocks) {
        hidden = apply_block(ctx, hidden, block, config, options);
    }
    hidden = modules::SliceModule({1, 1, temporal_tokens * spatial_tokens}).build(ctx, hidden);
    hidden = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                 .build(ctx, hidden, weights.norm);
    hidden = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, hidden), core::TensorShape::from_dims({video.shape.dims[0] / config.image_channels, temporal_tokens, spatial_tokens, config.hidden_size}));
    hidden = modules::TransposeModule({{0, 1, 3, 2}, hidden.shape.rank}).build(ctx, hidden);
    hidden = modules::TransposeModule({{0, 2, 1, 3}, hidden.shape.rank}).build(ctx, hidden);
    return spatial_aggregate(ctx, hidden, weights.spatial_aggregator, config, options);
}

core::TensorValue build_overlapping_segments(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & frames,
    const SynchformerConfig & config,
    int64_t stride_frames) {
    if (stride_frames <= 0) {
        throw std::runtime_error("Synchformer frame stride must be positive");
    }
    if (frames.shape.rank != 3 || frames.shape.dims[0] % config.image_channels != 0 ||
        frames.shape.dims[0] / config.image_channels < config.frames) {
        throw std::runtime_error("Synchformer continuous frame input shape mismatch");
    }
    const int64_t source_frames = frames.shape.dims[0] / config.image_channels;
    const int64_t segments = (source_frames - config.frames) / stride_frames + 1;
    if (segments <= 0) {
        throw std::runtime_error("Synchformer continuous frame input produced no segments");
    }
    core::TensorValue out;
    for (int64_t segment = 0; segment < segments; ++segment) {
        core::TensorValue segment_value;
        for (int64_t frame = 0; frame < config.frames; ++frame) {
            const int64_t source_frame = segment * stride_frames + frame;
            auto frame_value = modules::SliceModule({0, source_frame * config.image_channels, config.image_channels}).build(ctx, frames);
            frame_value = core::reshape_tensor(
                ctx,
                frame_value,
                core::TensorShape::from_dims({config.image_channels, 1, config.image_size, config.image_size}));
            if (frame == 0) {
                segment_value = frame_value;
            } else {
                segment_value = modules::ConcatModule({1}).build(ctx, segment_value, frame_value);
            }
        }
        if (segment == 0) {
            out = segment_value;
        } else {
            out = modules::ConcatModule({0}).build(ctx, out, segment_value);
        }
    }
    return out;
}

}  // namespace

struct SynchformerConditionerRuntime::Impl {
    Impl(
        std::shared_ptr<const assets::TensorSource> input_source,
        core::ExecutionContext & input_execution,
        SynchformerConfig input_config,
        SynchformerRuntimeOptions input_options)
        : source(std::move(input_source)),
          execution(input_execution),
          backend(input_execution.backend()),
          backend_type(input_execution.backend_type()),
          config(std::move(input_config)),
          options(input_options) {
        if (source == nullptr) {
            throw std::runtime_error("Synchformer runtime requires a tensor source");
        }
        if (backend == nullptr) {
            throw std::runtime_error("Synchformer runtime requires an initialized backend");
        }
        validate_config(config);
        weights = std::make_shared<SynchformerWeights>(
            load_weights(*source, config, backend, backend_type, options));
    }

    struct VideoGraph {
        VideoGraph(const Impl & owner, int64_t input_batch_segments, int64_t input_frames, int64_t input_height, int64_t input_width)
            : batch_segments(input_batch_segments),
              frames(input_frames),
              height(input_height),
              width(input_width),
              owner_backend(owner.backend) {
            ggml_init_params params{owner.options.graph_arena_bytes, nullptr, true};
            ctx.reset(ggml_init(params));
            if (ctx == nullptr) {
                throw std::runtime_error("Synchformer failed to create graph context");
            }
            core::ModuleBuildContext build{ctx.get(), "framework.synchformer", owner.backend_type};
            video = core::make_tensor(
                build,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({batch_segments * owner.config.image_channels, frames, height, width}));
            output = build_synchformer(build, video, *owner.weights, owner.config, owner.options);
            graph = ggml_new_graph_custom(ctx.get(), 65536, false);
            ggml_set_output(output.tensor);
            ggml_build_forward_expand(graph, output.tensor);
            gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(owner.backend));
            if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
                throw std::runtime_error("Synchformer failed to allocate graph");
            }
        }

        ~VideoGraph() {
            if (owner_backend != nullptr && graph != nullptr) {
                core::release_backend_graph_resources(owner_backend, graph);
                graph = nullptr;
            }
            if (gallocr != nullptr) {
                ggml_gallocr_free(gallocr);
                gallocr = nullptr;
            }
        }

        VideoGraph(const VideoGraph &) = delete;
        VideoGraph & operator=(const VideoGraph &) = delete;

        int64_t batch_segments = 0;
        int64_t frames = 0;
        int64_t height = 0;
        int64_t width = 0;
        ggml_backend_t owner_backend = nullptr;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
        ggml_cgraph * graph = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        core::TensorValue video;
        core::TensorValue output;
    };

    struct FrameGraph {
        FrameGraph(const Impl & owner, int64_t input_frames, int64_t input_height, int64_t input_width, int64_t input_stride_frames)
            : frames(input_frames),
              height(input_height),
              width(input_width),
              stride_frames(input_stride_frames),
              batch_segments((input_frames - owner.config.frames) / input_stride_frames + 1),
              owner_backend(owner.backend) {
            ggml_init_params params{owner.options.graph_arena_bytes, nullptr, true};
            ctx.reset(ggml_init(params));
            if (ctx == nullptr) {
                throw std::runtime_error("Synchformer failed to create frame graph context");
            }
            core::ModuleBuildContext build{ctx.get(), "framework.synchformer.frames", owner.backend_type};
            video = core::make_tensor(
                build,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({frames * owner.config.image_channels, height, width}));
            auto segmented = build_overlapping_segments(build, video, owner.config, stride_frames);
            output = build_synchformer(build, segmented, *owner.weights, owner.config, owner.options);
            graph = ggml_new_graph_custom(ctx.get(), 65536, false);
            ggml_set_output(output.tensor);
            ggml_build_forward_expand(graph, output.tensor);
            gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(owner.backend));
            if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
                throw std::runtime_error("Synchformer failed to allocate frame graph");
            }
        }

        ~FrameGraph() {
            if (owner_backend != nullptr && graph != nullptr) {
                core::release_backend_graph_resources(owner_backend, graph);
                graph = nullptr;
            }
            if (gallocr != nullptr) {
                ggml_gallocr_free(gallocr);
                gallocr = nullptr;
            }
        }

        FrameGraph(const FrameGraph &) = delete;
        FrameGraph & operator=(const FrameGraph &) = delete;

        int64_t frames = 0;
        int64_t height = 0;
        int64_t width = 0;
        int64_t stride_frames = 0;
        int64_t batch_segments = 0;
        ggml_backend_t owner_backend = nullptr;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
        ggml_cgraph * graph = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        core::TensorValue video;
        core::TensorValue output;
    };

    VideoGraph & graph_for_shape(int64_t batch_segments, int64_t frames, int64_t height, int64_t width) {
        if (graph == nullptr || graph->batch_segments != batch_segments || graph->frames != frames ||
            graph->height != height || graph->width != width) {
            graph = std::make_unique<VideoGraph>(*this, batch_segments, frames, height, width);
        }
        return *graph;
    }

    FrameGraph & frame_graph_for_shape(int64_t frames, int64_t height, int64_t width, int64_t stride_frames) {
        if (frame_graph == nullptr || frame_graph->frames != frames || frame_graph->height != height ||
            frame_graph->width != width || frame_graph->stride_frames != stride_frames) {
            frame_graph = std::make_unique<FrameGraph>(*this, frames, height, width, stride_frames);
        }
        return *frame_graph;
    }

    SynchformerVideoFeatures encode_segments(
        const std::vector<float> & video_values,
        int64_t batch_segments,
        int64_t frames,
        int64_t height,
        int64_t width) {
        if (batch_segments <= 0 || frames != config.frames || height != config.image_size || width != config.image_size ||
            static_cast<int64_t>(video_values.size()) != batch_segments * config.image_channels * frames * height * width) {
            throw std::runtime_error("Synchformer input shape mismatch");
        }
        auto & active_graph = graph_for_shape(batch_segments, frames, height, width);
        const double upload_ms = debug::measure_ms([&]() {
            core::write_tensor_f32(active_graph.video, video_values);
        });
        ggml_status status = GGML_STATUS_SUCCESS;
        const double compute_ms = debug::measure_ms([&]() {
            core::set_backend_threads(execution.backend(), execution.config().threads);
            status = core::compute_backend_graph(execution.backend(), active_graph.graph, nullptr, "framework.synchformer");
            ggml_backend_synchronize(execution.backend());
        });
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Synchformer graph compute failed");
        }
        SynchformerVideoFeatures features;
        features.batch_segments = batch_segments;
        features.temporal_tokens = config.frames / config.temporal_patch_size;
        features.hidden = config.hidden_size;
        const double read_ms = debug::measure_ms([&]() {
            core::read_tensor_float_into(active_graph.output.tensor, features.values);
        });
        if (static_cast<int64_t>(features.values.size()) != features.batch_segments * features.temporal_tokens * features.hidden) {
            throw std::runtime_error("Synchformer output shape mismatch");
        }
        debug::timing_log_scalar("framework.synchformer.upload_ms", upload_ms);
        debug::timing_log_scalar("framework.synchformer.compute_ms", compute_ms);
        debug::timing_log_scalar("framework.synchformer.read_ms", read_ms);
        return features;
    }

    SynchformerVideoFeatures encode_frames(
        const std::vector<float> & video_values,
        int64_t frames,
        int64_t height,
        int64_t width,
        int64_t stride_frames) {
        if (frames < config.frames || height != config.image_size || width != config.image_size ||
            stride_frames <= 0 ||
            static_cast<int64_t>(video_values.size()) != config.image_channels * frames * height * width) {
            throw std::runtime_error("Synchformer continuous frame input shape mismatch");
        }
        auto & active_graph = frame_graph_for_shape(frames, height, width, stride_frames);
        const double upload_ms = debug::measure_ms([&]() {
            core::write_tensor_f32(active_graph.video, video_values);
        });
        ggml_status status = GGML_STATUS_SUCCESS;
        const double compute_ms = debug::measure_ms([&]() {
            core::set_backend_threads(execution.backend(), execution.config().threads);
            status = core::compute_backend_graph(execution.backend(), active_graph.graph, nullptr, "framework.synchformer.frames");
            ggml_backend_synchronize(execution.backend());
        });
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Synchformer frame graph compute failed");
        }
        SynchformerVideoFeatures features;
        features.batch_segments = active_graph.batch_segments;
        features.temporal_tokens = config.frames / config.temporal_patch_size;
        features.hidden = config.hidden_size;
        const double read_ms = debug::measure_ms([&]() {
            core::read_tensor_float_into(active_graph.output.tensor, features.values);
        });
        if (static_cast<int64_t>(features.values.size()) != features.batch_segments * features.temporal_tokens * features.hidden) {
            throw std::runtime_error("Synchformer frame output shape mismatch");
        }
        debug::timing_log_scalar("framework.synchformer.frames.upload_ms", upload_ms);
        debug::timing_log_scalar("framework.synchformer.frames.compute_ms", compute_ms);
        debug::timing_log_scalar("framework.synchformer.frames.read_ms", read_ms);
        return features;
    }

    std::shared_ptr<const assets::TensorSource> source;
    core::ExecutionContext & execution;
    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    SynchformerConfig config;
    SynchformerRuntimeOptions options;
    std::shared_ptr<SynchformerWeights> weights;
    std::unique_ptr<VideoGraph> graph;
    std::unique_ptr<FrameGraph> frame_graph;
};

SynchformerConditionerRuntime::SynchformerConditionerRuntime(
    std::shared_ptr<const assets::TensorSource> source,
    core::ExecutionContext & execution,
    SynchformerConfig config,
    SynchformerRuntimeOptions options)
    : impl_(std::make_unique<Impl>(std::move(source), execution, std::move(config), options)) {}

SynchformerConditionerRuntime::~SynchformerConditionerRuntime() = default;

SynchformerVideoFeatures SynchformerConditionerRuntime::encode_segments(
    const std::vector<float> & video,
    int64_t batch_segments,
    int64_t frames,
    int64_t height,
    int64_t width) {
    return impl_->encode_segments(video, batch_segments, frames, height, width);
}

SynchformerVideoFeatures SynchformerConditionerRuntime::encode_frames(
    const std::vector<float> & video,
    int64_t frames,
    int64_t height,
    int64_t width,
    int64_t stride_frames) {
    return impl_->encode_frames(video, frames, height, width, stride_frames);
}

}  // namespace engine::conditioners
