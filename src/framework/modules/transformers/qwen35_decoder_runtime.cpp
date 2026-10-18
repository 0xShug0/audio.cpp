#include "engine/framework/modules/transformers/qwen35_decoder_runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/attention/grouped_query_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/transformers/causal_decoder.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::modules {
namespace {

namespace binding = engine::modules::binding;
namespace core = engine::core;
namespace modules = engine::modules;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept {
        if (alloc != nullptr) {
            ggml_gallocr_free(alloc);
        }
    }
};

struct GraphMemory {
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr;
    ggml_backend_buffer_t input_buffer = nullptr;
    ggml_cgraph * graph = nullptr;

    ~GraphMemory() {
        reset(nullptr);
    }

    void reset(ggml_backend_t backend) {
        if (graph != nullptr && backend != nullptr) {
            core::release_backend_graph_resources(backend, graph);
        }
        graph = nullptr;
        gallocr.reset();
        if (input_buffer != nullptr) {
            ggml_backend_buffer_free(input_buffer);
            input_buffer = nullptr;
        }
        input_ctx.reset();
        ctx.reset();
    }
};

std::vector<int32_t> position_ids(int64_t steps) {
    std::vector<int32_t> out(static_cast<size_t>(steps));
    for (int64_t i = 0; i < steps; ++i) {
        out[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    return out;
}

struct FullAttentionWeights {
    modules::LinearWeights q_proj;
    modules::LinearWeights k_proj;
    modules::LinearWeights v_proj;
    modules::LinearWeights o_proj;
    modules::NormWeights q_norm;
    modules::NormWeights k_norm;
};

struct LinearAttentionWeights {
    modules::LinearWeights in_proj_qkv;
    modules::LinearWeights in_proj_z;
    modules::LinearWeights in_proj_b;
    modules::LinearWeights in_proj_a;
    core::TensorValue conv_weight;
    core::TensorValue dt_bias;
    core::TensorValue a_log;
    modules::NormWeights norm;
    modules::LinearWeights out_proj;
};

struct Qwen35LayerWeights {
    bool full_attention = false;
    modules::NormWeights input_norm;
    modules::NormWeights post_norm;
    std::optional<FullAttentionWeights> full;
    std::optional<LinearAttentionWeights> linear;
    modules::LinearWeights mlp_gate;
    modules::LinearWeights mlp_up;
    modules::LinearWeights mlp_down;
};

struct Qwen35Weights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    std::vector<Qwen35LayerWeights> layers;
    modules::NormWeights final_norm;
    modules::LinearWeights lm_head;
};

FullAttentionWeights load_full_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const Qwen35DecoderConfig & config,
    assets::TensorStorageType storage_type) {
    FullAttentionWeights out;
    out.q_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".q_proj",
        storage_type,
        config.heads * config.head_dim * 2,
        config.hidden_size,
        false);
    out.k_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".k_proj",
        storage_type,
        config.kv_heads * config.head_dim,
        config.hidden_size,
        false);
    out.v_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".v_proj",
        storage_type,
        config.kv_heads * config.head_dim,
        config.hidden_size,
        false);
    out.o_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".o_proj",
        storage_type,
        config.hidden_size,
        config.heads * config.head_dim,
        false);
    out.q_norm = binding::norm_weight_from_source(store, source, prefix + ".q_norm", config.head_dim);
    out.k_norm = binding::norm_weight_from_source(store, source, prefix + ".k_norm", config.head_dim);
    return out;
}

LinearAttentionWeights load_linear_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const Qwen35DecoderConfig & config,
    assets::TensorStorageType storage_type) {
    LinearAttentionWeights out;
    const int64_t key_dim = config.linear_key_head_dim * config.linear_num_key_heads;
    const int64_t value_dim = config.linear_value_head_dim * config.linear_num_value_heads;
    const int64_t conv_dim = key_dim * 2 + value_dim;
    out.in_proj_qkv = binding::linear_from_source(store, source, prefix + ".in_proj_qkv", storage_type, conv_dim, config.hidden_size, false);
    out.in_proj_z = binding::linear_from_source(store, source, prefix + ".in_proj_z", storage_type, value_dim, config.hidden_size, false);
    out.in_proj_b = binding::linear_from_source(store, source, prefix + ".in_proj_b", storage_type, config.linear_num_value_heads, config.hidden_size, false);
    out.in_proj_a = binding::linear_from_source(store, source, prefix + ".in_proj_a", storage_type, config.linear_num_value_heads, config.hidden_size, false);
    out.conv_weight = store.load_tensor(
        source,
        prefix + ".conv1d.weight",
        assets::TensorStorageType::F32,
        {conv_dim, 1, config.linear_conv_kernel_dim});
    out.dt_bias = store.load_f32_tensor(source, prefix + ".dt_bias", {config.linear_num_value_heads});
    out.a_log = store.load_f32_tensor(source, prefix + ".A_log", {config.linear_num_value_heads});
    out.norm = binding::norm_weight_from_source(store, source, prefix + ".norm", config.linear_value_head_dim);
    out.out_proj = binding::linear_from_source(store, source, prefix + ".out_proj", storage_type, config.hidden_size, value_dim, false);
    return out;
}

std::shared_ptr<Qwen35Weights> load_qwen35_weights(
    const assets::TensorSource & source,
    const Qwen35DecoderConfig & config,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    auto weights = std::make_shared<Qwen35Weights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "qwen35_decoder.weights",
        weight_context_bytes);
    const auto & c = config;
    const std::string & root = c.weight_prefix;
    weights->token_embedding = weights->store->load_tensor(
        source,
        root + ".embed_tokens.weight",
        storage_type,
        {c.vocab_size, c.hidden_size});
    weights->layers.reserve(static_cast<size_t>(c.layers));
    for (int64_t layer = 0; layer < c.layers; ++layer) {
        const std::string prefix = root + ".layers." + std::to_string(layer);
        Qwen35LayerWeights out;
        out.full_attention = c.full_attention_interval > 0 && ((layer + 1) % c.full_attention_interval) == 0;
        out.input_norm = binding::norm_weight_from_source(*weights->store, source, prefix + ".input_layernorm", c.hidden_size);
        out.post_norm = binding::norm_weight_from_source(*weights->store, source, prefix + ".post_attention_layernorm", c.hidden_size);
        if (out.full_attention) {
            out.full = load_full_attention(*weights->store, source, prefix + ".self_attn", c, storage_type);
        } else {
            out.linear = load_linear_attention(*weights->store, source, prefix + ".linear_attn", c, storage_type);
        }
        out.mlp_gate = binding::linear_from_source(*weights->store, source, prefix + ".mlp.gate_proj", storage_type, c.intermediate_size, c.hidden_size, false);
        out.mlp_up = binding::linear_from_source(*weights->store, source, prefix + ".mlp.up_proj", storage_type, c.intermediate_size, c.hidden_size, false);
        out.mlp_down = binding::linear_from_source(*weights->store, source, prefix + ".mlp.down_proj", storage_type, c.hidden_size, c.intermediate_size, false);
        weights->layers.push_back(std::move(out));
    }
    weights->final_norm = binding::norm_weight_from_source(*weights->store, source, root + ".norm", c.hidden_size);
    if (c.tie_word_embeddings) {
        weights->lm_head = {weights->token_embedding, std::nullopt};
    } else {
        weights->lm_head = binding::linear_from_source(
            *weights->store,
            source,
            c.lm_head_prefix,
            storage_type,
            c.vocab_size,
            c.hidden_size,
            false);
    }
    weights->store->upload();
    return weights;
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t head_dim) {
    return core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, input),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, head_dim}));
}

core::TensorValue cache_view(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    int64_t start,
    int64_t steps,
    int64_t heads,
    int64_t head_dim) {
    if (start < 0 || steps <= 0 || start + steps > cache.shape.dims[1]) {
        throw std::runtime_error("Qwen3.5 decoder cache view range is invalid");
    }
    return core::wrap_tensor(
        ggml_view_4d(
            ctx.ggml,
            cache.tensor,
            head_dim,
            heads,
            steps,
            1,
            cache.tensor->nb[1],
            cache.tensor->nb[2],
            cache.tensor->nb[3],
            static_cast<size_t>(start) * cache.tensor->nb[2]),
        core::TensorShape::from_dims({1, steps, heads, head_dim}),
        cache.type);
}

core::TensorValue round_activation(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Qwen35DecoderConfig & config) {
    if (!config.round_bf16_activations) {
        return input;
    }
    return core::wrap_tensor(ggml_round_bf16(ctx.ggml, input.tensor), input.shape, GGML_TYPE_F32);
}

core::TensorValue mlp(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const Qwen35LayerWeights & weights,
    const Qwen35DecoderConfig & config) {
    auto gate = modules::LinearModule({config.hidden_size, config.intermediate_size, false}).build(ctx, x, weights.mlp_gate);
    gate = round_activation(ctx, gate, config);
    gate = modules::SiluModule{}.build(ctx, gate);
    gate = round_activation(ctx, gate, config);
    auto up = modules::LinearModule({config.hidden_size, config.intermediate_size, false}).build(ctx, x, weights.mlp_up);
    up = round_activation(ctx, up, config);
    auto gated = modules::MulModule{}.build(ctx, gate, up);
    gated = round_activation(ctx, gated, config);
    return round_activation(ctx,
        modules::LinearModule({config.intermediate_size, config.hidden_size, false}).build(ctx, gated, weights.mlp_down), config);
}

core::TensorValue apply_partial_rope(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const Qwen35DecoderConfig & config) {
    const int64_t rotary_dim = static_cast<int64_t>(std::llround(
        static_cast<double>(config.head_dim) * static_cast<double>(config.partial_rotary_factor)));
    if (rotary_dim <= 0 || rotary_dim > config.head_dim || (rotary_dim % 2) != 0) {
        throw std::runtime_error("Qwen3.5 decoder rotary dimension is invalid");
    }
    if (rotary_dim == config.head_dim) {
        return modules::RoPEModule({config.head_dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, input, positions);
    }
    auto rotary = modules::SliceModule({3, 0, rotary_dim}).build(ctx, input);
    auto pass = modules::SliceModule({3, rotary_dim, config.head_dim - rotary_dim}).build(ctx, input);
    rotary = modules::RoPEModule({rotary_dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, rotary, positions);
    return modules::ConcatModule({3}).build(ctx, rotary, pass);
}

core::TensorValue full_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue & attention_mask,
    const FullAttentionWeights & weights,
    const Qwen35DecoderConfig & config) {
    auto q_gate = modules::LinearModule({config.hidden_size, config.heads * config.head_dim * 2, false}).build(ctx, input, weights.q_proj);
    q_gate = round_activation(ctx, q_gate, config);
    q_gate = reshape_heads(ctx, q_gate, config.heads, config.head_dim * 2);
    auto q = modules::SliceModule({3, 0, config.head_dim}).build(ctx, q_gate);
    auto gate = modules::SliceModule({3, config.head_dim, config.head_dim}).build(ctx, q_gate);
    gate = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, gate),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.heads * config.head_dim}));
    auto k = modules::LinearModule({config.hidden_size, config.kv_heads * config.head_dim, false}).build(ctx, input, weights.k_proj);
    auto v = modules::LinearModule({config.hidden_size, config.kv_heads * config.head_dim, false}).build(ctx, input, weights.v_proj);
    k = round_activation(ctx, k, config);
    v = round_activation(ctx, v, config);
    k = reshape_heads(ctx, k, config.kv_heads, config.head_dim);
    v = reshape_heads(ctx, v, config.kv_heads, config.head_dim);
    q = modules::GemmaRMSNormModule({config.head_dim, config.rms_norm_eps, true, false}).build(ctx, q, weights.q_norm);
    k = modules::GemmaRMSNormModule({config.head_dim, config.rms_norm_eps, true, false}).build(ctx, k, weights.k_norm);
    q = round_activation(ctx, q, config);
    k = round_activation(ctx, k, config);
    q = apply_partial_rope(ctx, q, positions, config);
    k = apply_partial_rope(ctx, k, positions, config);
    q = round_activation(ctx, q, config);
    k = round_activation(ctx, k, config);
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context = modules::GroupedQueryAttentionModule({
        config.head_dim,
        modules::GroupedQueryAttentionLowering::FlashGroupedViewKV,
        GGML_PREC_F32,
        modules::AttentionCausality::Causal,
    }).build(ctx, q_heads, k_heads, v_heads, attention_mask);
    context = round_activation(ctx, context, config);
    context = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, context),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.heads * config.head_dim}));
    gate = modules::SigmoidModule{}.build(ctx, gate);
    gate = round_activation(ctx, gate, config);
    context = modules::MulModule{}.build(ctx, context, gate);
    context = round_activation(ctx, context, config);
    return round_activation(ctx,
        modules::LinearModule({config.heads * config.head_dim, config.hidden_size, false}).build(ctx, context, weights.o_proj), config);
}

core::TensorValue full_attention_prefill_cached(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue & attention_mask,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const FullAttentionWeights & weights,
    const Qwen35DecoderConfig & config) {
    const int64_t steps = input.shape.dims[1];
    auto q_gate = modules::LinearModule({config.hidden_size, config.heads * config.head_dim * 2, false}).build(ctx, input, weights.q_proj);
    q_gate = round_activation(ctx, q_gate, config);
    q_gate = reshape_heads(ctx, q_gate, config.heads, config.head_dim * 2);
    auto q = modules::SliceModule({3, 0, config.head_dim}).build(ctx, q_gate);
    auto gate = modules::SliceModule({3, config.head_dim, config.head_dim}).build(ctx, q_gate);
    gate = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, gate),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.heads * config.head_dim}));
    auto k = modules::LinearModule({config.hidden_size, config.kv_heads * config.head_dim, false}).build(ctx, input, weights.k_proj);
    auto v = modules::LinearModule({config.hidden_size, config.kv_heads * config.head_dim, false}).build(ctx, input, weights.v_proj);
    k = round_activation(ctx, k, config);
    v = round_activation(ctx, v, config);
    k = reshape_heads(ctx, k, config.kv_heads, config.head_dim);
    v = reshape_heads(ctx, v, config.kv_heads, config.head_dim);
    q = modules::GemmaRMSNormModule({config.head_dim, config.rms_norm_eps, true, false}).build(ctx, q, weights.q_norm);
    k = modules::GemmaRMSNormModule({config.head_dim, config.rms_norm_eps, true, false}).build(ctx, k, weights.k_norm);
    q = round_activation(ctx, q, config);
    k = round_activation(ctx, k, config);
    q = apply_partial_rope(ctx, q, positions, config);
    k = apply_partial_rope(ctx, k, positions, config);
    q = round_activation(ctx, q, config);
    k = round_activation(ctx, k, config);
    k = core::ensure_backend_addressable_layout(ctx, k);
    v = core::ensure_backend_addressable_layout(ctx, v);
    auto key_dest = cache_view(ctx, cache_key, 0, steps, config.kv_heads, config.head_dim);
    auto value_dest = cache_view(ctx, cache_value, 0, steps, config.kv_heads, config.head_dim);
    ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, k.tensor, key_dest.tensor));
    ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, v.tensor, value_dest.tensor));
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context = modules::GroupedQueryAttentionModule({
        config.head_dim,
        modules::GroupedQueryAttentionLowering::FlashGroupedViewKV,
        GGML_PREC_F32,
        modules::AttentionCausality::Causal,
    }).build(ctx, q_heads, k_heads, v_heads, attention_mask);
    context = round_activation(ctx, context, config);
    context = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, context),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.heads * config.head_dim}));
    gate = modules::SigmoidModule{}.build(ctx, gate);
    gate = round_activation(ctx, gate, config);
    context = modules::MulModule{}.build(ctx, context, gate);
    context = round_activation(ctx, context, config);
    return round_activation(ctx,
        modules::LinearModule({config.heads * config.head_dim, config.hidden_size, false}).build(ctx, context, weights.o_proj), config);
}

core::TensorValue full_attention_cached(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue & cache_slot,
    const core::TensorValue & attention_mask,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const FullAttentionWeights & weights,
    const Qwen35DecoderConfig & config) {
    auto q_gate = modules::LinearModule({config.hidden_size, config.heads * config.head_dim * 2, false}).build(ctx, input, weights.q_proj);
    q_gate = round_activation(ctx, q_gate, config);
    q_gate = reshape_heads(ctx, q_gate, config.heads, config.head_dim * 2);
    auto q = modules::SliceModule({3, 0, config.head_dim}).build(ctx, q_gate);
    auto gate = modules::SliceModule({3, config.head_dim, config.head_dim}).build(ctx, q_gate);
    gate = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, gate),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.heads * config.head_dim}));
    auto k = modules::LinearModule({config.hidden_size, config.kv_heads * config.head_dim, false}).build(ctx, input, weights.k_proj);
    auto v = modules::LinearModule({config.hidden_size, config.kv_heads * config.head_dim, false}).build(ctx, input, weights.v_proj);
    k = round_activation(ctx, k, config);
    v = round_activation(ctx, v, config);
    k = reshape_heads(ctx, k, config.kv_heads, config.head_dim);
    v = reshape_heads(ctx, v, config.kv_heads, config.head_dim);
    q = modules::GemmaRMSNormModule({config.head_dim, config.rms_norm_eps, true, false}).build(ctx, q, weights.q_norm);
    k = modules::GemmaRMSNormModule({config.head_dim, config.rms_norm_eps, true, false}).build(ctx, k, weights.k_norm);
    q = round_activation(ctx, q, config);
    k = round_activation(ctx, k, config);
    q = apply_partial_rope(ctx, q, positions, config);
    k = apply_partial_rope(ctx, k, positions, config);
    q = round_activation(ctx, q, config);
    k = round_activation(ctx, k, config);
    k = core::ensure_backend_addressable_layout(ctx, k);
    v = core::ensure_backend_addressable_layout(ctx, v);

    const modules::FastKVSetRowsModule set_rows({modules::FastKVSetRowsMode::BackendViewOptimized});
    auto attention_key_cache = set_rows.build(ctx, cache_key, k, cache_slot);
    auto attention_value_cache = set_rows.build(ctx, cache_value, v, cache_slot);

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, attention_key_cache.shape.rank}).build(ctx, attention_key_cache);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, attention_value_cache.shape.rank}).build(ctx, attention_value_cache);
    auto context = modules::GroupedQueryAttentionModule({
        config.head_dim,
        modules::GroupedQueryAttentionLowering::FlashGroupedViewKV,
        GGML_PREC_F32,
        modules::AttentionCausality::NonCausal,
    }).build(ctx, q_heads, k_heads, v_heads, attention_mask);
    context = round_activation(ctx, context, config);
    context = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, context),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.heads * config.head_dim}));
    gate = modules::SigmoidModule{}.build(ctx, gate);
    gate = round_activation(ctx, gate, config);
    context = modules::MulModule{}.build(ctx, context, gate);
    context = round_activation(ctx, context, config);
    return round_activation(ctx,
        modules::LinearModule({config.heads * config.head_dim, config.hidden_size, false}).build(ctx, context, weights.o_proj), config);
}

core::TensorValue repeat_head_values(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & values,
    int64_t steps,
    int64_t heads) {
    auto shaped = core::reshape_tensor(ctx, values, core::TensorShape::from_dims({1, 1, heads}));
    return modules::RepeatModule({core::TensorShape::from_dims({1, steps, heads})}).build(ctx, shaped);
}

core::TensorValue repeat_linear_attention_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t repeats) {
    if (repeats == 1) {
        return input;
    }
    if (input.shape.rank != 4) {
        throw std::runtime_error("Qwen3.5 decoder linear attention head repeat expects rank-4 input");
    }
    std::vector<core::TensorValue> repeated_heads;
    repeated_heads.reserve(static_cast<size_t>(input.shape.dims[2] * repeats));
    for (int64_t head = 0; head < input.shape.dims[2]; ++head) {
        auto one = modules::SliceModule({2, head, 1}).build(ctx, input);
        for (int64_t rep = 0; rep < repeats; ++rep) {
            repeated_heads.push_back(one);
        }
    }
    auto output = repeated_heads.front();
    for (size_t i = 1; i < repeated_heads.size(); ++i) {
        output = modules::ConcatModule({2}).build(ctx, output, repeated_heads[i]);
    }
    return output;
}

struct LinearAttentionCachedOutput {
    core::TensorValue output;
    core::TensorValue next_conv_tail;
    core::TensorValue next_recurrent_state;
};

LinearAttentionCachedOutput linear_attention_prefill(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & state,
    const LinearAttentionWeights & weights,
    const Qwen35DecoderConfig & config) {
    const int64_t steps = input.shape.dims[1];
    const int64_t key_dim = config.linear_key_head_dim * config.linear_num_key_heads;
    const int64_t value_dim = config.linear_value_head_dim * config.linear_num_value_heads;
    const int64_t conv_dim = key_dim * 2 + value_dim;
    auto qkv = modules::LinearModule({config.hidden_size, conv_dim, false}).build(ctx, input, weights.in_proj_qkv);
    qkv = round_activation(ctx, qkv, config);
    qkv = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, qkv);
    qkv = core::ensure_backend_addressable_layout(ctx, qkv);
    auto * padded_raw = ggml_pad_ext(
        ctx.ggml,
        qkv.tensor,
        static_cast<int>(config.linear_conv_kernel_dim - 1),
        0,
        0,
        0,
        0,
        0,
        0,
        0);
    auto padded = core::wrap_tensor(
        padded_raw,
        core::TensorShape::from_dims({1, conv_dim, steps + config.linear_conv_kernel_dim - 1}),
        GGML_TYPE_F32);
    auto conv_weight = core::reshape_tensor(ctx, weights.conv_weight, core::TensorShape::from_dims({conv_dim, config.linear_conv_kernel_dim}));
    auto conv = core::wrap_tensor(
        ggml_ssm_conv(ctx.ggml, padded.tensor, conv_weight.tensor),
        core::TensorShape::from_dims({1, steps, conv_dim}),
        GGML_TYPE_F32);
    conv = round_activation(ctx, conv, config);
    conv = modules::SiluModule{}.build(ctx, conv);
    conv = round_activation(ctx, conv, config);
    auto query = modules::SliceModule({2, 0, key_dim}).build(ctx, conv);
    auto key = modules::SliceModule({2, key_dim, key_dim}).build(ctx, conv);
    auto value = modules::SliceModule({2, 2 * key_dim, value_dim}).build(ctx, conv);
    query = reshape_heads(ctx, query, config.linear_num_key_heads, config.linear_key_head_dim);
    key = reshape_heads(ctx, key, config.linear_num_key_heads, config.linear_key_head_dim);
    value = reshape_heads(ctx, value, config.linear_num_value_heads, config.linear_value_head_dim);
    query = core::wrap_tensor(ggml_l2_norm(ctx.ggml, query.tensor, 1.0e-6F), query.shape, GGML_TYPE_F32);
    key = core::wrap_tensor(ggml_l2_norm(ctx.ggml, key.tensor, 1.0e-6F), key.shape, GGML_TYPE_F32);
    query = round_activation(ctx, query, config);
    key = round_activation(ctx, key, config);
    if (config.linear_num_value_heads % config.linear_num_key_heads != 0) {
        throw std::runtime_error("Qwen3.5 decoder linear attention key heads must divide value heads");
    }
    const int64_t key_repeats = config.linear_num_value_heads / config.linear_num_key_heads;
    query = repeat_linear_attention_heads(ctx, query, key_repeats);
    key = repeat_linear_attention_heads(ctx, key, key_repeats);
    auto beta = modules::LinearModule({config.hidden_size, config.linear_num_value_heads, false}).build(ctx, input, weights.in_proj_b);
    beta = round_activation(ctx, beta, config);
    beta = modules::SigmoidModule{}.build(ctx, beta);
    beta = round_activation(ctx, beta, config);
    auto a = modules::LinearModule({config.hidden_size, config.linear_num_value_heads, false}).build(ctx, input, weights.in_proj_a);
    a = round_activation(ctx, a, config);
    auto dt_bias = repeat_head_values(ctx, weights.dt_bias, steps, config.linear_num_value_heads);
    a = modules::AddModule{}.build(ctx, a, dt_bias);
    a = core::wrap_tensor(ggml_softplus(ctx.ggml, a.tensor), a.shape, GGML_TYPE_F32);
    auto a_log = repeat_head_values(ctx, weights.a_log, steps, config.linear_num_value_heads);
    a_log = core::wrap_tensor(ggml_exp(ctx.ggml, a_log.tensor), a_log.shape, GGML_TYPE_F32);
    auto g = modules::MulModule{}.build(ctx, a, a_log);
    g = core::wrap_tensor(ggml_scale(ctx.ggml, g.tensor, -1.0F), g.shape, GGML_TYPE_F32);
    beta = core::reshape_tensor(ctx, beta, core::TensorShape::from_dims({1, steps, config.linear_num_value_heads, 1}));
    g = core::reshape_tensor(ctx, g, core::TensorShape::from_dims({1, steps, config.linear_num_value_heads, 1}));
    auto delta = core::wrap_tensor(
        ggml_gated_delta_net(ctx.ggml, query.tensor, key.tensor, value.tensor, g.tensor, beta.tensor, state.tensor),
        core::TensorShape::from_dims({steps + config.linear_value_head_dim, value_dim}),
        GGML_TYPE_F32);
    auto current_delta = modules::SliceModule({0, 0, steps}).build(ctx, delta);
    auto next_state = modules::SliceModule({0, steps, config.linear_value_head_dim}).build(ctx, delta);
    current_delta = round_activation(ctx, current_delta, config);
    next_state = round_activation(ctx, next_state, config);
    next_state = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, next_state), state.shape);
    auto z = modules::LinearModule({config.hidden_size, value_dim, false}).build(ctx, input, weights.in_proj_z);
    z = round_activation(ctx, z, config);
    z = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, z), core::TensorShape::from_dims({steps, value_dim}));
    current_delta = modules::RMSNormModule({config.linear_value_head_dim, config.rms_norm_eps, true, false}).build(
        ctx,
        core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, current_delta), core::TensorShape::from_dims({steps * config.linear_num_value_heads, config.linear_value_head_dim})),
        weights.norm);
    current_delta = round_activation(ctx, current_delta, config);
    auto z_flat = core::reshape_tensor(ctx, z, core::TensorShape::from_dims({steps * config.linear_num_value_heads, config.linear_value_head_dim}));
    z_flat = modules::SiluModule{}.build(ctx, z_flat);
    z_flat = round_activation(ctx, z_flat, config);
    current_delta = modules::MulModule{}.build(ctx, current_delta, z_flat);
    current_delta = round_activation(ctx, current_delta, config);
    current_delta = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, current_delta), core::TensorShape::from_dims({1, steps, value_dim}));
    auto output = modules::LinearModule({value_dim, config.hidden_size, false}).build(ctx, current_delta, weights.out_proj);
    output = round_activation(ctx, output, config);
    auto next_tail = modules::SliceModule({2, steps, config.linear_conv_kernel_dim - 1}).build(ctx, padded);
    next_tail = core::ensure_backend_addressable_layout(ctx, next_tail);
    return {std::move(output), std::move(next_tail), std::move(next_state)};
}

core::TensorValue linear_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & state,
    const LinearAttentionWeights & weights,
    const Qwen35DecoderConfig & config) {
    return linear_attention_prefill(ctx, input, state, weights, config).output;
}

LinearAttentionCachedOutput linear_attention_cached(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & conv_tail,
    const core::TensorValue & recurrent_state,
    const LinearAttentionWeights & weights,
    const Qwen35DecoderConfig & config) {
    const int64_t steps = input.shape.dims[1];
    if (steps != 1) {
        throw std::runtime_error("Qwen3.5 decoder cached linear attention requires one decode step");
    }
    const int64_t key_dim = config.linear_key_head_dim * config.linear_num_key_heads;
    const int64_t value_dim = config.linear_value_head_dim * config.linear_num_value_heads;
    const int64_t conv_dim = key_dim * 2 + value_dim;
    auto qkv = modules::LinearModule({config.hidden_size, conv_dim, false}).build(ctx, input, weights.in_proj_qkv);
    qkv = round_activation(ctx, qkv, config);
    qkv = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, qkv);
    qkv = core::ensure_backend_addressable_layout(ctx, qkv);
    auto conv_input = modules::ConcatModule({2}).build(ctx, conv_tail, qkv);
    auto conv_weight = core::reshape_tensor(ctx, weights.conv_weight, core::TensorShape::from_dims({conv_dim, config.linear_conv_kernel_dim}));
    auto conv = core::wrap_tensor(
        ggml_ssm_conv(ctx.ggml, conv_input.tensor, conv_weight.tensor),
        core::TensorShape::from_dims({1, 1, conv_dim}),
        GGML_TYPE_F32);
    conv = round_activation(ctx, conv, config);
    conv = modules::SiluModule{}.build(ctx, conv);
    conv = round_activation(ctx, conv, config);
    auto query = modules::SliceModule({2, 0, key_dim}).build(ctx, conv);
    auto key = modules::SliceModule({2, key_dim, key_dim}).build(ctx, conv);
    auto value = modules::SliceModule({2, 2 * key_dim, value_dim}).build(ctx, conv);
    query = reshape_heads(ctx, query, config.linear_num_key_heads, config.linear_key_head_dim);
    key = reshape_heads(ctx, key, config.linear_num_key_heads, config.linear_key_head_dim);
    value = reshape_heads(ctx, value, config.linear_num_value_heads, config.linear_value_head_dim);
    query = core::wrap_tensor(ggml_l2_norm(ctx.ggml, query.tensor, 1.0e-6F), query.shape, GGML_TYPE_F32);
    key = core::wrap_tensor(ggml_l2_norm(ctx.ggml, key.tensor, 1.0e-6F), key.shape, GGML_TYPE_F32);
    query = round_activation(ctx, query, config);
    key = round_activation(ctx, key, config);
    if (config.linear_num_value_heads % config.linear_num_key_heads != 0) {
        throw std::runtime_error("Qwen3.5 decoder linear attention key heads must divide value heads");
    }
    const int64_t key_repeats = config.linear_num_value_heads / config.linear_num_key_heads;
    query = repeat_linear_attention_heads(ctx, query, key_repeats);
    key = repeat_linear_attention_heads(ctx, key, key_repeats);
    auto beta = modules::LinearModule({config.hidden_size, config.linear_num_value_heads, false}).build(ctx, input, weights.in_proj_b);
    beta = round_activation(ctx, beta, config);
    beta = modules::SigmoidModule{}.build(ctx, beta);
    beta = round_activation(ctx, beta, config);
    auto a = modules::LinearModule({config.hidden_size, config.linear_num_value_heads, false}).build(ctx, input, weights.in_proj_a);
    a = round_activation(ctx, a, config);
    auto dt_bias = repeat_head_values(ctx, weights.dt_bias, 1, config.linear_num_value_heads);
    a = modules::AddModule{}.build(ctx, a, dt_bias);
    a = core::wrap_tensor(ggml_softplus(ctx.ggml, a.tensor), a.shape, GGML_TYPE_F32);
    auto a_log = repeat_head_values(ctx, weights.a_log, 1, config.linear_num_value_heads);
    a_log = core::wrap_tensor(ggml_exp(ctx.ggml, a_log.tensor), a_log.shape, GGML_TYPE_F32);
    auto g = modules::MulModule{}.build(ctx, a, a_log);
    g = core::wrap_tensor(ggml_scale(ctx.ggml, g.tensor, -1.0F), g.shape, GGML_TYPE_F32);
    beta = core::reshape_tensor(ctx, beta, core::TensorShape::from_dims({1, 1, config.linear_num_value_heads, 1}));
    g = core::reshape_tensor(ctx, g, core::TensorShape::from_dims({1, 1, config.linear_num_value_heads, 1}));
    auto delta = core::wrap_tensor(
        ggml_gated_delta_net(ctx.ggml, query.tensor, key.tensor, value.tensor, g.tensor, beta.tensor, recurrent_state.tensor),
        core::TensorShape::from_dims({1 + config.linear_value_head_dim, value_dim}),
        GGML_TYPE_F32);
    auto current_delta = modules::SliceModule({0, 0, 1}).build(ctx, delta);
    auto next_state = modules::SliceModule({0, 1, config.linear_value_head_dim}).build(ctx, delta);
    current_delta = round_activation(ctx, current_delta, config);
    next_state = round_activation(ctx, next_state, config);
    next_state = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, next_state), recurrent_state.shape);
    auto z = modules::LinearModule({config.hidden_size, value_dim, false}).build(ctx, input, weights.in_proj_z);
    z = round_activation(ctx, z, config);
    z = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, z), core::TensorShape::from_dims({1, value_dim}));
    current_delta = modules::RMSNormModule({config.linear_value_head_dim, config.rms_norm_eps, true, false}).build(
        ctx,
        core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, current_delta), core::TensorShape::from_dims({config.linear_num_value_heads, config.linear_value_head_dim})),
        weights.norm);
    current_delta = round_activation(ctx, current_delta, config);
    auto z_flat = core::reshape_tensor(ctx, z, core::TensorShape::from_dims({config.linear_num_value_heads, config.linear_value_head_dim}));
    z_flat = modules::SiluModule{}.build(ctx, z_flat);
    z_flat = round_activation(ctx, z_flat, config);
    current_delta = modules::MulModule{}.build(ctx, current_delta, z_flat);
    current_delta = round_activation(ctx, current_delta, config);
    current_delta = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, current_delta), core::TensorShape::from_dims({1, 1, value_dim}));
    auto output = modules::LinearModule({value_dim, config.hidden_size, false}).build(ctx, current_delta, weights.out_proj);
    output = round_activation(ctx, output, config);
    auto next_tail = modules::SliceModule({2, 1, config.linear_conv_kernel_dim - 1}).build(ctx, conv_input);
    next_tail = core::ensure_backend_addressable_layout(ctx, next_tail);
    return {std::move(output), std::move(next_tail), std::move(next_state)};
}

class BackboneForwardGraph {
public:
    BackboneForwardGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const Qwen35Weights> weights,
        Qwen35DecoderConfig config,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          graph_arena_bytes_(graph_arena_bytes) {}

    ~BackboneForwardGraph() {
        mem_.reset(execution_.backend());
    }

    Qwen35DecoderForwardResult run(const std::vector<float> & embeddings, int64_t steps) {
        if (steps <= 0 || static_cast<int64_t>(embeddings.size()) != steps * config_.hidden_size) {
            throw std::runtime_error("Qwen3.5 decoder embedding input size mismatch");
        }
        ensure(steps);
        core::write_tensor_f32(core::wrap_tensor(input_, core::TensorShape::from_dims({1, steps, config_.hidden_size})), embeddings);
        if (core::compute_backend_graph(execution_.backend(), mem_.graph, nullptr, "qwen35_decoder.forward") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Qwen3.5 decoder graph compute failed");
        }
        Qwen35DecoderForwardResult out;
        out.steps = steps;
        const int64_t expected_output_values = steps * config_.hidden_size;
        if (ggml_nelements(output_) != expected_output_values) {
            throw std::runtime_error("Qwen3.5 decoder output tensor size mismatch");
        }
        out.hidden = core::read_tensor_f32(output_);
        if (static_cast<int64_t>(out.hidden.size()) != expected_output_values) {
            throw std::runtime_error("Qwen3.5 decoder readback size mismatch");
        }
        return out;
    }

    void release_graph() {
        mem_.reset(execution_.backend());
        steps_ = 0;
        input_ = nullptr;
        positions_ = nullptr;
        attention_mask_ = nullptr;
        output_ = nullptr;
        state_inputs_.clear();
    }

private:
    void ensure(int64_t steps) {
        if (mem_.graph != nullptr && steps_ == steps) {
            return;
        }
        mem_.reset(execution_.backend());
        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        mem_.ctx.reset(ggml_init(params));
        ggml_init_params input_params{16ull * 1024ull * 1024ull, nullptr, true};
        mem_.input_ctx.reset(ggml_init(input_params));
        core::ModuleBuildContext ctx{mem_.ctx.get(), "qwen35_decoder.forward", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{mem_.input_ctx.get(), "qwen35_decoder.forward.inputs", execution_.backend_type()};
        auto x = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, steps, config_.hidden_size}));
        input_ = x.tensor;
        ggml_set_input(input_);
        auto positions = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
        positions_ = positions.tensor;
        attention_mask_ = ggml_new_tensor_4d(mem_.input_ctx.get(), GGML_TYPE_F16, steps, steps, 1, 1);
        ggml_set_input(attention_mask_);
        auto attention_mask = core::wrap_tensor(
            attention_mask_,
            core::TensorShape::from_dims({1, 1, steps, steps}),
            GGML_TYPE_F16);
        state_inputs_.clear();
        for (const auto & layer : weights_->layers) {
            auto residual = x;
            auto h = modules::GemmaRMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false}).build(ctx, x, layer.input_norm);
            h = round_activation(ctx, h, config_);
            if (layer.full_attention) {
                h = full_attention(ctx, h, positions, attention_mask, *layer.full, config_);
            } else {
                auto state = core::make_tensor(
                    input_ctx,
                    GGML_TYPE_F32,
                    core::TensorShape::from_dims({
                        1,
                        1,
                        config_.linear_value_head_dim * config_.linear_value_head_dim * config_.linear_num_value_heads,
                    }));
                state_inputs_.push_back(state.tensor);
                ggml_set_input(state.tensor);
                h = linear_attention(ctx, h, state, *layer.linear, config_);
            }
            x = modules::AddModule{}.build(ctx, residual, h);
            x = round_activation(ctx, x, config_);
            residual = x;
            h = modules::GemmaRMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false}).build(ctx, x, layer.post_norm);
            h = round_activation(ctx, h, config_);
            h = mlp(ctx, h, layer, config_);
            x = modules::AddModule{}.build(ctx, residual, h);
            x = round_activation(ctx, x, config_);
        }
        x = modules::GemmaRMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false}).build(ctx, x, weights_->final_norm);
        x = round_activation(ctx, x, config_);
        output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        ggml_set_output(output_);
        mem_.graph = ggml_new_graph_custom(mem_.ctx.get(), 1600000, false);
        ggml_build_forward_expand(mem_.graph, output_);
        mem_.input_buffer = ggml_backend_alloc_ctx_tensors(mem_.input_ctx.get(), execution_.backend());
        mem_.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (mem_.input_buffer == nullptr || mem_.gallocr == nullptr ||
            !ggml_gallocr_reserve(mem_.gallocr.get(), mem_.graph) ||
            !ggml_gallocr_alloc_graph(mem_.gallocr.get(), mem_.graph)) {
            mem_.reset(execution_.backend());
            throw std::runtime_error("failed to allocate Qwen3.5 decoder graph");
        }
        const auto pos = position_ids(steps);
        ggml_backend_tensor_set(positions_, pos.data(), 0, pos.size() * sizeof(int32_t));
        const auto mask = modules::causal_prefill_mask_values(1, steps);
        ggml_backend_tensor_set(attention_mask_, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        const size_t state_values = static_cast<size_t>(
            config_.linear_value_head_dim * config_.linear_value_head_dim * config_.linear_num_value_heads);
        std::vector<float> zero_state(state_values, 0.0F);
        for (ggml_tensor * state : state_inputs_) {
            ggml_backend_tensor_set(state, zero_state.data(), 0, zero_state.size() * sizeof(float));
        }
        steps_ = steps;
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const Qwen35Weights> weights_;
    Qwen35DecoderConfig config_;
    size_t graph_arena_bytes_;
    GraphMemory mem_;
    int64_t steps_ = 0;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * output_ = nullptr;
    std::vector<ggml_tensor *> state_inputs_;
};

class TokenEmbeddingGraph {
public:
    TokenEmbeddingGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const Qwen35Weights> weights,
        Qwen35DecoderConfig config,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          graph_arena_bytes_(graph_arena_bytes) {}

    ~TokenEmbeddingGraph() {
        mem_.reset(execution_.backend());
    }

    std::vector<float> run(const std::vector<int32_t> & token_ids) {
        if (token_ids.empty()) {
            throw std::runtime_error("Qwen3.5 decoder token embedding requires tokens");
        }
        const int64_t steps = static_cast<int64_t>(token_ids.size());
        ensure(steps);
        ggml_backend_tensor_set(input_, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
        if (core::compute_backend_graph(execution_.backend(), mem_.graph, nullptr, "qwen35_decoder.token_embedding") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Qwen3.5 decoder token embedding graph compute failed");
        }
        return core::read_tensor_f32(output_);
    }

    void release_graph() {
        mem_.reset(execution_.backend());
        steps_ = 0;
        input_ = nullptr;
        output_ = nullptr;
    }

private:
    void ensure(int64_t steps) {
        if (mem_.graph != nullptr && steps_ == steps) {
            return;
        }
        mem_.reset(execution_.backend());
        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        mem_.ctx.reset(ggml_init(params));
        ggml_init_params input_params{8ull * 1024ull * 1024ull, nullptr, true};
        mem_.input_ctx.reset(ggml_init(input_params));
        core::ModuleBuildContext ctx{mem_.ctx.get(), "qwen35_decoder.token_embedding", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{mem_.input_ctx.get(), "qwen35_decoder.token_embedding.inputs", execution_.backend_type()};
        auto ids = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
        input_ = ids.tensor;
        ggml_set_input(input_);
        auto x = modules::EmbeddingModule({config_.vocab_size, config_.hidden_size}).build(ctx, ids, weights_->token_embedding);
        output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        ggml_set_output(output_);
        mem_.graph = ggml_new_graph_custom(mem_.ctx.get(), 4096, false);
        ggml_build_forward_expand(mem_.graph, output_);
        mem_.input_buffer = ggml_backend_alloc_ctx_tensors(mem_.input_ctx.get(), execution_.backend());
        mem_.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (mem_.input_buffer == nullptr || mem_.gallocr == nullptr ||
            !ggml_gallocr_reserve(mem_.gallocr.get(), mem_.graph) ||
            !ggml_gallocr_alloc_graph(mem_.gallocr.get(), mem_.graph)) {
            mem_.reset(execution_.backend());
            throw std::runtime_error("failed to allocate Qwen3.5 decoder token embedding graph");
        }
        steps_ = steps;
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const Qwen35Weights> weights_;
    Qwen35DecoderConfig config_;
    size_t graph_arena_bytes_;
    GraphMemory mem_;
    int64_t steps_ = 0;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

class LmHeadGraph {
public:
    LmHeadGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const Qwen35Weights> weights,
        Qwen35DecoderConfig config,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          graph_arena_bytes_(graph_arena_bytes) {}

    ~LmHeadGraph() {
        mem_.reset(execution_.backend());
    }

    std::vector<float> run(const std::vector<float> & hidden) {
        if (static_cast<int64_t>(hidden.size()) != config_.hidden_size) {
            throw std::runtime_error("Qwen3.5 decoder LM head hidden size mismatch");
        }
        ensure();
        core::write_tensor_f32(core::wrap_tensor(input_, core::TensorShape::from_dims({1, 1, config_.hidden_size})), hidden);
        if (core::compute_backend_graph(execution_.backend(), mem_.graph, nullptr, "qwen35_decoder.lm_head") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Qwen3.5 decoder LM head graph compute failed");
        }
        return core::read_tensor_f32(output_);
    }

    void release_graph() {
        mem_.reset(execution_.backend());
        input_ = nullptr;
        output_ = nullptr;
    }

private:
    void ensure() {
        if (mem_.graph != nullptr) {
            return;
        }
        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        mem_.ctx.reset(ggml_init(params));
        ggml_init_params input_params{8ull * 1024ull * 1024ull, nullptr, true};
        mem_.input_ctx.reset(ggml_init(input_params));
        core::ModuleBuildContext ctx{mem_.ctx.get(), "qwen35_decoder.lm_head", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{mem_.input_ctx.get(), "qwen35_decoder.lm_head.inputs", execution_.backend_type()};
        auto x = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, config_.hidden_size}));
        input_ = x.tensor;
        ggml_set_input(input_);
        x = modules::LinearModule({config_.hidden_size, config_.vocab_size, false}).build(ctx, x, weights_->lm_head);
        output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        ggml_set_output(output_);
        mem_.graph = ggml_new_graph_custom(mem_.ctx.get(), 4096, false);
        ggml_build_forward_expand(mem_.graph, output_);
        mem_.input_buffer = ggml_backend_alloc_ctx_tensors(mem_.input_ctx.get(), execution_.backend());
        mem_.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (mem_.input_buffer == nullptr || mem_.gallocr == nullptr ||
            !ggml_gallocr_reserve(mem_.gallocr.get(), mem_.graph) ||
            !ggml_gallocr_alloc_graph(mem_.gallocr.get(), mem_.graph)) {
            mem_.reset(execution_.backend());
            throw std::runtime_error("failed to allocate Qwen3.5 decoder LM head graph");
        }
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const Qwen35Weights> weights_;
    Qwen35DecoderConfig config_;
    size_t graph_arena_bytes_;
    GraphMemory mem_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

class Qwen35CachedDecodeGraph {
public:
    Qwen35CachedDecodeGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const Qwen35Weights> weights,
        Qwen35DecoderConfig config,
        int64_t cache_steps,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          cache_steps_(cache_steps) {
        if (cache_steps_ <= 0) {
            throw std::runtime_error("Qwen3.5 decoder cached decode requires positive cache length");
        }
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Qwen3.5 decoder cached decode graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "qwen35_decoder.cached_decode", execution_.backend_type()};
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, config_.hidden_size})).tensor;
        ggml_set_input(input_);
        positions_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1})).tensor;
        ggml_set_input(positions_);
        cache_slot_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1})).tensor;
        ggml_set_input(cache_slot_);
        attention_mask_ = core::make_tensor(ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, 1, 1, cache_steps_})).tensor;
        ggml_set_input(attention_mask_);

        auto x = core::wrap_tensor(input_, core::TensorShape::from_dims({1, 1, config_.hidden_size}), GGML_TYPE_F32);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto cache_slot = core::wrap_tensor(cache_slot_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto attention_mask = core::wrap_tensor(attention_mask_, core::TensorShape::from_dims({1, 1, 1, cache_steps_}), GGML_TYPE_F16);

        graph_ = ggml_new_graph_custom(ctx_.get(), 1600000, false);
        const int64_t state_values = config_.linear_value_head_dim * config_.linear_value_head_dim * config_.linear_num_value_heads;
        const int64_t key_dim = config_.linear_key_head_dim * config_.linear_num_key_heads;
        const int64_t value_dim = config_.linear_value_head_dim * config_.linear_num_value_heads;
        const int64_t conv_dim = key_dim * 2 + value_dim;
        for (const auto & layer : weights_->layers) {
            auto residual = x;
            auto h = modules::GemmaRMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false}).build(ctx, x, layer.input_norm);
            h = round_activation(ctx, h, config_);
            if (layer.full_attention) {
                auto key_cache = core::make_tensor(
                    ctx,
                    GGML_TYPE_F32,
                    core::TensorShape::from_dims({1, cache_steps_, config_.kv_heads, config_.head_dim}));
                auto value_cache = core::make_tensor(
                    ctx,
                    GGML_TYPE_F32,
                    core::TensorShape::from_dims({1, cache_steps_, config_.kv_heads, config_.head_dim}));
                full_key_caches_.push_back(key_cache.tensor);
                full_value_caches_.push_back(value_cache.tensor);
                h = full_attention_cached(ctx, h, positions, cache_slot, attention_mask, key_cache, value_cache, *layer.full, config_);
            } else {
                auto conv_tail = core::make_tensor(
                    ctx,
                    GGML_TYPE_F32,
                    core::TensorShape::from_dims({1, conv_dim, config_.linear_conv_kernel_dim - 1}));
                auto recurrent_state = core::make_tensor(
                    ctx,
                    GGML_TYPE_F32,
                    core::TensorShape::from_dims({1, 1, state_values}));
                linear_conv_tails_.push_back(conv_tail.tensor);
                linear_recurrent_states_.push_back(recurrent_state.tensor);
                auto out = linear_attention_cached(ctx, h, conv_tail, recurrent_state, *layer.linear, config_);
                ggml_build_forward_expand(graph_, ggml_cpy(ctx.ggml, out.next_conv_tail.tensor, conv_tail.tensor));
                ggml_build_forward_expand(graph_, ggml_cpy(ctx.ggml, out.next_recurrent_state.tensor, recurrent_state.tensor));
                h = std::move(out.output);
            }
            x = modules::AddModule{}.build(ctx, residual, h);
            x = round_activation(ctx, x, config_);
            residual = x;
            h = modules::GemmaRMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false}).build(ctx, x, layer.post_norm);
            h = round_activation(ctx, h, config_);
            h = mlp(ctx, h, layer, config_);
            x = modules::AddModule{}.build(ctx, residual, h);
            x = round_activation(ctx, x, config_);
        }
        x = modules::GemmaRMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false}).build(ctx, x, weights_->final_norm);
        x = round_activation(ctx, x, config_);
        output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        ggml_set_output(output_);
        ggml_build_forward_expand(graph_, output_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), execution_.backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Qwen3.5 decoder cached decode graph");
        }
        attention_mask_values_.assign(static_cast<size_t>(cache_steps_), ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        reset();
    }

    ~Qwen35CachedDecodeGraph() {
        prefill_mem_.reset(execution_.backend());
        core::release_backend_graph_resources(execution_.backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    void reset() {
        valid_steps_ = 0;
        const int64_t state_values = config_.linear_value_head_dim * config_.linear_value_head_dim * config_.linear_num_value_heads;
        const int64_t key_dim = config_.linear_key_head_dim * config_.linear_num_key_heads;
        const int64_t value_dim = config_.linear_value_head_dim * config_.linear_num_value_heads;
        const int64_t conv_dim = key_dim * 2 + value_dim;
        std::vector<float> zero_state(static_cast<size_t>(state_values), 0.0F);
        std::vector<float> zero_tail(static_cast<size_t>(conv_dim * (config_.linear_conv_kernel_dim - 1)), 0.0F);
        for (auto * state : linear_recurrent_states_) {
            ggml_backend_tensor_set(state, zero_state.data(), 0, zero_state.size() * sizeof(float));
        }
        for (auto * tail : linear_conv_tails_) {
            ggml_backend_tensor_set(tail, zero_tail.data(), 0, zero_tail.size() * sizeof(float));
        }
    }

    int64_t valid_steps() const noexcept {
        return valid_steps_;
    }

    int64_t cache_steps() const noexcept {
        return cache_steps_;
    }

    Qwen35DecoderForwardResult prefill_embeddings(const std::vector<float> & embeddings, int64_t steps) {
        if (steps <= 0 || static_cast<int64_t>(embeddings.size()) != steps * config_.hidden_size) {
            throw std::runtime_error("Qwen3.5 decoder prefill embedding input size mismatch");
        }
        if (steps > cache_steps_) {
            throw std::runtime_error("Qwen3.5 decoder prefill exceeds cache length");
        }
        ensure_prefill(steps);
        core::write_tensor_f32(core::wrap_tensor(prefill_input_, core::TensorShape::from_dims({1, steps, config_.hidden_size})), embeddings);
        if (core::compute_backend_graph(execution_.backend(), prefill_mem_.graph, nullptr, "qwen35_decoder.prefill") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Qwen3.5 decoder prefill graph compute failed");
        }
        Qwen35DecoderForwardResult out;
        out.steps = steps;
        out.hidden = core::read_tensor_f32(prefill_output_);
        if (static_cast<int64_t>(out.hidden.size()) != steps * config_.hidden_size) {
            throw std::runtime_error("Qwen3.5 decoder prefill readback size mismatch");
        }
        valid_steps_ = steps;
        return out;
    }

    std::vector<float> run_embedding_step(const std::vector<float> & embedding) {
        if (static_cast<int64_t>(embedding.size()) != config_.hidden_size) {
            throw std::runtime_error("Qwen3.5 decoder cached decode embedding size mismatch");
        }
        if (valid_steps_ >= cache_steps_) {
            throw std::runtime_error("Qwen3.5 decoder cached decode exceeded cache length");
        }
        ggml_backend_tensor_set(input_, embedding.data(), 0, embedding.size() * sizeof(float));
        const int32_t position = static_cast<int32_t>(valid_steps_);
        const int32_t slot = static_cast<int32_t>(valid_steps_);
        ggml_backend_tensor_set(positions_, &position, 0, sizeof(int32_t));
        ggml_backend_tensor_set(cache_slot_, &slot, 0, sizeof(int32_t));
        modules::write_decoder_cached_step_mask(
            attention_mask_,
            attention_mask_values_,
            cache_steps_,
            valid_steps_,
            valid_steps_);
        if (core::compute_backend_graph(execution_.backend(), graph_, nullptr, "qwen35_decoder.cached_decode") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Qwen3.5 decoder cached decode graph compute failed");
        }
        std::vector<float> hidden(static_cast<size_t>(config_.hidden_size));
        ggml_backend_tensor_get(output_, hidden.data(), 0, hidden.size() * sizeof(float));
        ++valid_steps_;
        return hidden;
    }

private:
    void ensure_prefill(int64_t steps) {
        if (prefill_mem_.graph != nullptr && prefill_steps_ == steps) {
            return;
        }
        prefill_mem_.reset(execution_.backend());
        prefill_state_inputs_.clear();
        prefill_attention_mask_ = nullptr;
        ggml_init_params params{128ull * 1024ull * 1024ull, nullptr, true};
        prefill_mem_.ctx.reset(ggml_init(params));
        ggml_init_params input_params{32ull * 1024ull * 1024ull, nullptr, true};
        prefill_mem_.input_ctx.reset(ggml_init(input_params));
        core::ModuleBuildContext ctx{prefill_mem_.ctx.get(), "qwen35_decoder.prefill", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{prefill_mem_.input_ctx.get(), "qwen35_decoder.prefill.inputs", execution_.backend_type()};
        auto x = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, steps, config_.hidden_size}));
        prefill_input_ = x.tensor;
        ggml_set_input(prefill_input_);
        auto positions = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
        prefill_positions_ = positions.tensor;
        ggml_set_input(prefill_positions_);
        prefill_attention_mask_ = ggml_new_tensor_4d(prefill_mem_.input_ctx.get(), GGML_TYPE_F16, steps, steps, 1, 1);
        ggml_set_input(prefill_attention_mask_);
        auto attention_mask = core::wrap_tensor(
            prefill_attention_mask_,
            core::TensorShape::from_dims({1, 1, steps, steps}),
            GGML_TYPE_F16);

        prefill_mem_.graph = ggml_new_graph_custom(prefill_mem_.ctx.get(), 1600000, false);
        const int64_t state_values = config_.linear_value_head_dim * config_.linear_value_head_dim * config_.linear_num_value_heads;
        const int64_t full_count = static_cast<int64_t>(full_key_caches_.size());
        const int64_t linear_count = static_cast<int64_t>(linear_recurrent_states_.size());
        int64_t full_index = 0;
        int64_t linear_index = 0;
        for (const auto & layer : weights_->layers) {
            auto residual = x;
            auto h = modules::GemmaRMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false}).build(ctx, x, layer.input_norm);
            h = round_activation(ctx, h, config_);
            if (layer.full_attention) {
                if (full_index >= full_count) {
                    throw std::runtime_error("Qwen3.5 decoder full-attention cache count mismatch");
                }
                h = full_attention_prefill_cached(
                    ctx,
                    prefill_mem_.graph,
                    h,
                    positions,
                    attention_mask,
                    core::wrap_tensor(
                        full_key_caches_[static_cast<size_t>(full_index)],
                        core::TensorShape::from_dims({1, cache_steps_, config_.kv_heads, config_.head_dim}),
                        GGML_TYPE_F32),
                    core::wrap_tensor(
                        full_value_caches_[static_cast<size_t>(full_index)],
                        core::TensorShape::from_dims({1, cache_steps_, config_.kv_heads, config_.head_dim}),
                        GGML_TYPE_F32),
                    *layer.full,
                    config_);
                ++full_index;
            } else {
                if (linear_index >= linear_count) {
                    throw std::runtime_error("Qwen3.5 decoder linear-attention state count mismatch");
                }
                auto state = core::make_tensor(
                    input_ctx,
                    GGML_TYPE_F32,
                    core::TensorShape::from_dims({1, 1, state_values}));
                prefill_state_inputs_.push_back(state.tensor);
                ggml_set_input(state.tensor);
                auto out = linear_attention_prefill(ctx, h, state, *layer.linear, config_);
                ggml_build_forward_expand(
                    prefill_mem_.graph,
                    ggml_cpy(ctx.ggml, out.next_conv_tail.tensor, linear_conv_tails_[static_cast<size_t>(linear_index)]));
                ggml_build_forward_expand(
                    prefill_mem_.graph,
                    ggml_cpy(ctx.ggml, out.next_recurrent_state.tensor, linear_recurrent_states_[static_cast<size_t>(linear_index)]));
                h = std::move(out.output);
                ++linear_index;
            }
            x = modules::AddModule{}.build(ctx, residual, h);
            x = round_activation(ctx, x, config_);
            residual = x;
            h = modules::GemmaRMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false}).build(ctx, x, layer.post_norm);
            h = round_activation(ctx, h, config_);
            h = mlp(ctx, h, layer, config_);
            x = modules::AddModule{}.build(ctx, residual, h);
            x = round_activation(ctx, x, config_);
        }
        if (full_index != full_count || linear_index != linear_count) {
            throw std::runtime_error("Qwen3.5 decoder prefill cache/state count mismatch");
        }
        x = modules::GemmaRMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false}).build(ctx, x, weights_->final_norm);
        x = round_activation(ctx, x, config_);
        prefill_output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        ggml_set_output(prefill_output_);
        ggml_build_forward_expand(prefill_mem_.graph, prefill_output_);
        prefill_mem_.input_buffer = ggml_backend_alloc_ctx_tensors(prefill_mem_.input_ctx.get(), execution_.backend());
        prefill_mem_.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (prefill_mem_.input_buffer == nullptr || prefill_mem_.gallocr == nullptr ||
            !ggml_gallocr_reserve(prefill_mem_.gallocr.get(), prefill_mem_.graph) ||
            !ggml_gallocr_alloc_graph(prefill_mem_.gallocr.get(), prefill_mem_.graph)) {
            prefill_mem_.reset(execution_.backend());
            throw std::runtime_error("failed to allocate Qwen3.5 decoder prefill graph");
        }
        const auto pos = position_ids(steps);
        ggml_backend_tensor_set(prefill_positions_, pos.data(), 0, pos.size() * sizeof(int32_t));
        const auto mask = modules::causal_prefill_mask_values(1, steps);
        ggml_backend_tensor_set(prefill_attention_mask_, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        std::vector<float> zero_state(static_cast<size_t>(state_values), 0.0F);
        for (ggml_tensor * state : prefill_state_inputs_) {
            ggml_backend_tensor_set(state, zero_state.data(), 0, zero_state.size() * sizeof(float));
        }
        prefill_steps_ = steps;
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const Qwen35Weights> weights_;
    Qwen35DecoderConfig config_;
    int64_t cache_steps_ = 0;
    int64_t valid_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * cache_slot_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * output_ = nullptr;
    std::vector<ggml_tensor *> full_key_caches_;
    std::vector<ggml_tensor *> full_value_caches_;
    std::vector<ggml_tensor *> linear_conv_tails_;
    std::vector<ggml_tensor *> linear_recurrent_states_;
    std::vector<ggml_fp16_t> attention_mask_values_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    GraphMemory prefill_mem_;
    int64_t prefill_steps_ = 0;
    ggml_tensor * prefill_input_ = nullptr;
    ggml_tensor * prefill_positions_ = nullptr;
    ggml_tensor * prefill_attention_mask_ = nullptr;
    ggml_tensor * prefill_output_ = nullptr;
    std::vector<ggml_tensor *> prefill_state_inputs_;
};

}  // namespace

class Qwen35DecoderRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const assets::TensorSource> source,
        Qwen35DecoderConfig config,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : config_(std::move(config)),
          weights_(load_qwen35_weights(*source, config_, execution, weight_context_bytes, storage_type)),
          token_embedding_(execution, weights_, config_, graph_arena_bytes),
          forward_(execution, weights_, config_, graph_arena_bytes),
          lm_head_(execution, weights_, config_, graph_arena_bytes),
          decode_graph_(execution, weights_, config_, config_.decode_cache_steps, graph_arena_bytes) {}

    std::vector<float> token_embedding(const std::vector<int32_t> & token_ids) {
        return token_embedding_.run(token_ids);
    }

    Qwen35DecoderForwardResult forward_embeddings(const std::vector<float> & embeddings, int64_t steps) {
        return forward_.run(embeddings, steps);
    }

    std::vector<float> lm_head(const std::vector<float> & hidden) {
        return lm_head_.run(hidden);
    }

    std::unique_ptr<Qwen35DecoderRuntime::DecodeSession> create_decode_session(int64_t cache_steps);

    Qwen35CachedDecodeGraph & decode_graph(int64_t cache_steps) {
        if (cache_steps > decode_graph_.cache_steps()) {
            throw std::runtime_error("Qwen3.5 decoder decode cache capacity exceeded");
        }
        decode_graph_.reset();
        return decode_graph_;
    }

    void release_graphs() {
        token_embedding_.release_graph();
        forward_.release_graph();
        lm_head_.release_graph();
        decode_graph_.reset();
    }

private:
    Qwen35DecoderConfig config_;
    std::shared_ptr<Qwen35Weights> weights_;
    TokenEmbeddingGraph token_embedding_;
    BackboneForwardGraph forward_;
    LmHeadGraph lm_head_;
    Qwen35CachedDecodeGraph decode_graph_;
};

class Qwen35DecodeSession final : public Qwen35DecoderRuntime::DecodeSession {
public:
    explicit Qwen35DecodeSession(Qwen35CachedDecodeGraph & graph) : graph_(graph) {}

    void reset() override {
        graph_.get().reset();
    }

    Qwen35DecoderForwardResult prefill_embeddings(
        const std::vector<float> & embeddings,
        int64_t steps) override {
        return graph_.get().prefill_embeddings(embeddings, steps);
    }

    std::vector<float> run_embedding_step(const std::vector<float> & embedding) override {
        return graph_.get().run_embedding_step(embedding);
    }

    int64_t valid_steps() const noexcept override {
        return graph_.get().valid_steps();
    }

private:
    std::reference_wrapper<Qwen35CachedDecodeGraph> graph_;
};

std::unique_ptr<Qwen35DecoderRuntime::DecodeSession> Qwen35DecoderRuntime::Impl::create_decode_session(int64_t cache_steps) {
    return std::make_unique<Qwen35DecodeSession>(decode_graph(cache_steps));
}

Qwen35DecoderRuntime::Qwen35DecoderRuntime(
    std::shared_ptr<const assets::TensorSource> source,
    Qwen35DecoderConfig config,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(std::move(source), std::move(config), execution,
                                   graph_arena_bytes, weight_context_bytes, storage_type)) {}

Qwen35DecoderRuntime::~Qwen35DecoderRuntime() = default;

std::vector<float> Qwen35DecoderRuntime::token_embedding(const std::vector<int32_t> & token_ids) {
    return impl_->token_embedding(token_ids);
}

Qwen35DecoderForwardResult Qwen35DecoderRuntime::forward_embeddings(
    const std::vector<float> & embeddings,
    int64_t steps) {
    return impl_->forward_embeddings(embeddings, steps);
}

std::vector<float> Qwen35DecoderRuntime::lm_head(const std::vector<float> & hidden) {
    return impl_->lm_head(hidden);
}

std::unique_ptr<Qwen35DecoderRuntime::DecodeSession> Qwen35DecoderRuntime::create_decode_session(int64_t cache_steps) {
    return impl_->create_decode_session(cache_steps);
}

void Qwen35DecoderRuntime::release_graphs() {
    impl_->release_graphs();
}

}  // namespace engine::modules
