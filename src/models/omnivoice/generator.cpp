#include "engine/models/omnivoice/generator.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/packed_linear_weights.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <chrono>
#include <utility>
#include <vector>

namespace engine::models::omnivoice {
namespace {

namespace assets_ns = engine::assets;
namespace modules = engine::modules;
namespace binding = engine::modules::binding;
using Clock = std::chrono::steady_clock;
constexpr float kMaskedAttentionBias = std::numeric_limits<float>::lowest();

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

void validate_weight_storage_type(assets_ns::TensorStorageType storage_type) {
    switch (storage_type) {
        case assets_ns::TensorStorageType::Native:
        case assets_ns::TensorStorageType::F32:
        case assets_ns::TensorStorageType::F16:
        case assets_ns::TensorStorageType::BF16:
        case assets_ns::TensorStorageType::Q8_0:
            return;
        default:
            throw std::runtime_error(
                "OmniVoice generator weight_type currently supports only native, f32, f16, bf16, and q8_0");
    }
}

int64_t checked_positive(int64_t value, const char * name) {
    if (value <= 0) {
        throw std::runtime_error(std::string("OmniVoice generator expected positive ") + name);
    }
    return value;
}

int64_t head_dim(const OmniVoiceQwen3DiffusionConfig & config) {
    if (config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 || config.head_dim <= 0) {
        throw std::runtime_error("OmniVoice generator attention configuration is invalid");
    }
    return config.head_dim;
}

core::TensorValue ensure_contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return core::ensure_backend_addressable_layout(ctx, value);
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    auto contiguous = ensure_contiguous(ctx, input);
    return core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

core::TensorValue repeat_kv_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t repeats) {
    if (repeats == 1) {
        return input;
    }
    if (input.shape.rank != 4) {
        throw std::runtime_error("OmniVoice generator repeat_kv_heads expects rank-4 input");
    }
    auto contiguous = ensure_contiguous(ctx, input);
    const int64_t batch = contiguous.shape.dims[0];
    const int64_t kv_heads = contiguous.shape.dims[1];
    const int64_t steps = contiguous.shape.dims[2];
    const int64_t dim = contiguous.shape.dims[3];
    auto expanded = core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({batch, kv_heads, 1, steps * dim}));
    expanded = modules::RepeatModule({core::TensorShape::from_dims({batch, kv_heads, repeats, steps * dim})})
                   .build(ctx, expanded);
    expanded = ensure_contiguous(ctx, expanded);
    return core::reshape_tensor(
        ctx,
        expanded,
        core::TensorShape::from_dims({batch, kv_heads * repeats, steps, dim}));
}

core::TensorValue attention_from_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const core::TensorValue & attention_mask,
    OmniVoiceGeneratorPerfMode perf_mode) {
    if (perf_mode == OmniVoiceGeneratorPerfMode::FlashAttention) {
        return modules::ScaledDotProductAttentionModule({
            dim,
            ctx.backend_type == core::BackendType::Cuda
                ? modules::ScaledDotProductAttentionLowering::FlashPreserveViews
                : modules::ScaledDotProductAttentionLowering::Flash,
            GGML_PREC_F32,
            modules::AttentionCausality::NonCausal,
        }).build(ctx, q_heads, k_heads, v_heads, attention_mask);
    }
    const modules::MatMulModule matmul;
    auto scores = matmul.build(
        ctx,
        q_heads,
        modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    scores = core::ensure_backend_addressable_layout(ctx, scores);
    auto attn = core::wrap_tensor(
        ggml_soft_max_ext(
            ctx.ggml,
            scores.tensor,
            attention_mask.tensor,
            1.0F / std::sqrt(static_cast<float>(dim)),
            0.0F),
        scores.shape,
        GGML_TYPE_F32);
    auto context = matmul.build(ctx, attn, v_heads);
    return modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
}

// [rows, tokens, heads, dim] view of `heads` heads starting at feature
// `feature_offset` of a contiguous packed projection [rows, tokens, width].
core::TensorValue packed_heads_view(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & packed,
    int64_t feature_offset,
    int64_t heads,
    int64_t dim) {
    ggml_tensor * tensor = packed.tensor;
    if (!ggml_is_contiguous(tensor) || packed.shape.rank != 3) {
        throw std::runtime_error("OmniVoice generator packed projection must be a contiguous rank-3 tensor");
    }
    const size_t element = ggml_element_size(tensor);
    auto * view = ggml_view_4d(
        ctx.ggml,
        tensor,
        dim,
        heads,
        tensor->ne[1],
        tensor->ne[2],
        static_cast<size_t>(dim) * element,
        tensor->nb[1],
        tensor->nb[2],
        static_cast<size_t>(feature_offset) * element);
    return core::wrap_tensor(
        view,
        core::TensorShape::from_dims({packed.shape.dims[0], packed.shape.dims[1], heads, dim}),
        packed.type);
}

// Flash attention on [batch, heads, steps, dim] heads where K/V may have
// fewer heads than Q (grouped-query attention): ggml_flash_attn_ext
// broadcasts each K/V head over its query-head group, so K/V are never
// repeated (the repeat costs copies, and the kernel would convert twice the
// K/V data to f16). Returns [batch, steps, heads, dim].
core::TensorValue grouped_flash_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const core::TensorValue & attention_mask) {
    if (q_heads.shape.dims[1] % k_heads.shape.dims[1] != 0 || k_heads.shape.dims[1] != v_heads.shape.dims[1]) {
        throw std::runtime_error("OmniVoice generator grouped attention head counts mismatch");
    }
    // CUDA reads the permuted views directly; other backends get contiguous inputs.
    const bool preserve_views = ctx.backend_type == core::BackendType::Cuda;
    const auto q = preserve_views ? q_heads : core::ensure_backend_addressable_layout(ctx, q_heads);
    const auto k = preserve_views ? k_heads : core::ensure_backend_addressable_layout(ctx, k_heads);
    const auto v = preserve_views ? v_heads : core::ensure_backend_addressable_layout(ctx, v_heads);
    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q.tensor,
        k.tensor,
        v.tensor,
        attention_mask.tensor,
        1.0F / std::sqrt(static_cast<float>(dim)),
        0.0F,
        0.0F);
    ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
    return core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[2], q.shape.dims[1], q.shape.dims[3]}),
        GGML_TYPE_F32);
}

struct Qwen3DiffusionLayerWeights {
    core::TensorValue input_norm;
    core::TensorValue q_proj;
    core::TensorValue k_proj;
    core::TensorValue v_proj;
    core::TensorValue o_proj;
    core::TensorValue q_norm;
    core::TensorValue k_norm;
    core::TensorValue post_norm;
    core::TensorValue gate_proj;
    core::TensorValue up_proj;
    core::TensorValue down_proj;
    // Packed projections (omnivoice.packed_projections): rows [q | k | v] and
    // [gate | up]. When valid, q/k/v_proj and gate/up_proj are not loaded.
    core::TensorValue qkv_proj;
    core::TensorValue gate_up_proj;
};

struct Qwen3DiffusionWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue text_embedding;
    core::TensorValue audio_embedding;
    std::vector<Qwen3DiffusionLayerWeights> layers;
    core::TensorValue norm;
    core::TensorValue audio_head;
};

Qwen3DiffusionWeights load_weights(
    const OmniVoiceAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets_ns::TensorStorageType storage_type,
    bool packed_projections) {
    validate_weight_storage_type(storage_type);
    const auto & config = assets.config;
    const auto & source = *assets.model_weights;
    Qwen3DiffusionWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "omnivoice.generator.weights",
        weight_context_bytes);
    weights.text_embedding = weights.store->load_tensor(
        source,
        "llm.embed_tokens.weight",
        storage_type,
        {config.llm.vocab_size, config.llm.hidden_size});
    weights.audio_embedding = weights.store->load_tensor(
        source,
        "audio_embeddings.weight",
        storage_type,
        {config.num_audio_codebook * config.audio_vocab_size, config.llm.hidden_size});
    const int64_t dim = head_dim(config.llm);
    weights.layers.reserve(static_cast<size_t>(config.llm.num_hidden_layers));
    for (int64_t layer = 0; layer < config.llm.num_hidden_layers; ++layer) {
        const std::string prefix = "llm.layers." + std::to_string(layer);
        Qwen3DiffusionLayerWeights w;
        w.input_norm = weights.store->load_f32_tensor(source, prefix + ".input_layernorm.weight", {config.llm.hidden_size});
        const int64_t q_out = config.llm.num_attention_heads * dim;
        const int64_t kv_out = config.llm.num_key_value_heads * dim;
        if (packed_projections) {
            // One GEMM per group: the shared input is converted to f16 once
            // instead of once per projection.
            w.qkv_proj = modules::PackedLinearWeightsBuilder({
                config.llm.hidden_size,
                {
                    {prefix + ".self_attn.q_proj.weight", std::nullopt, q_out},
                    {prefix + ".self_attn.k_proj.weight", std::nullopt, kv_out},
                    {prefix + ".self_attn.v_proj.weight", std::nullopt, kv_out},
                },
                false,
                true,
            }).build(*weights.store, source, storage_type).weight;
            w.gate_up_proj = modules::PackedLinearWeightsBuilder({
                config.llm.hidden_size,
                {
                    {prefix + ".mlp.gate_proj.weight", std::nullopt, config.llm.intermediate_size},
                    {prefix + ".mlp.up_proj.weight", std::nullopt, config.llm.intermediate_size},
                },
                false,
                true,
            }).build(*weights.store, source, storage_type).weight;
        } else {
            w.q_proj = weights.store->load_tensor(
                source,
                prefix + ".self_attn.q_proj.weight",
                storage_type,
                {q_out, config.llm.hidden_size});
            w.k_proj = weights.store->load_tensor(
                source,
                prefix + ".self_attn.k_proj.weight",
                storage_type,
                {kv_out, config.llm.hidden_size});
            w.v_proj = weights.store->load_tensor(
                source,
                prefix + ".self_attn.v_proj.weight",
                storage_type,
                {kv_out, config.llm.hidden_size});
            w.gate_proj = weights.store->load_tensor(
                source,
                prefix + ".mlp.gate_proj.weight",
                storage_type,
                {config.llm.intermediate_size, config.llm.hidden_size});
            w.up_proj = weights.store->load_tensor(
                source,
                prefix + ".mlp.up_proj.weight",
                storage_type,
                {config.llm.intermediate_size, config.llm.hidden_size});
        }
        w.o_proj = weights.store->load_tensor(
            source,
            prefix + ".self_attn.o_proj.weight",
            storage_type,
            {config.llm.hidden_size, config.llm.num_attention_heads * dim});
        w.q_norm = weights.store->load_f32_tensor(source, prefix + ".self_attn.q_norm.weight", {dim});
        w.k_norm = weights.store->load_f32_tensor(source, prefix + ".self_attn.k_norm.weight", {dim});
        w.post_norm = weights.store->load_f32_tensor(
            source,
            prefix + ".post_attention_layernorm.weight",
            {config.llm.hidden_size});
        w.down_proj = weights.store->load_tensor(
            source,
            prefix + ".mlp.down_proj.weight",
            storage_type,
            {config.llm.hidden_size, config.llm.intermediate_size});
        weights.layers.push_back(std::move(w));
    }
    weights.norm = weights.store->load_f32_tensor(source, "llm.norm.weight", {config.llm.hidden_size});
    weights.audio_head = weights.store->load_tensor(
        source,
        "audio_heads.weight",
        storage_type,
        {config.num_audio_codebook * config.audio_vocab_size, config.llm.hidden_size});
    weights.store->upload();
    return weights;
}

core::TensorValue build_embeddings(
    core::ModuleBuildContext & ctx,
    const OmniVoiceConfig & config,
    const Qwen3DiffusionWeights & weights,
    const core::TensorValue & text_ids,
    const std::array<core::TensorValue, 8> & audio_ids,
    const core::TensorValue & audio_mask,
    const core::TensorValue & text_mask) {
    auto text_embeds = modules::EmbeddingModule({config.llm.vocab_size, config.llm.hidden_size})
                           .build(ctx, text_ids, weights.text_embedding);
    core::TensorValue audio_embeds = modules::EmbeddingModule(
                                         {config.num_audio_codebook * config.audio_vocab_size, config.llm.hidden_size})
                                         .build(ctx, audio_ids[0], weights.audio_embedding);
    for (int64_t codebook = 1; codebook < config.num_audio_codebook; ++codebook) {
        auto embed = modules::EmbeddingModule(
                         {config.num_audio_codebook * config.audio_vocab_size, config.llm.hidden_size})
                         .build(ctx, audio_ids[static_cast<size_t>(codebook)], weights.audio_embedding);
        audio_embeds = modules::AddModule{}.build(ctx, audio_embeds, embed);
    }
    const auto broadcast_shape =
        core::TensorShape::from_dims({text_ids.shape.dims[0], text_ids.shape.dims[1], config.llm.hidden_size});
    auto audio_mask_expanded = modules::RepeatModule({broadcast_shape}).build(ctx, audio_mask);
    auto text_mask_expanded = modules::RepeatModule({broadcast_shape}).build(ctx, text_mask);
    auto text_part = modules::MulModule{}.build(ctx, text_embeds, text_mask_expanded);
    auto audio_part = modules::MulModule{}.build(ctx, audio_embeds, audio_mask_expanded);
    return modules::AddModule{}.build(ctx, text_part, audio_part);
}

// q, k, v: [rows, tokens, heads, dim] after RoPE. Returns the attention
// context as [rows, tokens, heads * dim].
core::TensorValue self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q,
    const core::TensorValue & k,
    const core::TensorValue & v,
    const OmniVoiceQwen3DiffusionConfig & config,
    const core::TensorValue & attention_mask,
    OmniVoiceGeneratorPerfMode perf_mode) {
    const int64_t dim = head_dim(config);
    const int64_t kv_repeats = config.num_attention_heads / config.num_key_value_heads;
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    core::TensorValue context;
    if (perf_mode == OmniVoiceGeneratorPerfMode::FlashAttention) {
        context = grouped_flash_attention(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
    } else {
        k_heads = repeat_kv_heads(ctx, k_heads, kv_repeats);
        v_heads = repeat_kv_heads(ctx, v_heads, kv_repeats);
        context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask, perf_mode);
    }
    context = ensure_contiguous(ctx, context);
    return core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[1], config.num_attention_heads * dim}));
}

// One Qwen3 decoder layer on [rows, tokens, hidden]. `attention` maps the
// RoPE'd q, k, v ([rows, tokens, heads, dim]) to the [rows, tokens,
// heads * dim] context, so the token layout decides how attention is split.
template <typename Attention>
core::TensorValue decoder_layer_with(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const Qwen3DiffusionLayerWeights & weights,
    const OmniVoiceQwen3DiffusionConfig & config,
    Attention && attention) {
    const int64_t dim = head_dim(config);
    const modules::LinearModule q_proj(
        binding::linear_config(config.hidden_size, config.num_attention_heads * dim, false));
    const modules::LinearModule k_proj(
        binding::linear_config(config.hidden_size, config.num_key_value_heads * dim, false));
    const modules::LinearModule v_proj(
        binding::linear_config(config.hidden_size, config.num_key_value_heads * dim, false));
    const modules::LinearModule o_proj(
        binding::linear_config(config.num_attention_heads * dim, config.hidden_size, false));
    const modules::RMSNormModule hidden_norm({config.hidden_size, config.rms_norm_eps, true, false});
    const modules::RMSNormModule head_norm({dim, config.rms_norm_eps, true, false});
    auto x_norm = hidden_norm.build(ctx, input, binding::norm_data(ctx, weights.input_norm));
    core::TensorValue q;
    core::TensorValue k;
    core::TensorValue v;
    if (weights.qkv_proj.valid()) {
        const int64_t q_out = config.num_attention_heads * dim;
        const int64_t kv_out = config.num_key_value_heads * dim;
        const auto qkv = modules::LinearModule(binding::linear_config(config.hidden_size, q_out + 2 * kv_out, false))
                             .build(ctx, x_norm, binding::linear_data(ctx, weights.qkv_proj));
        // The head norm reads the strided q/k views in place; v stays a view.
        const modules::RMSNormModule strided_head_norm({dim, config.rms_norm_eps, true, false, true});
        q = strided_head_norm.build(
            ctx,
            packed_heads_view(ctx, qkv, 0, config.num_attention_heads, dim),
            binding::norm_data(ctx, weights.q_norm));
        k = strided_head_norm.build(
            ctx,
            packed_heads_view(ctx, qkv, q_out, config.num_key_value_heads, dim),
            binding::norm_data(ctx, weights.k_norm));
        v = packed_heads_view(ctx, qkv, q_out + kv_out, config.num_key_value_heads, dim);
    } else {
        q = q_proj.build(ctx, x_norm, binding::linear_data(ctx, weights.q_proj));
        k = k_proj.build(ctx, x_norm, binding::linear_data(ctx, weights.k_proj));
        v = v_proj.build(ctx, x_norm, binding::linear_data(ctx, weights.v_proj));
        q = head_norm.build(ctx, reshape_heads(ctx, q, config.num_attention_heads, dim), binding::norm_data(ctx, weights.q_norm));
        k = head_norm.build(ctx, reshape_heads(ctx, k, config.num_key_value_heads, dim), binding::norm_data(ctx, weights.k_norm));
        v = reshape_heads(ctx, v, config.num_key_value_heads, dim);
    }
    q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, q, positions);
    k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, k, positions);

    auto context = attention(q, k, v);
    auto x = modules::AddModule{}.build(ctx, input, o_proj.build(ctx, context, binding::linear_data(ctx, weights.o_proj)));

    auto ff_in = hidden_norm.build(ctx, x, binding::norm_data(ctx, weights.post_norm));
    if (weights.gate_up_proj.valid()) {
        const auto gate_up =
            modules::LinearModule(binding::linear_config(config.hidden_size, 2 * config.intermediate_size, false))
                .build(ctx, ff_in, binding::linear_data(ctx, weights.gate_up_proj));
        // silu(gate) * up in one kernel; gate is the first half of each row.
        auto act_shape = gate_up.shape;
        act_shape.dims[act_shape.rank - 1] = config.intermediate_size;
        const auto act = core::wrap_tensor(ggml_swiglu(ctx.ggml, gate_up.tensor), act_shape, GGML_TYPE_F32);
        auto ff = modules::LinearModule(
                      binding::linear_config(config.intermediate_size, config.hidden_size, false))
                      .build(ctx, act, binding::linear_data(ctx, weights.down_proj));
        return modules::AddModule{}.build(ctx, x, ff);
    }
    auto gate = modules::LinearModule(
                    binding::linear_config(config.hidden_size, config.intermediate_size, false))
                    .build(ctx, ff_in, binding::linear_data(ctx, weights.gate_proj));
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule(
                  binding::linear_config(config.hidden_size, config.intermediate_size, false))
                  .build(ctx, ff_in, binding::linear_data(ctx, weights.up_proj));
    auto ff = modules::LinearModule(
                  binding::linear_config(config.intermediate_size, config.hidden_size, false))
                  .build(ctx, modules::MulModule{}.build(ctx, gate, up), binding::linear_data(ctx, weights.down_proj));
    return modules::AddModule{}.build(ctx, x, ff);
}

// Decoder layer on [rows, tokens, hidden] rows of equal length, with one
// [rows, 1, tokens, tokens] attention mask.
core::TensorValue decoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const Qwen3DiffusionLayerWeights & weights,
    const OmniVoiceQwen3DiffusionConfig & config,
    const core::TensorValue & attention_mask,
    OmniVoiceGeneratorPerfMode perf_mode = OmniVoiceGeneratorPerfMode::Standard) {
    return decoder_layer_with(ctx, input, positions, weights, config, [&](const auto & q, const auto & k, const auto & v) {
        return self_attention(ctx, q, k, v, config, attention_mask, perf_mode);
    });
}

// One attention segment of the packed layout: `rows` consecutive sequences
// of `tokens` positions each, starting at flat token `token_start`, with a
// [rows, 1, tokens, tokens] mask.
struct AttentionSegment {
    int64_t token_start = 0;
    int64_t rows = 1;
    int64_t tokens = 0;
    core::TensorValue mask;
};

// Decoder layer on the packed layout [1, flat_tokens, hidden], where every
// sequence (the conditional and the unconditional row of each request) has
// exactly its own length. Every per-token op runs once over all tokens;
// attention runs per segment of equal-length sequences, so nothing is padded.
core::TensorValue packed_decoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & flat_positions,
    const Qwen3DiffusionLayerWeights & weights,
    const OmniVoiceQwen3DiffusionConfig & config,
    const std::vector<AttentionSegment> & segments,
    OmniVoiceGeneratorPerfMode perf_mode) {
    const int64_t dim = head_dim(config);
    const int64_t flat_tokens = input.shape.dims[1];
    return decoder_layer_with(ctx, input, flat_positions, weights, config, [&](const auto & q, const auto & k, const auto & v) {
        // [1, flat_tokens, heads, dim] -> [rows, tokens, heads, dim] of one segment.
        const auto slice = [&](const core::TensorValue & value, const AttentionSegment & segment) {
            auto part = ensure_contiguous(
                ctx,
                modules::SliceModule({1, segment.token_start, segment.rows * segment.tokens}).build(ctx, value));
            return core::reshape_tensor(
                ctx,
                part,
                core::TensorShape::from_dims({segment.rows, segment.tokens, value.shape.dims[2], dim}));
        };
        // Each segment's context is written into its slot of one
        // [flat_tokens, heads * dim] buffer (seeded from q, whose every
        // element is overwritten), so the layer copies each context once.
        // ggml_set writes with the destination's row stride, so the buffer
        // is viewed as heads * dim wide rows.
        const int64_t width = config.num_attention_heads * dim;
        auto * base = ggml_reshape_2d(ctx.ggml, q.tensor, width, flat_tokens);
        ggml_tensor * context = nullptr;
        for (const auto & segment : segments) {
            auto segment_context =
                self_attention(ctx, slice(q, segment), slice(k, segment), slice(v, segment), config, segment.mask, perf_mode);
            auto * rows = ggml_reshape_2d(ctx.ggml, segment_context.tensor, width, segment.rows * segment.tokens);
            const size_t offset = static_cast<size_t>(segment.token_start * width) * sizeof(float);
            context = context == nullptr
                ? ggml_set_1d(ctx.ggml, base, rows, offset)
                : ggml_set_1d_inplace(ctx.ggml, context, rows, offset);
        }
        return core::wrap_tensor(
            ggml_reshape_3d(ctx.ggml, context, width, flat_tokens, 1),
            core::TensorShape::from_dims({1, flat_tokens, width}),
            GGML_TYPE_F32);
    });
}

class Qwen3DiffusionWeightsRuntime {
public:
    Qwen3DiffusionWeightsRuntime(
        std::shared_ptr<const OmniVoiceAssets> assets,
        core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        assets_ns::TensorStorageType storage_type,
        bool packed_projections)
        : assets_(std::move(assets)),
          backend_(execution_context.backend()),
          backend_type_(execution_context.backend_type()),
          threads_(std::max(1, execution_context.config().threads)),
          weights_(std::make_shared<Qwen3DiffusionWeights>(
              load_weights(*assets_, backend_, backend_type_, weight_context_bytes, storage_type, packed_projections))) {
        if (assets_ == nullptr) {
            throw std::runtime_error("OmniVoice generator requires assets");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("OmniVoice generator backend is not initialized");
        }
    }

    const OmniVoiceAssets & assets() const noexcept {
        return *assets_;
    }

    const Qwen3DiffusionWeights & weights() const noexcept {
        return *weights_;
    }

    ggml_backend_t backend() const noexcept {
        return backend_;
    }

    core::BackendType backend_type() const noexcept {
        return backend_type_;
    }

    int threads() const noexcept {
        return threads_;
    }

private:
    std::shared_ptr<const OmniVoiceAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const Qwen3DiffusionWeights> weights_;
};

struct PackedInputs {
    int64_t style_tokens = 0;
    int64_t text_tokens = 0;
    int64_t reference_frames = 0;
    int64_t target_frames = 0;
    std::vector<int32_t> conditional_text_ids;
    std::vector<int32_t> unconditional_text_ids;
    std::array<std::vector<int32_t>, 8> conditional_audio_ids;
    std::array<std::vector<int32_t>, 8> unconditional_audio_ids;

    int64_t total_tokens() const noexcept {
        return style_tokens + text_tokens + reference_frames + target_frames;
    }
    int64_t conditional_target_start() const noexcept {
        return style_tokens + text_tokens + reference_frames;
    }
};

// Host copies of the generator graph inputs for a batch of requests.
//
// The graph runs 2 * batch sequences: rows 0..batch-1 are the conditional
// sequences of the requests and rows batch..2*batch-1 their unconditional,
// classifier-free guidance sequences. The rows are stored back to back,
// conditional rows first, each with its own length. A request shorter than
// its rows is padded; padded positions only attend to themselves, so they
// never change the valid positions.
//
// The packed layout gives every row exactly its request's length (the
// prompt for the conditional row, the target frames for the unconditional
// one), so nothing is padded. The unpacked layout gives every row the
// longest prompt of the batch, which is a plain [2 * batch, tokens] tensor;
// for a batch of one it is exactly the two-row layout of a single request.
class BatchedGraphInputs {
public:
    // conditional_lengths[i] / unconditional_lengths[i]: row lengths of
    // request i. target_frames: frames gathered per request for the logits.
    void reset(
        const OmniVoiceAssets & assets,
        const std::vector<int64_t> & conditional_lengths,
        const std::vector<int64_t> & unconditional_lengths,
        int64_t target_frames,
        bool f16_attention) {
        if (conditional_lengths.empty() || conditional_lengths.size() != unconditional_lengths.size() ||
            target_frames <= 0) {
            throw std::runtime_error("OmniVoice generator batch capacities are invalid");
        }
        batch_ = static_cast<int64_t>(conditional_lengths.size());
        row_lengths_ = conditional_lengths;
        row_lengths_.insert(row_lengths_.end(), unconditional_lengths.begin(), unconditional_lengths.end());
        row_offsets_.assign(row_lengths_.size() + 1, 0);
        attention_offsets_.assign(row_lengths_.size() + 1, 0);
        int64_t longest_row = 0;
        for (size_t row = 0; row < row_lengths_.size(); ++row) {
            if (row_lengths_[row] <= 0) {
                throw std::runtime_error("OmniVoice generator batch capacities are invalid");
            }
            row_offsets_[row + 1] = row_offsets_[row] + row_lengths_[row];
            attention_offsets_[row + 1] = attention_offsets_[row] + row_lengths_[row] * row_lengths_[row];
            longest_row = std::max(longest_row, row_lengths_[row]);
        }
        target_frames_ = target_frames;
        codebooks_ = assets.config.num_audio_codebook;
        audio_vocab_size_ = assets.config.audio_vocab_size;
        mask_id_ = assets.config.audio_mask_id;
        f16_attention_ = f16_attention;
        const size_t token_values = static_cast<size_t>(flat_tokens());
        text_ids.assign(token_values, 0);
        for (auto & ids : audio_ids) {
            ids.assign(token_values, 0);
        }
        audio_mask.assign(token_values, 0.0F);
        text_mask.assign(token_values, 0.0F);
        const size_t attention_values = static_cast<size_t>(attention_offsets_.back());
        if (f16_attention_) {
            attention_f16.assign(attention_values, ggml_fp32_to_fp16(kMaskedAttentionBias));
            attention_f32.clear();
        } else {
            attention_f32.assign(attention_values, kMaskedAttentionBias);
            attention_f16.clear();
        }
        target_indices.assign(static_cast<size_t>(2 * batch_ * target_frames_), 0);
        guidance_scales.assign(static_cast<size_t>(batch_), 0.0F);
        positions.assign(static_cast<size_t>(longest_row), 0);
        for (int64_t i = 0; i < longest_row; ++i) {
            positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        flat_positions.assign(token_values, 0);
        for (int64_t row = 0; row < 2 * batch_; ++row) {
            for (int64_t pos = 0; pos < row_tokens(row); ++pos) {
                flat_positions[row_offset(row) + static_cast<size_t>(pos)] = static_cast<int32_t>(pos);
            }
        }
        layout_.clear();
        last_layout_.clear();
        last_guidance_scales_.clear();
        layout_dirty_ = true;
        guidance_dirty_ = true;
    }

    // Copies every request's token ids into its two rows. Masks and target
    // indices are rebuilt only when the per-request layout changed, and the
    // guidance scales only when they changed.
    void prepare(const std::vector<PackedInputs> & inputs, const std::vector<float> & scales) {
        if (static_cast<int64_t>(inputs.size()) != batch_ || static_cast<int64_t>(scales.size()) != batch_) {
            throw std::runtime_error("OmniVoice generator batch size does not match graph capacity");
        }
        layout_.assign(inputs.size(), {});
        std::fill(text_ids.begin(), text_ids.end(), static_cast<int32_t>(mask_id_));
        for (int64_t codebook = 0; codebook < codebooks_; ++codebook) {
            std::fill(
                audio_ids[static_cast<size_t>(codebook)].begin(),
                audio_ids[static_cast<size_t>(codebook)].end(),
                static_cast<int32_t>(codebook * audio_vocab_size_ + mask_id_));
        }
        for (int64_t item = 0; item < batch_; ++item) {
            const auto & input = inputs[static_cast<size_t>(item)];
            const int64_t total = input.total_tokens();
            if (total <= 0 || total > row_tokens(item) || input.target_frames <= 0 ||
                input.target_frames > target_frames_ || input.target_frames > row_tokens(batch_ + item)) {
                throw std::runtime_error("OmniVoice generator packed input shape exceeds graph capacity");
            }
            if (static_cast<int64_t>(input.conditional_text_ids.size()) != total ||
                static_cast<int64_t>(input.unconditional_text_ids.size()) != total) {
                throw std::runtime_error("OmniVoice generator text id tensor shape is invalid");
            }
            // The unconditional ids past the target frames are mask ids, the
            // same as the padding, so only the part that fits is copied.
            const int64_t unconditional_copy = std::min(total, row_tokens(batch_ + item));
            const size_t conditional_row = row_offset(item);
            const size_t unconditional_row = row_offset(batch_ + item);
            for (int64_t pos = 0; pos < total; ++pos) {
                text_ids[conditional_row + static_cast<size_t>(pos)] = input.conditional_text_ids[static_cast<size_t>(pos)];
            }
            for (int64_t pos = 0; pos < unconditional_copy; ++pos) {
                text_ids[unconditional_row + static_cast<size_t>(pos)] = input.unconditional_text_ids[static_cast<size_t>(pos)];
            }
            for (int64_t codebook = 0; codebook < codebooks_; ++codebook) {
                const auto & conditional_ids = input.conditional_audio_ids[static_cast<size_t>(codebook)];
                const auto & unconditional_ids = input.unconditional_audio_ids[static_cast<size_t>(codebook)];
                if (static_cast<int64_t>(conditional_ids.size()) != total ||
                    static_cast<int64_t>(unconditional_ids.size()) != total) {
                    throw std::runtime_error("OmniVoice generator audio id tensor shape is invalid");
                }
                auto & ids = audio_ids[static_cast<size_t>(codebook)];
                for (int64_t pos = 0; pos < total; ++pos) {
                    ids[conditional_row + static_cast<size_t>(pos)] = conditional_ids[static_cast<size_t>(pos)];
                }
                for (int64_t pos = 0; pos < unconditional_copy; ++pos) {
                    ids[unconditional_row + static_cast<size_t>(pos)] = unconditional_ids[static_cast<size_t>(pos)];
                }
            }
            auto & layout = layout_[static_cast<size_t>(item)];
            layout.conditional_total = total;
            layout.conditional_audio_start = input.style_tokens + input.text_tokens;
            layout.conditional_target_start = input.conditional_target_start();
            layout.target_frames = input.target_frames;
        }
        if (layout_ != last_layout_) {
            rebuild_masks_and_indices();
            last_layout_ = layout_;
            layout_dirty_ = true;
        }
        if (scales != last_guidance_scales_) {
            guidance_scales = scales;
            last_guidance_scales_ = scales;
            guidance_dirty_ = true;
        }
    }

    // Copies the current target tokens of every request into its two rows.
    void update_targets(const std::vector<PackedInputs> & inputs) {
        if (static_cast<int64_t>(inputs.size()) != batch_ || layout_.size() != inputs.size()) {
            throw std::runtime_error("OmniVoice generator target token update requires a prepared request");
        }
        for (int64_t item = 0; item < batch_; ++item) {
            const auto & input = inputs[static_cast<size_t>(item)];
            const auto & layout = layout_[static_cast<size_t>(item)];
            if (input.target_frames != layout.target_frames) {
                throw std::runtime_error("OmniVoice generator target token update requires a prepared request");
            }
            const size_t conditional_row = row_offset(item);
            const size_t unconditional_row = row_offset(batch_ + item);
            for (int64_t codebook = 0; codebook < codebooks_; ++codebook) {
                const auto & conditional_ids = input.conditional_audio_ids[static_cast<size_t>(codebook)];
                const auto & unconditional_ids = input.unconditional_audio_ids[static_cast<size_t>(codebook)];
                auto & ids = audio_ids[static_cast<size_t>(codebook)];
                for (int64_t frame = 0; frame < layout.target_frames; ++frame) {
                    const size_t pos = static_cast<size_t>(layout.conditional_target_start + frame);
                    ids[conditional_row + pos] = conditional_ids[pos];
                    ids[unconditional_row + static_cast<size_t>(frame)] = unconditional_ids[static_cast<size_t>(frame)];
                }
            }
        }
    }

    bool prepared() const noexcept { return !layout_.empty(); }
    bool take_layout_dirty() noexcept { return std::exchange(layout_dirty_, false); }
    bool take_guidance_dirty() noexcept { return std::exchange(guidance_dirty_, false); }
    int64_t batch() const noexcept { return batch_; }
    int64_t flat_tokens() const noexcept { return row_offsets_.back(); }
    int64_t target_frames() const noexcept { return target_frames_; }
    // Row r: conditional row of request r for r < batch, else the
    // unconditional row of request r - batch.
    int64_t row_tokens(int64_t row) const noexcept {
        return row_lengths_[static_cast<size_t>(row)];
    }
    size_t row_offset(int64_t row) const noexcept {
        return static_cast<size_t>(row_offsets_[static_cast<size_t>(row)]);
    }
    // First value of row r's [tokens, tokens] mask in attention_f32/f16.
    size_t attention_offset(int64_t row) const noexcept {
        return static_cast<size_t>(attention_offsets_[static_cast<size_t>(row)]);
    }

    // One value per token of the back-to-back rows.
    std::vector<int32_t> text_ids;
    std::array<std::vector<int32_t>, 8> audio_ids{};
    std::vector<float> audio_mask;
    std::vector<float> text_mask;
    // Per-row [tokens, tokens] attention biases, conditional rows first.
    std::vector<float> attention_f32;
    std::vector<ggml_fp16_t> attention_f16;
    // Tokens of the flattened [flat_tokens, hidden] state gathered for the
    // logits: target frames of every conditional row, then of every
    // unconditional row. Padded frames point at a valid position and are
    // ignored by the caller.
    std::vector<int32_t> target_indices;
    std::vector<float> guidance_scales;
    // 0..longest row - 1, shared by every row of the unpacked layout.
    std::vector<int32_t> positions;
    // Position of every token of the back-to-back rows (packed layout).
    std::vector<int32_t> flat_positions;

private:
    struct ItemLayout {
        int64_t conditional_total = 0;
        int64_t conditional_audio_start = 0;
        int64_t conditional_target_start = 0;
        int64_t target_frames = 0;

        bool operator==(const ItemLayout & other) const noexcept {
            return conditional_total == other.conditional_total &&
                conditional_audio_start == other.conditional_audio_start &&
                conditional_target_start == other.conditional_target_start &&
                target_frames == other.target_frames;
        }
        bool operator!=(const ItemLayout & other) const noexcept {
            return !(*this == other);
        }
    };

    void rebuild_masks_and_indices() {
        std::fill(audio_mask.begin(), audio_mask.end(), 0.0F);
        std::fill(text_mask.begin(), text_mask.end(), 1.0F);
        if (f16_attention_) {
            std::fill(attention_f16.begin(), attention_f16.end(), ggml_fp32_to_fp16(kMaskedAttentionBias));
        } else {
            std::fill(attention_f32.begin(), attention_f32.end(), kMaskedAttentionBias);
        }
        const auto write_row_attention = [&](int64_t row, int64_t valid) {
            const int64_t tokens = row_tokens(row);
            const size_t base = attention_offset(row);
            const auto write_zero = [&](int64_t q, int64_t k) {
                const size_t index = base + static_cast<size_t>(k + tokens * q);
                if (f16_attention_) {
                    attention_f16[index] = ggml_fp32_to_fp16(0.0F);
                } else {
                    attention_f32[index] = 0.0F;
                }
            };
            for (int64_t q = 0; q < valid; ++q) {
                for (int64_t k = 0; k < valid; ++k) {
                    write_zero(q, k);
                }
            }
            for (int64_t q = valid; q < tokens; ++q) {
                write_zero(q, q);
            }
        };
        for (int64_t item = 0; item < batch_; ++item) {
            const auto & layout = layout_[static_cast<size_t>(item)];
            const int64_t conditional_row = item;
            const int64_t unconditional_row = batch_ + item;
            for (int64_t pos = layout.conditional_audio_start; pos < layout.conditional_total; ++pos) {
                audio_mask[row_offset(conditional_row) + static_cast<size_t>(pos)] = 1.0F;
                text_mask[row_offset(conditional_row) + static_cast<size_t>(pos)] = 0.0F;
            }
            for (int64_t pos = 0; pos < layout.target_frames; ++pos) {
                audio_mask[row_offset(unconditional_row) + static_cast<size_t>(pos)] = 1.0F;
                text_mask[row_offset(unconditional_row) + static_cast<size_t>(pos)] = 0.0F;
            }
            write_row_attention(conditional_row, layout.conditional_total);
            write_row_attention(unconditional_row, layout.target_frames);
            for (int64_t frame = 0; frame < target_frames_; ++frame) {
                const int64_t valid_frame = frame < layout.target_frames ? frame : 0;
                target_indices[static_cast<size_t>(conditional_row * target_frames_ + frame)] = static_cast<int32_t>(
                    row_offset(conditional_row) + static_cast<size_t>(layout.conditional_target_start + valid_frame));
                target_indices[static_cast<size_t>(unconditional_row * target_frames_ + frame)] = static_cast<int32_t>(
                    row_offset(unconditional_row) + static_cast<size_t>(valid_frame));
            }
        }
    }

    int64_t batch_ = 0;
    std::vector<int64_t> row_lengths_;
    std::vector<int64_t> row_offsets_;
    std::vector<int64_t> attention_offsets_;
    int64_t target_frames_ = 0;
    int64_t codebooks_ = 0;
    int64_t audio_vocab_size_ = 0;
    int64_t mask_id_ = 0;
    bool f16_attention_ = false;
    std::vector<ItemLayout> layout_;
    std::vector<ItemLayout> last_layout_;
    std::vector<float> last_guidance_scales_;
    bool layout_dirty_ = true;
    bool guidance_dirty_ = true;
};

// Final norm, target-frame gather, audio head and classifier-free guidance
// for the decoder output of BatchedGraphInputs' rows (`flat_tokens` tokens of
// any [..., hidden] shape). Returns [batch, frames, codebooks * vocab] guided
// logits.
core::TensorValue build_guided_logits(
    core::ModuleBuildContext & ctx,
    const OmniVoiceConfig & config,
    const Qwen3DiffusionWeights & weights,
    const core::TensorValue & hidden_states,
    const core::TensorValue & target_indices,
    const core::TensorValue & guidance_scales,
    int64_t batch,
    int64_t flat_tokens,
    int64_t target_frames) {
    const int64_t rows = 2 * batch;
    auto x = modules::RMSNormModule({config.llm.hidden_size, config.llm.rms_norm_eps, true, false})
                 .build(ctx, hidden_states, binding::norm_data(ctx, weights.norm));
    x = core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, x),
        core::TensorShape::from_dims({flat_tokens, config.llm.hidden_size}));
    auto hidden = modules::EmbeddingModule({flat_tokens, config.llm.hidden_size}).build(ctx, target_indices, x);
    hidden = core::reshape_tensor(
        ctx,
        hidden,
        core::TensorShape::from_dims({rows, target_frames, config.llm.hidden_size}));
    auto logits =
        modules::LinearModule(
            binding::linear_config(
                config.llm.hidden_size,
                config.num_audio_codebook * config.audio_vocab_size,
                false))
            .build(ctx, hidden, binding::linear_data(ctx, weights.audio_head));
    auto conditional_logits = modules::SliceModule({0, 0, batch}).build(ctx, logits);
    auto unconditional_logits = modules::SliceModule({0, batch, batch}).build(ctx, logits);
    auto scale = core::reshape_tensor(ctx, guidance_scales, core::TensorShape::from_dims({batch, 1, 1}));
    auto scale_expanded = modules::RepeatModule({conditional_logits.shape}).build(ctx, scale);
    auto diff = core::wrap_tensor(
        ggml_sub(ctx.ggml, conditional_logits.tensor, unconditional_logits.tensor),
        conditional_logits.shape,
        GGML_TYPE_F32);
    auto scaled_diff = modules::MulModule{}.build(ctx, diff, scale_expanded);
    auto combined = modules::AddModule{}.build(ctx, conditional_logits, scaled_diff);
    return ensure_contiguous(ctx, combined);
}

struct BestTokenTensors {
    ggml_tensor * tokens = nullptr;
    ggml_tensor * log_probs = nullptr;
};

// Greedy scoring of guided logits on the device: for every [vocab] row, the
// best token other than `mask_bias`'s masked id and its log-probability
// under the full row (logit - logsumexp), as the CPU scorer computes it.
// f16 weights make exact ties common, and ggml_argmax does not say which
// tied index it returns, so ties are broken towards the lowest index like
// the CPU scorer: `index_ramp` holds vocab - i for token i.
BestTokenTensors build_best_tokens(
    core::ModuleBuildContext & ctx,
    ggml_tensor * guided_logits,
    ggml_tensor * mask_bias,
    ggml_tensor * index_ramp,
    int64_t vocab_size) {
    const int64_t rows = ggml_nelements(guided_logits) / vocab_size;
    auto * logits = ggml_reshape_2d(ctx.ggml, guided_logits, vocab_size, rows);
    // Gathers logits[row, index[row]] for an i32 [rows] index.
    const auto pick = [&](ggml_tensor * index) {
        auto * picked = ggml_get_rows(
            ctx.ggml,
            ggml_reshape_3d(ctx.ggml, logits, 1, vocab_size, rows),
            ggml_reshape_2d(ctx.ggml, index, 1, rows));
        return ggml_reshape_2d(ctx.ggml, picked, 1, rows);
    };
    BestTokenTensors out;
    auto * masked = ggml_add(ctx.ggml, logits, mask_bias);
    auto * masked_max = ggml_get_rows(
        ctx.ggml,
        ggml_reshape_3d(ctx.ggml, masked, 1, vocab_size, rows),
        ggml_reshape_2d(ctx.ggml, ggml_argmax(ctx.ggml, masked), 1, rows));
    masked_max = ggml_reshape_2d(ctx.ggml, masked_max, 1, rows);
    // 1 where the token reaches the row maximum (logit differences are far
    // above 1e-30), else 0; times the ramp, the lowest tied index wins.
    auto * at_max = ggml_step(
        ctx.ggml,
        ggml_scale_bias(ctx.ggml, ggml_sub(ctx.ggml, masked, masked_max), 1.0F, 1.0e-30F));
    out.tokens = ggml_argmax(ctx.ggml, ggml_mul(ctx.ggml, at_max, index_ramp));
    auto * row_max = pick(ggml_argmax(ctx.ggml, logits));
    auto * sum_exp = ggml_sum_rows(ctx.ggml, ggml_exp(ctx.ggml, ggml_sub(ctx.ggml, logits, row_max)));
    auto * log_sum_exp = ggml_add(ctx.ggml, row_max, ggml_log(ctx.ggml, sum_exp));
    out.log_probs = ggml_sub(ctx.ggml, pick(out.tokens), log_sum_exp);
    return out;
}

// What one decoding step reads back. Pointers are null for outputs that were
// not requested.
struct GraphStepOutputs {
    const std::vector<float> * logits = nullptr;          // [batch, frame capacity, codebooks * vocab]
    const std::vector<int32_t> * best_tokens = nullptr;   // [batch, frame capacity, codebooks]
    const std::vector<float> * best_log_probs = nullptr;  // [batch, frame capacity, codebooks]
};

// Sequence lengths of one request in a batch.
struct GraphItemShape {
    int64_t total_tokens = 0;
    int64_t target_frames = 0;

    bool operator==(const GraphItemShape & other) const noexcept {
        return total_tokens == other.total_tokens && target_frames == other.target_frames;
    }
};

struct GraphCapacity {
    int64_t batch = 0;
    int64_t total_tokens = 0;
    int64_t target_frames = 0;

    explicit GraphCapacity(const std::vector<GraphItemShape> & items)
        : batch(static_cast<int64_t>(items.size())) {
        for (const auto & item : items) {
            total_tokens = std::max(total_tokens, item.total_tokens);
            target_frames = std::max(target_frames, item.target_frames);
        }
        if (batch <= 0 || total_tokens <= 0 || target_frames <= 0) {
            throw std::runtime_error("OmniVoice generator total token capacity is invalid");
        }
    }
};

class Qwen3DiffusionGraph {
public:
    virtual ~Qwen3DiffusionGraph() = default;
    virtual bool matches(const Qwen3DiffusionWeightsRuntime & runtime, const std::vector<GraphItemShape> & items) const = 0;
    virtual void rebuild(const std::vector<GraphItemShape> & items) = 0;
    virtual void prepare_request(const std::vector<PackedInputs> & inputs, const std::vector<float> & guidance_scales) = 0;
    virtual void update_generated_target_tokens(const std::vector<PackedInputs> & inputs) = 0;
    // Whether compute() can return greedy best tokens scored on the device.
    virtual bool computes_best_tokens() const noexcept = 0;
    virtual GraphStepOutputs compute(bool read_logits, bool read_best_tokens, double & compute_ms, double & readback_ms) = 0;
    virtual int64_t batch_capacity() const noexcept = 0;
    virtual int64_t total_token_capacity() const noexcept = 0;
    virtual int64_t target_frame_capacity() const noexcept = 0;
    virtual double rebuild_clear_ms() const noexcept = 0;
    virtual double rebuild_build_ms() const noexcept = 0;
    virtual double rebuild_alloc_ms() const noexcept = 0;
    virtual double rebuild_init_ms() const noexcept = 0;
};

class Qwen3DiffusionForwardGraph final : public Qwen3DiffusionGraph {
public:
    Qwen3DiffusionForwardGraph(
        std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime,
        size_t graph_arena_bytes,
        const std::vector<GraphItemShape> & items,
        OmniVoiceGeneratorPerfMode perf_mode,
        bool packed_cfg,
        bool device_scoring)
        : runtime_(std::move(runtime)),
          graph_arena_bytes_(graph_arena_bytes),
          perf_mode_(perf_mode),
          packed_cfg_(packed_cfg),
          device_scoring_(device_scoring) {
        rebuild(items);
    }

    ~Qwen3DiffusionForwardGraph() {
        clear_graph();
    }

    // Packed graphs have one segment per sequence, so they fit only the exact
    // lengths they were built for; unpacked graphs pad to the longest ones.
    bool matches(const Qwen3DiffusionWeightsRuntime & runtime, const std::vector<GraphItemShape> & items) const override {
        if (runtime_.get() != &runtime) {
            return false;
        }
        if (packed_cfg_) {
            return items == items_;
        }
        const GraphCapacity capacity(items);
        return batch_capacity_ == capacity.batch &&
            total_tokens_capacity_ == capacity.total_tokens &&
            target_frame_capacity_ == capacity.target_frames;
    }

    void rebuild(const std::vector<GraphItemShape> & items) override {
        const GraphCapacity capacity(items);
        const auto clear_start = Clock::now();
        clear_graph();
        const auto clear_end = Clock::now();
        rebuild_clear_ms_ = engine::debug::elapsed_ms(clear_start, clear_end);
        items_ = items;
        batch_capacity_ = capacity.batch;
        total_tokens_capacity_ = capacity.total_tokens;
        target_frame_capacity_ = capacity.target_frames;
        const int64_t rows = 2 * batch_capacity_;
        const bool f16_attention = perf_mode_ == OmniVoiceGeneratorPerfMode::FlashAttention;
        // Packed: each row is exactly its request's prompt or target length.
        std::vector<int64_t> conditional_lengths(items.size(), total_tokens_capacity_);
        std::vector<int64_t> unconditional_lengths(items.size(), total_tokens_capacity_);
        if (packed_cfg_) {
            for (size_t item = 0; item < items.size(); ++item) {
                conditional_lengths[item] = items[item].total_tokens;
                unconditional_lengths[item] = items[item].target_frames;
            }
        }
        host_.reset(runtime_->assets(), conditional_lengths, unconditional_lengths, target_frame_capacity_, f16_attention);
        const int64_t flat_tokens = host_.flat_tokens();
        // Token-major shapes: [1, flat_tokens] packed, [rows, tokens] unpacked.
        const int64_t token_dim0 = packed_cfg_ ? 1 : rows;
        const int64_t token_dim1 = packed_cfg_ ? flat_tokens : total_tokens_capacity_;
        const auto token_ids_shape = core::TensorShape::from_dims({token_dim0, token_dim1});
        const auto token_mask_shape = core::TensorShape::from_dims({token_dim0, token_dim1, 1});

        const auto build_start = Clock::now();
        ggml_init_params input_params{16ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize OmniVoice generator input context");
        }
        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize OmniVoice generator graph context");
        }

        const auto & config = runtime_->assets().config;
        const auto & weights = runtime_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), "omnivoice.generator.forward", runtime_->backend_type()};
        core::ModuleBuildContext input_ctx{
            input_ctx_.get(),
            "omnivoice.generator.forward.inputs",
            runtime_->backend_type()};

        std::array<core::TensorValue, 8> audio_id_values = {};

        auto text_ids = core::make_tensor(input_ctx, GGML_TYPE_I32, token_ids_shape);
        text_ids_ = text_ids.tensor;
        ggml_set_input(text_ids_);
        for (int64_t codebook = 0; codebook < config.num_audio_codebook; ++codebook) {
            auto audio_ids = core::make_tensor(input_ctx, GGML_TYPE_I32, token_ids_shape);
            audio_ids_[static_cast<size_t>(codebook)] = audio_ids.tensor;
            ggml_set_input(audio_ids_[static_cast<size_t>(codebook)]);
            audio_id_values[static_cast<size_t>(codebook)] = audio_ids;
        }
        auto audio_mask = core::make_tensor(input_ctx, GGML_TYPE_F32, token_mask_shape);
        auto text_mask = core::make_tensor(input_ctx, GGML_TYPE_F32, token_mask_shape);
        audio_mask_ = audio_mask.tensor;
        text_mask_ = text_mask.tensor;
        ggml_set_input(audio_mask_);
        ggml_set_input(text_mask_);
        const int64_t position_count = packed_cfg_ ? flat_tokens : total_tokens_capacity_;
        positions_ =
            core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({position_count})).tensor;
        ggml_set_input(positions_);
        auto positions =
            core::wrap_tensor(positions_, core::TensorShape::from_dims({position_count}), GGML_TYPE_I32);
        auto target_indices =
            core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({rows * target_frame_capacity_}));
        target_indices_ = target_indices.tensor;
        ggml_set_input(target_indices_);
        // Unpacked graphs use one [rows, 1, capacity, capacity] mask. Packed
        // graphs give every attention segment its own [rows, 1, tokens, tokens]
        // mask input, so each mask starts at an offset the backend allocator
        // aligned: views into one shared buffer start at arbitrary offsets,
        // which backends such as Vulkan reject.
        const ggml_type mask_type = f16_attention ? GGML_TYPE_F16 : GGML_TYPE_F32;
        segment_masks_.clear();
        std::vector<AttentionSegment> segments;
        core::TensorValue attention_mask;
        if (packed_cfg_) {
            // Consecutive rows of equal length share one attention segment.
            for (int64_t first = 0; first < rows;) {
                const int64_t tokens = host_.row_tokens(first);
                int64_t end = first + 1;
                while (end < rows && host_.row_tokens(end) == tokens) {
                    ++end;
                }
                const int64_t segment_rows = end - first;
                auto mask = core::make_tensor(
                    input_ctx, mask_type, core::TensorShape::from_dims({segment_rows, 1, tokens, tokens}));
                ggml_set_input(mask.tensor);
                segment_masks_.push_back({
                    mask.tensor,
                    host_.attention_offset(first),
                    host_.attention_offset(end) - host_.attention_offset(first),
                });
                segments.push_back({static_cast<int64_t>(host_.row_offset(first)), segment_rows, tokens, mask});
                first = end;
            }
        } else {
            attention_mask = core::make_tensor(
                input_ctx,
                mask_type,
                core::TensorShape::from_dims({rows, 1, total_tokens_capacity_, total_tokens_capacity_}));
            attention_mask_ = attention_mask.tensor;
            ggml_set_input(attention_mask_);
        }
        auto guidance_scales =
            core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_capacity_}));
        guidance_scales_ = guidance_scales.tensor;
        ggml_set_input(guidance_scales_);
        if (device_scoring_) {
            mask_bias_ =
                core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({config.audio_vocab_size})).tensor;
            ggml_set_input(mask_bias_);
            index_ramp_ =
                core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({config.audio_vocab_size})).tensor;
            ggml_set_input(index_ramp_);
        }

        auto x = build_embeddings(ctx, config, weights, text_ids, audio_id_values, audio_mask, text_mask);
        if (packed_cfg_) {
            for (const auto & layer : weights.layers) {
                x = packed_decoder_layer(ctx, x, positions, layer, config.llm, segments, perf_mode_);
            }
        } else {
            for (const auto & layer : weights.layers) {
                x = decoder_layer(ctx, x, positions, layer, config.llm, attention_mask, perf_mode_);
            }
        }
        logits_ = build_guided_logits(
            ctx,
            config,
            weights,
            x,
            target_indices,
            guidance_scales,
            batch_capacity_,
            flat_tokens,
            target_frame_capacity_).tensor;

        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, logits_);
        if (device_scoring_) {
            const auto best = build_best_tokens(ctx, logits_, mask_bias_, index_ramp_, config.audio_vocab_size);
            best_tokens_ = best.tokens;
            best_log_probs_ = best.log_probs;
            ggml_set_output(best_tokens_);
            ggml_set_output(best_log_probs_);
            ggml_build_forward_expand(graph_, best_tokens_);
            ggml_build_forward_expand(graph_, best_log_probs_);
        }
        const auto build_end = Clock::now();
        rebuild_build_ms_ = engine::debug::elapsed_ms(build_start, build_end);

        const auto alloc_start = Clock::now();
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), runtime_->backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate OmniVoice generator input buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate OmniVoice generator graph buffer");
        }
        const auto alloc_end = Clock::now();
        rebuild_alloc_ms_ = engine::debug::elapsed_ms(alloc_start, alloc_end);

        const auto init_start = Clock::now();
        const auto & position_values = packed_cfg_ ? host_.flat_positions : host_.positions;
        ggml_backend_tensor_set(positions_, position_values.data(), 0, position_values.size() * sizeof(int32_t));
        if (device_scoring_) {
            std::vector<float> mask_bias(static_cast<size_t>(config.audio_vocab_size), 0.0F);
            mask_bias[static_cast<size_t>(config.audio_mask_id)] = std::numeric_limits<float>::lowest();
            ggml_backend_tensor_set(mask_bias_, mask_bias.data(), 0, mask_bias.size() * sizeof(float));
            std::vector<float> index_ramp(static_cast<size_t>(config.audio_vocab_size), 0.0F);
            for (int64_t token = 0; token < config.audio_vocab_size; ++token) {
                index_ramp[static_cast<size_t>(token)] = static_cast<float>(config.audio_vocab_size - token);
            }
            ggml_backend_tensor_set(index_ramp_, index_ramp.data(), 0, index_ramp.size() * sizeof(float));
        }
        logits_host_.clear();
        best_tokens_host_.clear();
        best_log_probs_host_.clear();
        const auto init_end = Clock::now();
        rebuild_init_ms_ = engine::debug::elapsed_ms(init_start, init_end);
    }

    void prepare_request(const std::vector<PackedInputs> & inputs, const std::vector<float> & guidance_scales) override {
        host_.prepare(inputs, guidance_scales);
        upload_token_ids();
        if (host_.take_layout_dirty()) {
            ggml_backend_tensor_set(audio_mask_, host_.audio_mask.data(), 0, host_.audio_mask.size() * sizeof(float));
            ggml_backend_tensor_set(text_mask_, host_.text_mask.data(), 0, host_.text_mask.size() * sizeof(float));
            const bool f16_mask = perf_mode_ == OmniVoiceGeneratorPerfMode::FlashAttention;
            const auto upload_mask = [&](ggml_tensor * tensor, size_t first_value, size_t values) {
                if (f16_mask) {
                    ggml_backend_tensor_set(
                        tensor, host_.attention_f16.data() + first_value, 0, values * sizeof(ggml_fp16_t));
                } else {
                    ggml_backend_tensor_set(tensor, host_.attention_f32.data() + first_value, 0, values * sizeof(float));
                }
            };
            if (packed_cfg_) {
                for (const auto & mask : segment_masks_) {
                    upload_mask(mask.tensor, mask.first_value, mask.values);
                }
            } else {
                upload_mask(
                    attention_mask_,
                    0,
                    f16_mask ? host_.attention_f16.size() : host_.attention_f32.size());
            }
            ggml_backend_tensor_set(
                target_indices_,
                host_.target_indices.data(),
                0,
                host_.target_indices.size() * sizeof(int32_t));
        }
        if (host_.take_guidance_dirty()) {
            ggml_backend_tensor_set(
                guidance_scales_,
                host_.guidance_scales.data(),
                0,
                host_.guidance_scales.size() * sizeof(float));
        }
    }

    void update_generated_target_tokens(const std::vector<PackedInputs> & inputs) override {
        host_.update_targets(inputs);
        for (int64_t codebook = 0; codebook < runtime_->assets().config.num_audio_codebook; ++codebook) {
            const auto & ids = host_.audio_ids[static_cast<size_t>(codebook)];
            ggml_backend_tensor_set(audio_ids_[static_cast<size_t>(codebook)], ids.data(), 0, ids.size() * sizeof(int32_t));
        }
    }

    bool computes_best_tokens() const noexcept override {
        return device_scoring_;
    }

    GraphStepOutputs compute(bool read_logits, bool read_best_tokens, double & compute_ms, double & readback_ms) override {
        if (!host_.prepared()) {
            throw std::runtime_error("OmniVoice generator compute requires a prepared request");
        }
        if (read_best_tokens && !device_scoring_) {
            throw std::runtime_error("OmniVoice generator graph was built without device scoring");
        }

        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        const auto compute_end = Clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("OmniVoice generator graph compute failed");
        }
        compute_ms += engine::debug::elapsed_ms(compute_start, compute_end);

        GraphStepOutputs out;
        const auto readback_start = Clock::now();
        const size_t row_count = static_cast<size_t>(
            batch_capacity_ * target_frame_capacity_ * runtime_->assets().config.num_audio_codebook);
        if (read_logits) {
            logits_host_.resize(row_count * static_cast<size_t>(runtime_->assets().config.audio_vocab_size));
            ggml_backend_tensor_get(logits_, logits_host_.data(), 0, logits_host_.size() * sizeof(float));
            out.logits = &logits_host_;
        }
        if (read_best_tokens) {
            best_tokens_host_.resize(row_count);
            best_log_probs_host_.resize(row_count);
            ggml_backend_tensor_get(best_tokens_, best_tokens_host_.data(), 0, row_count * sizeof(int32_t));
            ggml_backend_tensor_get(best_log_probs_, best_log_probs_host_.data(), 0, row_count * sizeof(float));
            out.best_tokens = &best_tokens_host_;
            out.best_log_probs = &best_log_probs_host_;
        }
        readback_ms += engine::debug::elapsed_ms(readback_start, Clock::now());
        return out;
    }

    int64_t batch_capacity() const noexcept override {
        return batch_capacity_;
    }
    int64_t total_token_capacity() const noexcept override {
        return total_tokens_capacity_;
    }
    int64_t target_frame_capacity() const noexcept override {
        return target_frame_capacity_;
    }
    double rebuild_clear_ms() const noexcept override { return rebuild_clear_ms_; }
    double rebuild_build_ms() const noexcept override { return rebuild_build_ms_; }
    double rebuild_alloc_ms() const noexcept override { return rebuild_alloc_ms_; }
    double rebuild_init_ms() const noexcept override { return rebuild_init_ms_; }

private:
    void upload_token_ids() {
        ggml_backend_tensor_set(text_ids_, host_.text_ids.data(), 0, host_.text_ids.size() * sizeof(int32_t));
        for (int64_t codebook = 0; codebook < runtime_->assets().config.num_audio_codebook; ++codebook) {
            const auto & ids = host_.audio_ids[static_cast<size_t>(codebook)];
            ggml_backend_tensor_set(audio_ids_[static_cast<size_t>(codebook)], ids.data(), 0, ids.size() * sizeof(int32_t));
        }
    }

    void clear_graph() {
        if (graph_ != nullptr) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
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
        logits_ = nullptr;
        best_tokens_ = nullptr;
        best_log_probs_ = nullptr;
        mask_bias_ = nullptr;
        index_ramp_ = nullptr;
        guidance_scales_ = nullptr;
        attention_mask_ = nullptr;
        segment_masks_.clear();
        target_indices_ = nullptr;
        positions_ = nullptr;
        audio_mask_ = nullptr;
        text_mask_ = nullptr;
        for (auto & tensor : audio_ids_) {
            tensor = nullptr;
        }
        text_ids_ = nullptr;
        input_ctx_.reset();
        ctx_.reset();
    }

    std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime_;
    size_t graph_arena_bytes_ = 0;
    OmniVoiceGeneratorPerfMode perf_mode_ = OmniVoiceGeneratorPerfMode::Standard;
    bool packed_cfg_ = true;
    bool device_scoring_ = true;
    std::vector<GraphItemShape> items_;
    int64_t batch_capacity_ = 0;
    int64_t total_tokens_capacity_ = 0;
    int64_t target_frame_capacity_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    ggml_tensor * text_ids_ = nullptr;
    std::array<ggml_tensor *, 8> audio_ids_{};
    ggml_tensor * audio_mask_ = nullptr;
    ggml_tensor * text_mask_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * target_indices_ = nullptr;
    // Unpacked layout only; the packed layout uses segment_masks_.
    ggml_tensor * attention_mask_ = nullptr;
    // One mask input per attention segment of the packed layout, with the
    // range of host mask values it receives.
    struct SegmentMask {
        ggml_tensor * tensor = nullptr;
        size_t first_value = 0;
        size_t values = 0;
    };
    std::vector<SegmentMask> segment_masks_;
    ggml_tensor * guidance_scales_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    ggml_tensor * mask_bias_ = nullptr;
    ggml_tensor * index_ramp_ = nullptr;
    ggml_tensor * best_tokens_ = nullptr;
    ggml_tensor * best_log_probs_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    BatchedGraphInputs host_;
    std::vector<float> logits_host_;
    std::vector<int32_t> best_tokens_host_;
    std::vector<float> best_log_probs_host_;
    double rebuild_clear_ms_ = 0.0;
    double rebuild_build_ms_ = 0.0;
    double rebuild_alloc_ms_ = 0.0;
    double rebuild_init_ms_ = 0.0;
};

#include "generator_layerwise.inc"

std::pair<int32_t, float> argmax_with_value_excluding_mask(const float * values, int64_t count, int64_t mask_id) {
    int32_t best_index = -1;
    float best_value = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < count; ++i) {
        if (i == mask_id) {
            continue;
        }
        const float value = values[static_cast<size_t>(i)];
        if (value > best_value) {
            best_value = value;
            best_index = static_cast<int32_t>(i);
        }
    }
    if (best_index < 0) {
        throw std::runtime_error("OmniVoice generator could not choose a token after masking");
    }
    return {best_index, best_value};
}

std::pair<int32_t, float> best_log_prob_excluding_mask(const float * logits, int64_t count, int64_t mask_id) {
    if (count <= 0) {
        throw std::runtime_error("OmniVoice generator best_log_prob requires positive vocabulary size");
    }
    int32_t best_index = -1;
    float best_logit = -std::numeric_limits<float>::infinity();
    float max_logit = logits[0];
    for (int64_t i = 0; i < count; ++i) {
        const float value = logits[static_cast<size_t>(i)];
        max_logit = std::max(max_logit, value);
        if (i == mask_id) {
            continue;
        }
        if (value > best_logit) {
            best_logit = value;
            best_index = static_cast<int32_t>(i);
        }
    }
    if (best_index < 0) {
        throw std::runtime_error("OmniVoice generator could not choose a token after masking");
    }
    double sum = 0.0;
    for (int64_t i = 0; i < count; ++i) {
        sum += std::exp(static_cast<double>(logits[static_cast<size_t>(i)] - max_logit));
    }
    const float logsum = max_logit + static_cast<float>(std::log(sum));
    return {best_index, best_logit - logsum};
}

void log_softmax_into(const float * values, int64_t count, std::vector<float> & out) {
    if (count <= 0) {
        throw std::runtime_error("OmniVoice generator log_softmax requires positive vocabulary size");
    }
    float max_value = values[0];
    for (int64_t i = 1; i < count; ++i) {
        max_value = std::max(max_value, values[i]);
    }
    double sum = 0.0;
    for (int64_t i = 0; i < count; ++i) {
        sum += std::exp(static_cast<double>(values[i] - max_value));
    }
    const float logsum = max_value + static_cast<float>(std::log(sum));
    out.resize(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        out[static_cast<size_t>(i)] = values[i] - logsum;
    }
}

float uniform_open01(std::mt19937 & rng) {
    static std::uniform_real_distribution<float> dist(1.0e-6F, 1.0F - 1.0e-6F);
    return dist(rng);
}

float gumbel_sample_scalar(float logit, float temperature, std::mt19937 & rng) {
    const float scaled = logit / temperature;
    const float u = uniform_open01(rng);
    const float noise = -std::log(-std::log(u + 1.0e-10F) + 1.0e-10F);
    return scaled + noise;
}

int32_t sample_class(
    const float * log_probs,
    int64_t vocab_size,
    float temperature,
    int64_t mask_id,
    std::mt19937 & rng) {
    const int64_t k = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(0.1 * static_cast<double>(vocab_size))));
    std::vector<int32_t> indices(static_cast<size_t>(vocab_size));
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(),
        indices.begin() + k,
        indices.end(),
        [&](int32_t lhs, int32_t rhs) {
            return log_probs[static_cast<size_t>(lhs)] > log_probs[static_cast<size_t>(rhs)];
        });
    float best_score = -std::numeric_limits<float>::infinity();
    int32_t best_index = 0;
    for (int64_t i = 0; i < k; ++i) {
        const int32_t token = indices[static_cast<size_t>(i)];
        if (token == mask_id) {
            continue;
        }
        const float score = gumbel_sample_scalar(log_probs[static_cast<size_t>(token)], temperature, rng);
        if (score > best_score) {
            best_score = score;
            best_index = token;
        }
    }
    return best_index;
}

std::vector<double> time_steps(int64_t num_inference_steps, float t_shift) {
    if (num_inference_steps <= 0) {
        throw std::runtime_error("OmniVoice generator num_inference_steps must be positive");
    }
    std::vector<double> out(static_cast<size_t>(num_inference_steps + 1), 0.0);
    for (int64_t i = 0; i <= num_inference_steps; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(num_inference_steps);
        out[static_cast<size_t>(i)] = static_cast<double>(t_shift) * t /
            (1.0 + (static_cast<double>(t_shift) - 1.0) * t);
    }
    return out;
}

std::vector<int64_t> make_schedule(int64_t total_masked_tokens, int64_t num_inference_steps, float t_shift) {
    const auto steps = time_steps(num_inference_steps, t_shift);
    std::vector<int64_t> schedule(static_cast<size_t>(num_inference_steps), 0);
    int64_t remaining = total_masked_tokens;
    for (int64_t step = 0; step < num_inference_steps; ++step) {
        int64_t num = 0;
        if (step == num_inference_steps - 1) {
            num = remaining;
        } else {
            const double fraction = steps[static_cast<size_t>(step + 1)] - steps[static_cast<size_t>(step)];
            num = std::min<int64_t>(
                static_cast<int64_t>(std::ceil(static_cast<double>(total_masked_tokens) * fraction)),
                remaining);
        }
        schedule[static_cast<size_t>(step)] = num;
        remaining -= num;
    }
    return schedule;
}

void validate_prompt_token_ranges(const OmniVoiceAssets & assets, const OmniVoicePrompt & prompt) {
    const int32_t text_vocab_limit = static_cast<int32_t>(assets.config.llm.vocab_size);
    for (size_t i = 0; i < prompt.style_token_ids.size(); ++i) {
        const int32_t token = prompt.style_token_ids[i];
        if (token < 0 || token >= text_vocab_limit) {
            throw std::runtime_error(
                "OmniVoice style token id out of range at index " + std::to_string(i) + ": " + std::to_string(token) +
                " vs llm vocab " + std::to_string(text_vocab_limit));
        }
    }
    for (size_t i = 0; i < prompt.text_token_ids.size(); ++i) {
        const int32_t token = prompt.text_token_ids[i];
        if (token < 0 || token >= text_vocab_limit) {
            throw std::runtime_error(
                "OmniVoice text token id out of range at index " + std::to_string(i) + ": " + std::to_string(token) +
                " vs llm vocab " + std::to_string(text_vocab_limit));
        }
    }
    if (!prompt.reference_audio_tokens.has_value()) {
        return;
    }
    const int32_t audio_vocab_limit = static_cast<int32_t>(assets.config.audio_vocab_size);
    const auto & audio = *prompt.reference_audio_tokens;
    for (size_t i = 0; i < audio.token_ids.size(); ++i) {
        const int32_t token = audio.token_ids[i];
        if (token < 0 || token >= audio_vocab_limit) {
            throw std::runtime_error(
                "OmniVoice reference audio token id out of range at index " + std::to_string(i) + ": " +
                std::to_string(token) + " vs audio vocab " + std::to_string(audio_vocab_limit));
        }
    }
}

PackedInputs pack_initial_inputs(const OmniVoiceAssets & assets, const OmniVoicePrompt & prompt) {
    const int64_t codebooks = checked_positive(assets.config.num_audio_codebook, "num_audio_codebook");
    if (codebooks != 8) {
        throw std::runtime_error("OmniVoice native generator currently expects exactly 8 codebooks");
    }
    PackedInputs packed;
    packed.style_tokens = static_cast<int64_t>(prompt.style_token_ids.size());
    packed.text_tokens = static_cast<int64_t>(prompt.text_token_ids.size());
    packed.reference_frames = 0;
    if (prompt.reference_audio_tokens.has_value()) {
        packed.reference_frames = prompt.reference_audio_tokens->frames;
        if (prompt.reference_audio_tokens->codebooks != codebooks) {
            throw std::runtime_error("OmniVoice reference audio token codebook count does not match generator");
        }
    }
    packed.target_frames = checked_positive(prompt.target_audio_tokens, "target_audio_tokens");
    const int64_t total = packed.style_tokens + packed.text_tokens + packed.reference_frames + packed.target_frames;
    packed.conditional_text_ids.assign(static_cast<size_t>(total), static_cast<int32_t>(assets.config.audio_mask_id));
    packed.unconditional_text_ids.assign(static_cast<size_t>(total), static_cast<int32_t>(assets.config.audio_mask_id));
    for (auto & ids : packed.conditional_audio_ids) {
        ids.assign(static_cast<size_t>(total), static_cast<int32_t>(assets.config.audio_mask_id));
    }
    for (auto & ids : packed.unconditional_audio_ids) {
        ids.assign(static_cast<size_t>(total), static_cast<int32_t>(assets.config.audio_mask_id));
    }

    int64_t pos = 0;
    for (int32_t token : prompt.style_token_ids) {
        packed.conditional_text_ids[static_cast<size_t>(pos)] = token;
        ++pos;
    }
    for (int32_t token : prompt.text_token_ids) {
        packed.conditional_text_ids[static_cast<size_t>(pos)] = token;
        ++pos;
    }
    if (prompt.reference_audio_tokens.has_value()) {
        for (int64_t frame = 0; frame < packed.reference_frames; ++frame) {
            packed.conditional_text_ids[static_cast<size_t>(pos + frame)] =
                prompt.reference_audio_tokens->token_ids[static_cast<size_t>(frame * codebooks)];
            for (int64_t codebook = 0; codebook < codebooks; ++codebook) {
                packed.conditional_audio_ids[static_cast<size_t>(codebook)][static_cast<size_t>(pos + frame)] =
                    prompt.reference_audio_tokens->token_ids[static_cast<size_t>(frame * codebooks + codebook)];
            }
        }
        pos += packed.reference_frames;
    }
    for (int64_t frame = 0; frame < packed.target_frames; ++frame) {
        packed.conditional_text_ids[static_cast<size_t>(pos + frame)] = static_cast<int32_t>(assets.config.audio_mask_id);
        for (int64_t codebook = 0; codebook < codebooks; ++codebook) {
            packed.conditional_audio_ids[static_cast<size_t>(codebook)][static_cast<size_t>(pos + frame)] =
                static_cast<int32_t>(assets.config.audio_mask_id);
        }
    }

    for (int64_t frame = 0; frame < packed.target_frames; ++frame) {
        packed.unconditional_text_ids[static_cast<size_t>(frame)] = static_cast<int32_t>(assets.config.audio_mask_id);
        for (int64_t codebook = 0; codebook < codebooks; ++codebook) {
            packed.unconditional_audio_ids[static_cast<size_t>(codebook)][static_cast<size_t>(frame)] =
                static_cast<int32_t>(assets.config.audio_mask_id);
        }
    }

    const int32_t vocab_stride = static_cast<int32_t>(assets.config.audio_vocab_size);
    for (int64_t codebook = 0; codebook < codebooks; ++codebook) {
        const int32_t offset = static_cast<int32_t>(codebook) * vocab_stride;
        auto & cond_ids = packed.conditional_audio_ids[static_cast<size_t>(codebook)];
        for (int32_t & value : cond_ids) {
            value += offset;
        }
        auto & uncond_ids = packed.unconditional_audio_ids[static_cast<size_t>(codebook)];
        for (int32_t & value : uncond_ids) {
            value += offset;
        }
    }
    return packed;
}

void update_generated_tokens(
    PackedInputs & packed,
    const OmniVoiceAssets & assets,
    const std::vector<int32_t> & generated_tokens) {
    const int64_t codebooks = assets.config.num_audio_codebook;
    const int64_t target_frames = packed.target_frames;
    const int64_t cond_start = packed.style_tokens + packed.text_tokens + packed.reference_frames;
    const int32_t vocab_stride = static_cast<int32_t>(assets.config.audio_vocab_size);
    for (int64_t codebook = 0; codebook < codebooks; ++codebook) {
        const int32_t offset = static_cast<int32_t>(codebook) * vocab_stride;
        for (int64_t frame = 0; frame < target_frames; ++frame) {
            const int32_t token = generated_tokens[static_cast<size_t>(codebook * target_frames + frame)];
            packed.conditional_audio_ids[static_cast<size_t>(codebook)][static_cast<size_t>(cond_start + frame)] =
                token + offset;
            packed.unconditional_audio_ids[static_cast<size_t>(codebook)][static_cast<size_t>(frame)] =
                token + offset;
        }
    }
}

}  // namespace

struct OmniVoiceQwen3DiffusionRuntime::Impl {
    std::shared_ptr<const OmniVoiceAssets> assets;
    size_t graph_arena_bytes = 0;
    std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime;
    std::unique_ptr<Qwen3DiffusionGraph> forward_graph;
    bool mem_saver = false;
    OmniVoiceGeneratorPerfMode perf_mode = OmniVoiceGeneratorPerfMode::Standard;
    bool packed_cfg = true;
    bool device_scoring = true;
    OmniVoiceQwen3DiffusionRuntimeStats last_stats = {};
    std::mt19937 rng{std::random_device{}()};
};

OmniVoiceQwen3DiffusionRuntime::OmniVoiceQwen3DiffusionRuntime(
    std::shared_ptr<const OmniVoiceAssets> assets,
    core::ExecutionContext & execution_context,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes,
    size_t weight_context_bytes,
    assets_ns::TensorStorageType weight_storage_type,
    bool mem_saver,
    OmniVoiceGeneratorPerfMode perf_mode,
    bool packed_cfg,
    bool device_scoring,
    bool packed_projections)
    : impl_(std::make_unique<Impl>()) {
    if (assets == nullptr) {
        throw std::runtime_error("OmniVoice generator requires assets");
    }
    if (execution_context.backend() == nullptr) {
        throw std::runtime_error("OmniVoice generator backend is not initialized");
    }
    impl_->assets = std::move(assets);
    impl_->graph_arena_bytes = std::max(prefill_graph_arena_bytes, decode_graph_arena_bytes);
    impl_->runtime = std::make_shared<Qwen3DiffusionWeightsRuntime>(
        impl_->assets,
        execution_context,
        weight_context_bytes,
        weight_storage_type,
        packed_projections);
    impl_->mem_saver = mem_saver;
    impl_->perf_mode = perf_mode;
    impl_->packed_cfg = packed_cfg;
    impl_->device_scoring = device_scoring;
}

OmniVoiceQwen3DiffusionRuntime::~OmniVoiceQwen3DiffusionRuntime() = default;

const OmniVoiceQwen3DiffusionRuntimeStats & OmniVoiceQwen3DiffusionRuntime::last_stats() const noexcept {
    return impl_->last_stats;
}

void OmniVoiceQwen3DiffusionRuntime::seed_rng(uint32_t seed) {
    impl_->rng.seed(seed);
}

void OmniVoiceQwen3DiffusionRuntime::release_runtime_graphs() {
    impl_->forward_graph.reset();
}

std::mt19937 OmniVoiceQwen3DiffusionRuntime::make_request_rng(const std::optional<uint32_t> & seed) {
    if (seed.has_value()) {
        return std::mt19937(*seed);
    }
    return std::mt19937(static_cast<std::mt19937::result_type>(impl_->rng()));
}

OmniVoiceGeneratedAudioTokens OmniVoiceQwen3DiffusionRuntime::generate(
    const OmniVoicePrompt & prompt,
    const OmniVoiceGenerationOptions & options) {
    auto results = generate_items({prompt}, {options}, {&impl_->rng});
    return std::move(results.front());
}

std::vector<OmniVoiceGeneratedAudioTokens> OmniVoiceQwen3DiffusionRuntime::generate_batch(
    const std::vector<OmniVoicePrompt> & prompts,
    const std::vector<OmniVoiceGenerationOptions> & options,
    std::vector<std::mt19937> & rngs) {
    if (rngs.size() != prompts.size()) {
        throw std::runtime_error("OmniVoice generator batch needs one RNG per prompt");
    }
    // Requests of equal length next to each other share attention segments
    // (and graph shapes repeat more often), so run them sorted by length.
    const auto length_key = [&](size_t index) {
        const auto & prompt = prompts[index];
        const int64_t reference_frames =
            prompt.reference_audio_tokens.has_value() ? prompt.reference_audio_tokens->frames : 0;
        return std::make_pair(
            static_cast<int64_t>(prompt.style_token_ids.size() + prompt.text_token_ids.size()) + reference_frames +
                prompt.target_audio_tokens,
            prompt.target_audio_tokens);
    };
    std::vector<size_t> order(prompts.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        return length_key(lhs) < length_key(rhs);
    });
    std::vector<OmniVoicePrompt> sorted_prompts;
    std::vector<OmniVoiceGenerationOptions> sorted_options;
    std::vector<std::mt19937 *> sorted_rngs;
    sorted_prompts.reserve(order.size());
    sorted_options.reserve(order.size());
    sorted_rngs.reserve(order.size());
    for (const size_t index : order) {
        sorted_prompts.push_back(prompts[index]);
        sorted_options.push_back(options.at(index));
        sorted_rngs.push_back(&rngs[index]);
    }
    auto sorted_results = generate_items(sorted_prompts, sorted_options, sorted_rngs);
    std::vector<OmniVoiceGeneratedAudioTokens> results(order.size());
    for (size_t slot = 0; slot < order.size(); ++slot) {
        results[order[slot]] = std::move(sorted_results[slot]);
    }
    return results;
}

std::vector<OmniVoiceGeneratedAudioTokens> OmniVoiceQwen3DiffusionRuntime::generate_items(
    const std::vector<OmniVoicePrompt> & prompts,
    const std::vector<OmniVoiceGenerationOptions> & options,
    const std::vector<std::mt19937 *> & rngs) {
    const size_t batch = prompts.size();
    if (batch == 0) {
        throw std::runtime_error("OmniVoice generator batch is empty");
    }
    if (options.size() != batch || rngs.size() != batch) {
        throw std::runtime_error("OmniVoice generator batch needs options and an RNG per prompt");
    }
    const auto & assets = *impl_->assets;
    const int64_t codebooks = assets.config.num_audio_codebook;
    const int64_t vocab_size = assets.config.audio_vocab_size;
    const int64_t mask_id = assets.config.audio_mask_id;

    // Per-request decoding state. Each request keeps its own schedule, RNG and
    // accepted tokens; only the forward pass is shared.
    struct ItemState {
        std::vector<int64_t> schedule;
        std::vector<int32_t> accepted;
        std::vector<size_t> active_indices;
        int64_t executed_steps = 0;
    };
    std::vector<PackedInputs> inputs(batch);
    std::vector<ItemState> items(batch);
    std::vector<float> guidance_scales(batch, 0.0F);
    int64_t max_steps = 0;
    for (size_t item = 0; item < batch; ++item) {
        const auto & prompt = prompts[item];
        if (prompt.style_token_ids.empty()) {
            throw std::runtime_error("OmniVoice generator requires non-empty style tokens");
        }
        if (prompt.text_token_ids.empty()) {
            throw std::runtime_error("OmniVoice generator requires non-empty text tokens");
        }
        if (prompt.reference_audio_tokens.has_value() &&
            prompt.reference_audio_tokens->codebooks != codebooks) {
            throw std::runtime_error("OmniVoice generator reference audio token codebooks do not match model configuration");
        }
        if (rngs[item] == nullptr) {
            throw std::runtime_error("OmniVoice generator batch RNG is null");
        }
        validate_prompt_token_ranges(assets, prompt);
        inputs[item] = pack_initial_inputs(assets, prompt);
        auto & state = items[item];
        const int64_t target_frames = inputs[item].target_frames;
        state.schedule = make_schedule(
            codebooks * target_frames,
            options[item].num_inference_steps,
            options[item].t_shift);
        state.accepted.assign(static_cast<size_t>(codebooks * target_frames), static_cast<int32_t>(mask_id));
        state.active_indices.resize(static_cast<size_t>(codebooks * target_frames));
        std::iota(state.active_indices.begin(), state.active_indices.end(), 0);
        update_generated_tokens(inputs[item], assets, state.accepted);
        guidance_scales[item] = options[item].guidance_scale;
        max_steps = std::max(max_steps, options[item].num_inference_steps);
    }

    impl_->last_stats.graph_rebuilt = false;
    impl_->last_stats.rebuild_ms = 0.0;
    impl_->last_stats.rebuild_clear_ms = 0.0;
    impl_->last_stats.rebuild_build_ms = 0.0;
    impl_->last_stats.rebuild_alloc_ms = 0.0;
    impl_->last_stats.rebuild_init_ms = 0.0;
    std::vector<GraphItemShape> shapes(batch);
    for (size_t item = 0; item < batch; ++item) {
        shapes[item] = {inputs[item].total_tokens(), inputs[item].target_frames};
    }
    const bool needs_rebuild =
        impl_->forward_graph == nullptr || !impl_->forward_graph->matches(*impl_->runtime, shapes);
    if (needs_rebuild) {
        const auto rebuild_start = Clock::now();
        if (impl_->forward_graph == nullptr) {
            if (impl_->mem_saver) {
                impl_->forward_graph = std::make_unique<Qwen3DiffusionLayerwiseGraph>(
                    impl_->runtime,
                    impl_->graph_arena_bytes,
                    shapes);
            } else {
                impl_->forward_graph = std::make_unique<Qwen3DiffusionForwardGraph>(
                    impl_->runtime,
                    impl_->graph_arena_bytes,
                    shapes,
                    impl_->perf_mode,
                    impl_->packed_cfg,
                    impl_->device_scoring);
            }
        } else {
            impl_->forward_graph->rebuild(shapes);
        }
        const auto rebuild_end = Clock::now();
        impl_->last_stats.graph_rebuilt = true;
        impl_->last_stats.rebuild_ms = engine::debug::elapsed_ms(rebuild_start, rebuild_end);
        impl_->last_stats.rebuild_clear_ms = impl_->forward_graph->rebuild_clear_ms();
        impl_->last_stats.rebuild_build_ms = impl_->forward_graph->rebuild_build_ms();
        impl_->last_stats.rebuild_alloc_ms = impl_->forward_graph->rebuild_alloc_ms();
        impl_->last_stats.rebuild_init_ms = impl_->forward_graph->rebuild_init_ms();
    }
    impl_->last_stats.batch_capacity = impl_->forward_graph->batch_capacity();
    impl_->last_stats.total_token_capacity = impl_->forward_graph->total_token_capacity();
    impl_->last_stats.target_frame_capacity = impl_->forward_graph->target_frame_capacity();
    const int64_t frame_capacity = impl_->forward_graph->target_frame_capacity();
    const size_t item_logits_stride = static_cast<size_t>(frame_capacity * codebooks * vocab_size);

    double upload_ms = 0.0;
    double compute_ms = 0.0;
    double readback_ms = 0.0;
    double scoring_ms = 0.0;
    double update_ms = 0.0;
    const auto prepare_upload_start = Clock::now();
    impl_->forward_graph->prepare_request(inputs, guidance_scales);
    upload_ms += engine::debug::elapsed_ms(prepare_upload_start, Clock::now());

    struct Candidate {
        float score = -std::numeric_limits<float>::infinity();
        int32_t predicted = 0;
        size_t index = 0;
    };
    struct GreedySlot {
        size_t item = 0;
        size_t slot = 0;
    };
    std::vector<int64_t> fill_counts(batch, 0);
    std::vector<std::vector<Candidate>> candidates(batch);
    std::vector<GreedySlot> greedy_slots;
    for (int64_t step = 0; step < max_steps; ++step) {
        bool any_fill = false;
        for (size_t item = 0; item < batch; ++item) {
            const auto & state = items[item];
            fill_counts[item] =
                step < options[item].num_inference_steps && !state.active_indices.empty()
                ? state.schedule[static_cast<size_t>(step)]
                : 0;
            any_fill = any_fill || fill_counts[item] > 0;
        }
        if (!any_fill) {
            continue;
        }

        // Greedy-class requests use the device scores when the graph has
        // them; sampled-class requests (and the layerwise graph) need the
        // full logits on the host.
        bool any_greedy = false;
        bool any_sampled = false;
        for (size_t item = 0; item < batch; ++item) {
            if (fill_counts[item] > 0) {
                any_greedy = any_greedy || options[item].class_temperature == 0.0F;
                any_sampled = any_sampled || options[item].class_temperature != 0.0F;
            }
        }
        const bool device_scores = any_greedy && impl_->forward_graph->computes_best_tokens();
        const auto outputs = impl_->forward_graph->compute(
            any_sampled || (any_greedy && !device_scores),
            device_scores,
            compute_ms,
            readback_ms);
        const auto scoring_start = Clock::now();

        // Greedy-class requests: score every active slot of every request in
        // one parallel pass. Each candidate lands at its slot position, so a
        // request sees its candidates in the same order as when run alone.
        greedy_slots.clear();
        for (size_t item = 0; item < batch; ++item) {
            candidates[item].clear();
            if (fill_counts[item] <= 0 || options[item].class_temperature != 0.0F) {
                continue;
            }
            candidates[item].resize(items[item].active_indices.size());
            for (size_t slot = 0; slot < items[item].active_indices.size(); ++slot) {
                greedy_slots.push_back({item, slot});
            }
        }
        if (!greedy_slots.empty()) {
            const size_t slot_count = greedy_slots.size();
            const unsigned desired_threads = static_cast<unsigned>(std::max<int>(1, impl_->runtime->threads()));
            // Device scores are a lookup, so they need no worker threads.
            const unsigned worker_count = device_scores
                ? 1U
                : std::min<unsigned>(desired_threads, static_cast<unsigned>(std::max<size_t>(1, slot_count)));
            auto score_range = [&](size_t begin_index, size_t end_index) {
                for (size_t index = begin_index; index < end_index; ++index) {
                    const auto [item, slot] = greedy_slots[index];
                    const auto & state = items[item];
                    const int64_t target_frames = inputs[item].target_frames;
                    const size_t flat_index = state.active_indices[slot];
                    const int64_t codebook = static_cast<int64_t>(flat_index / static_cast<size_t>(target_frames));
                    const int64_t frame = static_cast<int64_t>(flat_index % static_cast<size_t>(target_frames));
                    int32_t best_token = 0;
                    float best_score = 0.0F;
                    if (device_scores) {
                        const size_t row = (item * static_cast<size_t>(frame_capacity) + static_cast<size_t>(frame)) *
                                static_cast<size_t>(codebooks) +
                            static_cast<size_t>(codebook);
                        best_token = (*outputs.best_tokens)[row];
                        best_score = (*outputs.best_log_probs)[row];
                    } else {
                        const size_t offset = item * item_logits_stride +
                            static_cast<size_t>(frame * codebooks * vocab_size + codebook * vocab_size);
                        const float * combined_logits = outputs.logits->data() + static_cast<std::ptrdiff_t>(offset);
                        std::tie(best_token, best_score) = best_log_prob_excluding_mask(combined_logits, vocab_size, mask_id);
                    }
                    Candidate candidate;
                    candidate.score = best_score - static_cast<float>(codebook) * options[item].layer_penalty_factor;
                    candidate.predicted = best_token;
                    candidate.index = flat_index;
                    candidates[item][slot] = candidate;
                }
            };
            const size_t block = (slot_count + worker_count - 1) / worker_count;
            std::vector<std::thread> workers;
            workers.reserve(worker_count > 0 ? worker_count - 1 : 0);
            size_t begin = 0;
            for (unsigned worker = 1; worker < worker_count; ++worker) {
                const size_t end = std::min(slot_count, begin + block);
                workers.emplace_back(score_range, begin, end);
                begin = end;
            }
            score_range(begin, slot_count);
            for (auto & worker : workers) {
                worker.join();
            }
            for (size_t item = 0; item < batch; ++item) {
                if (fill_counts[item] <= 0 || options[item].class_temperature != 0.0F) {
                    continue;
                }
                // Keep the candidate order of the single-request scorer, which
                // split the slots into per-thread blocks and listed the last
                // block first. The position noise below is drawn in this
                // order, so it decides the result for a given seed.
                auto & item_candidates = candidates[item];
                const size_t item_slots = item_candidates.size();
                const unsigned item_workers = std::min<unsigned>(
                    desired_threads,
                    static_cast<unsigned>(std::max<size_t>(1, item_slots)));
                const size_t item_block = (item_slots + item_workers - 1) / item_workers;
                size_t last_block_begin = 0;
                for (unsigned worker = 1; worker < item_workers; ++worker) {
                    last_block_begin = std::min(item_slots, last_block_begin + item_block);
                }
                std::rotate(
                    item_candidates.begin(),
                    item_candidates.begin() + static_cast<std::ptrdiff_t>(last_block_begin),
                    item_candidates.end());
                if (options[item].position_temperature <= 0.0F) {
                    continue;
                }
                for (auto & candidate : item_candidates) {
                    candidate.score =
                        gumbel_sample_scalar(candidate.score, options[item].position_temperature, *rngs[item]);
                }
            }
        }

        // Sampled-class requests draw from their RNG per slot, so they are
        // scored one request at a time.
        for (size_t item = 0; item < batch; ++item) {
            if (fill_counts[item] <= 0 || options[item].class_temperature == 0.0F) {
                continue;
            }
            const auto & state = items[item];
            const auto & option = options[item];
            const int64_t target_frames = inputs[item].target_frames;
            std::vector<float> combined_log_probs;
            candidates[item].reserve(state.active_indices.size());
            for (const size_t token_index : state.active_indices) {
                const int64_t codebook = static_cast<int64_t>(token_index / static_cast<size_t>(target_frames));
                const int64_t frame = static_cast<int64_t>(token_index % static_cast<size_t>(target_frames));
                const size_t offset = item * item_logits_stride +
                    static_cast<size_t>(frame * codebooks * vocab_size + codebook * vocab_size);
                const float * combined_logits = outputs.logits->data() + static_cast<std::ptrdiff_t>(offset);
                log_softmax_into(combined_logits, vocab_size, combined_log_probs);
                auto [best_token, best_score] =
                    argmax_with_value_excluding_mask(combined_log_probs.data(), vocab_size, mask_id);
                Candidate candidate;
                candidate.index = token_index;
                candidate.predicted =
                    sample_class(combined_log_probs.data(), vocab_size, option.class_temperature, mask_id, *rngs[item]);
                candidate.score = best_score - static_cast<float>(codebook) * option.layer_penalty_factor;
                if (option.position_temperature > 0.0F) {
                    candidate.score = gumbel_sample_scalar(candidate.score, option.position_temperature, *rngs[item]);
                }
                candidates[item].push_back(candidate);
            }
        }

        const auto candidate_order = [](const Candidate & lhs, const Candidate & rhs) {
            if (lhs.score != rhs.score) {
                return lhs.score > rhs.score;
            }
            return lhs.index < rhs.index;
        };
        for (size_t item = 0; item < batch; ++item) {
            auto & item_candidates = candidates[item];
            if (fill_counts[item] <= 0 || item_candidates.empty()) {
                continue;
            }
            ++items[item].executed_steps;
            const int64_t actual_fill = std::min<int64_t>(fill_counts[item], static_cast<int64_t>(item_candidates.size()));
            if (actual_fill < static_cast<int64_t>(item_candidates.size())) {
                auto top_end = item_candidates.begin() + actual_fill;
                std::nth_element(item_candidates.begin(), top_end, item_candidates.end(), candidate_order);
                std::sort(item_candidates.begin(), top_end, candidate_order);
            } else {
                std::sort(item_candidates.begin(), item_candidates.end(), candidate_order);
            }
        }
        const auto scoring_end = Clock::now();
        scoring_ms += engine::debug::elapsed_ms(scoring_start, scoring_end);

        const auto update_start = Clock::now();
        for (size_t item = 0; item < batch; ++item) {
            auto & item_candidates = candidates[item];
            if (fill_counts[item] <= 0 || item_candidates.empty()) {
                continue;
            }
            auto & state = items[item];
            const int64_t actual_fill = std::min<int64_t>(fill_counts[item], static_cast<int64_t>(item_candidates.size()));
            for (int64_t i = 0; i < actual_fill; ++i) {
                const auto & candidate = item_candidates[static_cast<size_t>(i)];
                state.accepted[candidate.index] = candidate.predicted;
            }
            update_generated_tokens(inputs[item], assets, state.accepted);
            state.active_indices.erase(
                std::remove_if(
                    state.active_indices.begin(),
                    state.active_indices.end(),
                    [&](size_t flat_index) { return state.accepted[flat_index] != static_cast<int32_t>(mask_id); }),
                state.active_indices.end());
        }
        const auto upload_start = Clock::now();
        impl_->forward_graph->update_generated_target_tokens(inputs);
        const auto upload_end = Clock::now();
        upload_ms += engine::debug::elapsed_ms(upload_start, upload_end);
        update_ms += engine::debug::elapsed_ms(update_start, upload_end);
    }

    std::vector<OmniVoiceGeneratedAudioTokens> outputs(batch);
    for (size_t item = 0; item < batch; ++item) {
        const auto & state = items[item];
        for (int32_t token : state.accepted) {
            if (token == static_cast<int32_t>(mask_id)) {
                throw std::runtime_error("OmniVoice generator left masked tokens after iterative decoding");
            }
        }
        const int64_t target_frames = inputs[item].target_frames;
        auto & out = outputs[item];
        out.frames = target_frames;
        out.codebooks = codebooks;
        out.graph_rebuilt = impl_->last_stats.graph_rebuilt;
        out.graph_total_token_capacity = impl_->last_stats.total_token_capacity;
        out.graph_target_frame_capacity = impl_->last_stats.target_frame_capacity;
        // Timings cover the whole batch: every request shares the passes.
        out.forward_ms = upload_ms + compute_ms + readback_ms;
        out.upload_ms = upload_ms;
        out.compute_ms = compute_ms;
        out.readback_ms = readback_ms;
        out.scoring_ms = scoring_ms;
        out.update_ms = update_ms;
        out.decode_steps = state.executed_steps;
        out.token_ids.resize(static_cast<size_t>(target_frames * codebooks), 0);
        for (int64_t frame = 0; frame < target_frames; ++frame) {
            for (int64_t codebook = 0; codebook < codebooks; ++codebook) {
                out.token_ids[static_cast<size_t>(frame * codebooks + codebook)] =
                    state.accepted[static_cast<size_t>(codebook * target_frames + frame)];
            }
        }
    }
    impl_->last_stats.upload_ms = upload_ms;
    impl_->last_stats.compute_ms = compute_ms;
    impl_->last_stats.readback_ms = readback_ms;
    return outputs;
}

}  // namespace engine::models::omnivoice
