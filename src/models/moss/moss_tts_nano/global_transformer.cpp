#include "engine/models/moss/moss_tts_nano/global_transformer.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/transformers/causal_decoder.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/kv_cache.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::moss_tts_nano {
namespace {

namespace modules = engine::modules;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct TransformerLayerWeights {
    core::TensorValue ln1_weight;
    core::TensorValue ln1_bias;
    modules::LinearWeights c_attn;
    modules::LinearWeights c_proj;
    core::TensorValue ln2_weight;
    core::TensorValue ln2_bias;
    modules::LinearWeights fc_in;
    modules::LinearWeights fc_out;
};

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::reshape_tensor(ctx, contiguous, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

// Attention over `heads` shaped rows with an explicit additive mask. ggml_soft_max_ext takes the
// 1/sqrt(head_dim) scale directly, so no separate scaling node is needed. The prefill path passes a
// square causal mask, the decode path a single-row mask over the K/V cache.
core::TensorValue attention_from_heads_masked(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t head_dim,
    const core::TensorValue & mask) {
    const modules::MatMulModule matmul;
    auto scores = matmul.build(ctx, q_heads, modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    scores = core::ensure_backend_addressable_layout(ctx, scores);
    auto attn = core::wrap_tensor(
        ggml_soft_max_ext(
            ctx.ggml,
            scores.tensor,
            mask.tensor,
            1.0F / std::sqrt(static_cast<float>(head_dim)),
            0.0F),
        scores.shape,
        GGML_TYPE_F32);
    return matmul.build(ctx, attn, v_heads);
}

struct HeadProjections {
    core::TensorValue q_heads;      // {1, heads, rows, head_dim}
    core::TensorValue k_heads_raw;  // {1, rows, heads, head_dim} (post RoPE, pre transpose)
    core::TensorValue v_heads_raw;  // {1, rows, heads, head_dim}
};

// LayerNorm + fused c_attn + RoPE, i.e. everything before attention scoring.
HeadProjections project_attention_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const TransformerLayerWeights & weights,
    const MossTTSNanoGlobalTransformerConfig & config) {
    const int64_t head_dim = config.hidden_size / config.num_attention_heads;
    auto x = modules::LayerNormModule({config.hidden_size, config.layer_norm_epsilon, true, true})
                 .build(ctx, input, {weights.ln1_weight, weights.ln1_bias});
    auto qkv = modules::LinearModule({config.hidden_size, config.hidden_size * 3, true})
                   .build(ctx, x, {weights.c_attn.weight, weights.c_attn.bias});
    auto q = modules::SliceModule({2, 0, config.hidden_size}).build(ctx, qkv);
    auto k = modules::SliceModule({2, config.hidden_size, config.hidden_size}).build(ctx, qkv);
    auto v = modules::SliceModule({2, config.hidden_size * 2, config.hidden_size}).build(ctx, qkv);
    q = modules::RoPEModule({head_dim, GGML_ROPE_TYPE_NORMAL, config.rope_base})
            .build(ctx, reshape_heads(ctx, q, config.num_attention_heads, head_dim), positions);
    k = modules::RoPEModule({head_dim, GGML_ROPE_TYPE_NORMAL, config.rope_base})
            .build(ctx, reshape_heads(ctx, k, config.num_attention_heads, head_dim), positions);
    v = reshape_heads(ctx, v, config.num_attention_heads, head_dim);
    HeadProjections out;
    out.q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    out.k_heads_raw = k;
    out.v_heads_raw = v;
    return out;
}

// c_proj + residual + GELU MLP + residual, i.e. everything after attention.
core::TensorValue finish_attention_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & context_heads,
    const TransformerLayerWeights & weights,
    const MossTTSNanoGlobalTransformerConfig & config) {
    auto context = modules::TransposeModule({{0, 2, 1, 3}, context_heads.shape.rank}).build(ctx, context_heads);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.hidden_size}));
    const modules::AddModule add;
    auto x = add.build(
        ctx,
        input,
        modules::LinearModule({config.hidden_size, config.hidden_size, true})
            .build(ctx, context, {weights.c_proj.weight, weights.c_proj.bias}));
    auto ff = modules::LayerNormModule({config.hidden_size, config.layer_norm_epsilon, true, true})
                  .build(ctx, x, {weights.ln2_weight, weights.ln2_bias});
    ff = modules::LinearModule({config.hidden_size, config.intermediate_size, true})
             .build(ctx, ff, {weights.fc_in.weight, weights.fc_in.bias});
    ff = modules::GeluModule({modules::GeluApproximation::Tanh}).build(ctx, ff);
    ff = modules::LinearModule({config.intermediate_size, config.hidden_size, true})
             .build(ctx, ff, {weights.fc_out.weight, weights.fc_out.bias});
    return add.build(ctx, x, ff);
}

// Chunked prompt layer: appends the chunk's K/V to the cache in-graph and attends to the whole
// visible prefix [0, reuse_rows + steps). Reading the cached prefix (instead of only the chunk)
// is what makes a reused prefix numerically identical to forwarding the full prompt.
core::TensorValue cached_prefill_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const TransformerLayerWeights & weights,
    const MossTTSNanoGlobalTransformerConfig & config,
    const core::TensorValue & cache_keys,
    const core::TensorValue & cache_values,
    const core::TensorValue & kv_indices,
    int64_t visible_steps,
    const core::TensorValue & attention_mask,
    const modules::FastKVSetRowsModule & set_rows) {
    const int64_t head_dim = config.hidden_size / config.num_attention_heads;
    auto heads = project_attention_heads(ctx, input, positions, weights, config);
    const auto k_row = core::ensure_backend_addressable_layout(ctx, heads.k_heads_raw);
    const auto v_row = core::ensure_backend_addressable_layout(ctx, heads.v_heads_raw);
    const auto k_cache = set_rows.build_block(ctx, cache_keys, k_row, kv_indices);
    const auto v_cache = set_rows.build_block(ctx, cache_values, v_row, kv_indices);
    const auto k_used = engine::runtime::view_transformer_kv_cache_steps(
        ctx, k_cache, 0, visible_steps, config.num_attention_heads, head_dim, "moss_nano.prefill.key");
    const auto v_used = engine::runtime::view_transformer_kv_cache_steps(
        ctx, v_cache, 0, visible_steps, config.num_attention_heads, head_dim, "moss_nano.prefill.value");
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k_used.shape.rank}).build(ctx, k_used);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v_used.shape.rank}).build(ctx, v_used);
    auto context = attention_from_heads_masked(ctx, heads.q_heads, k_heads, v_heads, head_dim, attention_mask);
    return finish_attention_layer(ctx, input, context, weights, config);
}

// Decode row: attends to the K/V cache, appending this row's K/V through ggml_set_rows first.
core::TensorValue decode_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const TransformerLayerWeights & weights,
    const MossTTSNanoGlobalTransformerConfig & config,
    const core::TensorValue & cache_keys,
    const core::TensorValue & cache_values,
    const core::TensorValue & cache_slot,
    const core::TensorValue & attention_mask,
    const modules::FastKVSetRowsModule & set_rows) {
    const int64_t head_dim = config.hidden_size / config.num_attention_heads;
    auto heads = project_attention_heads(ctx, input, positions, weights, config);
    auto k_row = core::ensure_backend_addressable_layout(ctx, heads.k_heads_raw);
    auto v_row = core::ensure_backend_addressable_layout(ctx, heads.v_heads_raw);
    const auto k_cache = set_rows.build(ctx, cache_keys, k_row, cache_slot);
    const auto v_cache = set_rows.build(ctx, cache_values, v_row, cache_slot);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k_cache.shape.rank}).build(ctx, k_cache);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v_cache.shape.rank}).build(ctx, v_cache);
    auto context = attention_from_heads_masked(ctx, heads.q_heads, k_heads, v_heads, head_dim, attention_mask);
    return finish_attention_layer(ctx, input, context, weights, config);
}

}  // namespace

struct MossTTSNanoGlobalTransformerRuntime::Weights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue text_embedding;
    std::vector<core::TensorValue> audio_embeddings;
    std::vector<TransformerLayerWeights> layers;
    core::TensorValue final_norm_weight;
    core::TensorValue final_norm_bias;
};

// Persistent single-row decode graph. Every tensor (inputs, K/V cache, intermediates) is allocated
// once by alloc_ctx_tensors so its address stays stable across steps.
struct MossTTSNanoGlobalTransformerRuntime::DecodeGraph {
public:
    DecodeGraph(
        std::shared_ptr<const MossTTSNanoAssets> assets,
        const Weights & weights,
        core::ExecutionContext & execution,
        size_t arena_bytes,
        int64_t cache_steps)
        : assets_(std::move(assets)),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          compute_threads_(std::max(1, execution.config().threads)),
          cache_steps_(cache_steps) {
        if (assets_ == nullptr || backend_ == nullptr) {
            throw std::runtime_error("MOSS global decode graph requires assets and a backend");
        }
        if (cache_steps_ <= 0) {
            throw std::runtime_error("MOSS global decode graph requires a positive cache size");
        }
        const auto & config = assets_->config.global_transformer;
        const int64_t hidden_size = config.hidden_size;
        const int64_t head_dim = hidden_size / config.num_attention_heads;

        ggml_init_params params{arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MOSS global decode context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "moss_tts_nano.global.decode", backend_type_};

        auto text_token = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
        ggml_set_input(text_token.tensor);
        text_token_ = text_token.tensor;
        auto positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
        ggml_set_input(positions.tensor);
        positions_ = positions.tensor;
        auto cache_slot = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
        ggml_set_input(cache_slot.tensor);
        cache_slot_ = cache_slot.tensor;
        auto attention_mask = core::make_tensor(ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, 1, 1, cache_steps_}));
        ggml_set_input(attention_mask.tensor);
        attention_mask_ = attention_mask.tensor;

        auto hidden = modules::EmbeddingModule({config.vocab_size, hidden_size})
                          .build(ctx, text_token, weights.text_embedding);
        audio_ids_.reserve(static_cast<size_t>(assets_->config.n_vq));
        audio_masks_.reserve(static_cast<size_t>(assets_->config.n_vq));
        for (int64_t q = 0; q < assets_->config.n_vq; ++q) {
            auto ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
            ggml_set_input(ids.tensor);
            audio_ids_.push_back(ids.tensor);
            auto mask = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, hidden_size}));
            ggml_set_input(mask.tensor);
            audio_masks_.push_back(mask.tensor);
            auto emb = modules::EmbeddingModule(
                           {assets_->config.audio_codebook_sizes.at(static_cast<size_t>(q)), hidden_size})
                           .build(ctx, ids, weights.audio_embeddings.at(static_cast<size_t>(q)));
            hidden = modules::AddModule().build(ctx, hidden, modules::MulModule().build(ctx, emb, mask));
        }
        hidden = core::reshape_tensor(ctx, hidden, core::TensorShape::from_dims({1, 1, hidden_size}));

        const modules::FastKVSetRowsModule set_rows({modules::FastKVSetRowsMode::Exact});
        std::vector<core::TensorValue> cache_keys;
        std::vector<core::TensorValue> cache_values;
        cache_keys.reserve(weights.layers.size());
        cache_values.reserve(weights.layers.size());
        for (const auto & layer : weights.layers) {
            auto keys = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_, config.num_attention_heads, head_dim}));
            auto values = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_, config.num_attention_heads, head_dim}));
            if (keys.tensor == nullptr || values.tensor == nullptr) {
                throw std::runtime_error("failed to allocate MOSS global K/V cache tensors");
            }
            cache_keys.push_back(keys);
            cache_values.push_back(values);
            hidden = decode_layer(
                ctx,
                hidden,
                positions,
                layer,
                config,
                keys,
                values,
                cache_slot,
                attention_mask,
                set_rows);
        }

        hidden = modules::LayerNormModule({hidden_size, config.layer_norm_epsilon, true, true})
                     .build(ctx, hidden, {weights.final_norm_weight, weights.final_norm_bias});
        hidden = core::ensure_backend_addressable_layout(ctx, hidden);
        auto output = modules::ReshapeModule({core::TensorShape::from_dims({1, hidden_size})}).build(ctx, hidden);
        output = core::ensure_backend_addressable_layout(ctx, output);
        ggml_set_output(output.tensor);
        hidden_ = output.tensor;
        output_values_.resize(static_cast<size_t>(hidden_size));

        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, hidden_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), backend_);
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate MOSS global decode graph");
        }

        engine::runtime::TransformerKVCacheOptions cache_options;
        cache_options.lazy_import_scratch = true;
        cache_ = engine::runtime::TransformerKVCache(
            cache_steps_,
            config.num_attention_heads * head_dim,
            std::move(cache_keys),
            std::move(cache_values),
            cache_options);
    }

    ~DecodeGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
            buffer_ = nullptr;
        }
        ctx_.reset();
    }

    DecodeGraph(const DecodeGraph &) = delete;
    DecodeGraph & operator=(const DecodeGraph &) = delete;

    void reset_cache() {
        cache_.clear_on_backend();
    }

    int64_t cache_steps() const noexcept {
        return cache_.cache_steps();
    }

    int64_t valid_steps() const noexcept {
        return cache_.valid_steps();
    }

    const core::TensorValue & key_tensor(size_t layer) const {
        return cache_.key_tensor(layer);
    }

    const core::TensorValue & value_tensor(size_t layer) const {
        return cache_.value_tensor(layer);
    }

    void advance_cache(int64_t steps) {
        cache_.advance_after_direct_append(steps);
    }

    // Drops everything after `rows` so a new prompt can reuse the cached prefix.
    void retain_prefix(int64_t rows) {
        cache_.retain_prefix(rows);
    }

    void forward(
        int32_t text_token,
        const std::vector<int32_t> & code_row,
        const int64_t n_vq,
        const int64_t hidden_size,
        std::vector<float> & hidden_state) {
        if (cache_.valid_steps() >= cache_.cache_steps()) {
            throw std::runtime_error("MOSS global decode step exceeds the K/V cache capacity");
        }
        if (static_cast<int64_t>(code_row.size()) != n_vq) {
            throw std::runtime_error("MOSS global decode step requires one code per codebook");
        }
        const int32_t position = static_cast<int32_t>(cache_.current_end());
        const int32_t slot = static_cast<int32_t>(cache_.valid_steps());
        ggml_backend_tensor_set(text_token_, &text_token, 0, sizeof(int32_t));
        ggml_backend_tensor_set(positions_, &position, 0, sizeof(int32_t));
        ggml_backend_tensor_set(cache_slot_, &slot, 0, sizeof(int32_t));

        mask_row_.assign(static_cast<size_t>(hidden_size), 1.0F);
        for (int64_t q = 0; q < n_vq; ++q) {
            const int32_t code = code_row[static_cast<size_t>(q)];
            const bool is_pad = code == static_cast<int32_t>(assets_->config.audio_pad_token_id);
            const int32_t id = is_pad ? 0 : code;
            ggml_backend_tensor_set(audio_ids_[static_cast<size_t>(q)], &id, 0, sizeof(int32_t));
            std::fill(mask_row_.begin(), mask_row_.end(), is_pad ? 0.0F : 1.0F);
            ggml_backend_tensor_set(
                audio_masks_[static_cast<size_t>(q)],
                mask_row_.data(),
                0,
                mask_row_.size() * sizeof(float));
        }
        modules::write_decoder_cached_step_mask(
            attention_mask_,
            mask_scratch_,
            cache_.cache_steps(),
            cache_.valid_steps(),
            cache_.valid_steps());

        core::set_backend_threads(backend_, compute_threads_);
        const ggml_status status = ggml_backend_graph_compute(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MOSS global decode graph compute failed");
        }
        ggml_backend_tensor_get(hidden_, output_values_.data(), 0, output_values_.size() * sizeof(float));
        hidden_state = output_values_;
        cache_.advance_after_direct_append(1);
    }

private:
    std::shared_ptr<const MossTTSNanoAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int compute_threads_ = 1;
    int64_t cache_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_tensor * text_token_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * cache_slot_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    std::vector<ggml_tensor *> audio_ids_;
    std::vector<ggml_tensor *> audio_masks_;
    ggml_tensor * hidden_ = nullptr;
    std::vector<float> output_values_;
    std::vector<float> mask_row_;
    std::vector<ggml_fp16_t> mask_scratch_;
    engine::runtime::TransformerKVCache cache_;
};

MossTTSNanoGlobalTransformerRuntime::MossTTSNanoGlobalTransformerRuntime(
    std::shared_ptr<const MossTTSNanoAssets> assets,
    core::ExecutionContext & execution_context,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType weight_storage_type,
    MossTTSNanoPrefillMode prefill_mode)
    : assets_(std::move(assets)),
      execution_context_(execution_context),
      prefill_mode_(prefill_mode) {
    if (assets_ == nullptr) {
        throw std::runtime_error("MOSS global transformer requires assets");
    }
    weights_ = std::make_shared<Weights>();
    const auto & source = *assets_->model_weights;
    const auto & config = assets_->config.global_transformer;
    weights_->store = std::make_shared<core::BackendWeightStore>(
        execution_context_.backend(),
        execution_context_.backend_type(),
        "moss_tts_nano.global.weights",
        weight_context_bytes);
    weights_->text_embedding = weights_->store->load_tensor(
        source,
        "transformer.wte.weight",
        weight_storage_type,
        {config.vocab_size, config.hidden_size});
    weights_->audio_embeddings.reserve(static_cast<size_t>(assets_->config.n_vq));
    for (int64_t q = 0; q < assets_->config.n_vq; ++q) {
        weights_->audio_embeddings.push_back(weights_->store->load_tensor(
            source,
            "audio_embeddings." + std::to_string(q) + ".weight",
            weight_storage_type,
            {assets_->config.audio_codebook_sizes.at(static_cast<size_t>(q)), config.hidden_size}));
    }
    weights_->layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = "transformer.h." + std::to_string(layer);
        TransformerLayerWeights w;
        w.ln1_weight = weights_->store->load_f32_tensor(source, prefix + ".ln_1.weight", {config.hidden_size});
        w.ln1_bias = weights_->store->load_f32_tensor(source, prefix + ".ln_1.bias", {config.hidden_size});
        w.c_attn = modules::binding::linear_from_source(*weights_->store, source, prefix + ".attn.c_attn", weight_storage_type, config.hidden_size * 3, config.hidden_size, true);
        w.c_proj = modules::binding::linear_from_source(*weights_->store, source, prefix + ".attn.c_proj", weight_storage_type, config.hidden_size, config.hidden_size, true);
        w.ln2_weight = weights_->store->load_f32_tensor(source, prefix + ".ln_2.weight", {config.hidden_size});
        w.ln2_bias = weights_->store->load_f32_tensor(source, prefix + ".ln_2.bias", {config.hidden_size});
        w.fc_in = modules::binding::linear_from_source(*weights_->store, source, prefix + ".mlp.fc_in", weight_storage_type, config.intermediate_size, config.hidden_size, true);
        w.fc_out = modules::binding::linear_from_source(*weights_->store, source, prefix + ".mlp.fc_out", weight_storage_type, config.hidden_size, config.intermediate_size, true);
        weights_->layers.push_back(std::move(w));
    }
    weights_->final_norm_weight = weights_->store->load_f32_tensor(source, "transformer.ln_f.weight", {config.hidden_size});
    weights_->final_norm_bias = weights_->store->load_f32_tensor(source, "transformer.ln_f.bias", {config.hidden_size});
    weights_->store->upload();
    prefill_graph_arena_bytes_ = prefill_graph_arena_bytes;
    decode_graph_arena_bytes_ = decode_graph_arena_bytes;
}

MossTTSNanoGlobalTransformerRuntime::~MossTTSNanoGlobalTransformerRuntime() = default;

void MossTTSNanoGlobalTransformerRuntime::begin_generation(int64_t max_positions) {
    if (max_positions <= 0) {
        throw std::runtime_error("MOSS global generation requires a positive row budget");
    }
    if (assets_ == nullptr || weights_ == nullptr) {
        throw std::runtime_error("MOSS global generation requires loaded weights");
    }
    const auto & config = assets_->config.global_transformer;
    // 2 tensors (K and V) per layer covering `max_positions` steps of `hidden_size` f32. The K/V
    // data lives in its own backend buffer (ggml_backend_alloc_ctx_tensors inside DecodeGraph);
    // the graph arena only ever holds tensor metadata and graph nodes, so it does not bound the
    // cache size and must not be checked against it.
    const int64_t cache_bytes =
        2 * config.num_hidden_layers * max_positions * config.hidden_size * static_cast<int64_t>(sizeof(float));
    engine::debug::trace_log_scalar("moss_tts_nano.global.decode.rows", max_positions);
    engine::debug::trace_log_scalar("moss_tts_nano.global.decode.kv_cache_bytes", cache_bytes);
    // Already large enough: keep the graph and its cache so the new prompt can reuse its prefix.
    if (decode_ != nullptr && decode_->cache_steps() >= max_positions) {
        return;
    }
    decode_ = std::make_unique<DecodeGraph>(
        assets_,
        *weights_,
        execution_context_,
        decode_graph_arena_bytes_,
        max_positions);
    decode_->reset_cache();
    ++graph_builds_;
}

std::vector<float> MossTTSNanoGlobalTransformerRuntime::prefill(
    const std::vector<int32_t> & rows,
    int64_t row_count,
    int64_t row_width,
    int64_t reuse_rows) {
    if (decode_ == nullptr) {
        throw std::runtime_error("MOSS global prefill requires begin_generation first");
    }
    const auto & config = assets_->config.global_transformer;
    const int64_t n_vq = assets_->config.n_vq;
    const int64_t hidden_size = config.hidden_size;
    const int64_t head_dim = hidden_size / config.num_attention_heads;
    if (row_width != n_vq + 1) {
        throw std::runtime_error("MOSS global prefill row width does not match the codebook count");
    }
    if (row_count <= 0 || static_cast<int64_t>(rows.size()) != row_count * row_width) {
        throw std::runtime_error("MOSS global prefill row count mismatch");
    }
    if (reuse_rows < 0 || reuse_rows > row_count) {
        throw std::runtime_error("MOSS global prefill reuse row count is out of range");
    }
    if (reuse_rows > decode_->valid_steps()) {
        throw std::runtime_error("MOSS global prefill reuse exceeds the cached prefix");
    }
    if (reuse_rows == row_count) {
        // The cache already covers the whole prompt, but the caller still needs the last row's
        // hidden state, so the final row is always forwarded.
        reuse_rows = row_count - 1;
    }
    if (decode_->valid_steps() != reuse_rows) {
        decode_->retain_prefix(reuse_rows);
    }
    const int64_t steps = row_count - reuse_rows;
    if (decode_->valid_steps() + steps > decode_->cache_steps()) {
        throw std::runtime_error("MOSS global prefill exceeds the K/V cache capacity");
    }
    engine::debug::trace_log_scalar("moss_tts_nano.global.prefill.total_rows", row_count);
    engine::debug::trace_log_scalar("moss_tts_nano.global.prefill.reuse_rows", reuse_rows);
    engine::debug::trace_log_scalar("moss_tts_nano.global.prefill.forward_rows", steps);

    if (prefill_mode_ == MossTTSNanoPrefillMode::SingleRow) {
        // Forward the remaining rows one at a time through the single-row decode graph: with
        // quantized weights every row of the trajectory (prompt and generated frames) then runs
        // the same batch-1 matmul kernels, which keeps the incremental decode numerically
        // self-consistent on backends whose batched and single-row quantized kernels differ.
        // Slower than the batched prefill — one graph launch per row.
        std::vector<float> hidden;
        std::vector<int32_t> code_row(static_cast<size_t>(n_vq));
        for (int64_t row = reuse_rows; row < row_count; ++row) {
            const auto offset = static_cast<size_t>(row * row_width);
            for (int64_t q = 0; q < n_vq; ++q) {
                code_row[static_cast<size_t>(q)] = rows[offset + 1 + static_cast<size_t>(q)];
            }
            step_into(rows[offset], code_row, hidden);
        }
        return hidden;
    }

    ggml_init_params params{prefill_graph_arena_bytes_, nullptr, true};
    std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx(ggml_init(params));
    if (graph_ctx == nullptr) {
        throw std::runtime_error("failed to initialize MOSS global prefill context");
    }
    core::ModuleBuildContext ctx{graph_ctx.get(), "moss_tts_nano.global.prefill", execution_context_.backend_type()};

    auto text_ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
    ggml_set_input(text_ids.tensor);
    auto positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
    ggml_set_input(positions.tensor);
    const int64_t visible_steps = reuse_rows + steps;
    auto attention_mask = core::make_tensor(ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, 1, steps, visible_steps}));
    ggml_set_input(attention_mask.tensor);
    auto kv_indices = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
    ggml_set_input(kv_indices.tensor);
    std::vector<core::TensorValue> audio_ids;
    std::vector<core::TensorValue> audio_masks;
    audio_ids.reserve(static_cast<size_t>(n_vq));
    audio_masks.reserve(static_cast<size_t>(n_vq));
    for (int64_t q = 0; q < n_vq; ++q) {
        auto ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
        ggml_set_input(ids.tensor);
        audio_ids.push_back(ids);
        auto mask = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({steps, hidden_size}));
        ggml_set_input(mask.tensor);
        audio_masks.push_back(mask);
    }

    auto hidden = modules::EmbeddingModule({config.vocab_size, hidden_size})
                      .build(ctx, text_ids, weights_->text_embedding);
    for (int64_t q = 0; q < n_vq; ++q) {
        auto emb = modules::EmbeddingModule(
                       {assets_->config.audio_codebook_sizes.at(static_cast<size_t>(q)), hidden_size})
                       .build(ctx, audio_ids[static_cast<size_t>(q)], weights_->audio_embeddings.at(static_cast<size_t>(q)));
        hidden = modules::AddModule().build(
            ctx, hidden, modules::MulModule().build(ctx, emb, audio_masks[static_cast<size_t>(q)]));
    }
    hidden = core::reshape_tensor(ctx, hidden, core::TensorShape::from_dims({1, steps, hidden_size}));

    const modules::FastKVSetRowsModule set_rows({modules::FastKVSetRowsMode::Exact});
    for (size_t li = 0; li < weights_->layers.size(); ++li) {
        hidden = cached_prefill_layer(
            ctx,
            hidden,
            positions,
            weights_->layers[li],
            config,
            decode_->key_tensor(li),
            decode_->value_tensor(li),
            kv_indices,
            visible_steps,
            attention_mask,
            set_rows);
    }
    hidden = modules::LayerNormModule({hidden_size, config.layer_norm_epsilon, true, true})
                 .build(ctx, hidden, {weights_->final_norm_weight, weights_->final_norm_bias});
    hidden = core::ensure_backend_addressable_layout(ctx, hidden);
    auto output = modules::ReshapeModule({core::TensorShape::from_dims({steps, hidden_size})}).build(ctx, hidden);
    output = core::ensure_backend_addressable_layout(ctx, output);
    ggml_set_output(output.tensor);

    // The K/V writes into the persistent cache happen inside each layer's subgraph (ggml_set_rows),
    // so the whole prompt chunk is one graph that both updates and reads the cache.
    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx.get(), 65536, false);
    ggml_build_forward_expand(graph, output.tensor);

    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_context_.backend()));
    if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
        throw std::runtime_error("failed to allocate MOSS global prefill graph");
    }

    std::vector<int32_t> text_host(static_cast<size_t>(steps), 0);
    std::vector<int32_t> position_host(static_cast<size_t>(steps), 0);
    std::vector<std::vector<int32_t>> audio_id_host(static_cast<size_t>(n_vq), std::vector<int32_t>(static_cast<size_t>(steps), 0));
    std::vector<std::vector<float>> audio_mask_host(
        static_cast<size_t>(n_vq),
        std::vector<float>(static_cast<size_t>(steps * hidden_size), 0.0F));
    for (int64_t row = 0; row < steps; ++row) {
        const int64_t source_row = reuse_rows + row;
        position_host[static_cast<size_t>(row)] = static_cast<int32_t>(source_row);
        text_host[static_cast<size_t>(row)] = rows[static_cast<size_t>(source_row * row_width)];
        for (int64_t q = 0; q < n_vq; ++q) {
            const int32_t token = rows[static_cast<size_t>(source_row * row_width + 1 + q)];
            const bool is_pad = token == static_cast<int32_t>(assets_->config.audio_pad_token_id);
            audio_id_host[static_cast<size_t>(q)][static_cast<size_t>(row)] = is_pad ? 0 : token;
            if (!is_pad) {
                std::fill_n(
                    audio_mask_host[static_cast<size_t>(q)].begin() + static_cast<std::ptrdiff_t>(row * hidden_size),
                    static_cast<size_t>(hidden_size),
                    1.0F);
            }
        }
    }

    ggml_backend_tensor_set(text_ids.tensor, text_host.data(), 0, text_host.size() * sizeof(int32_t));
    ggml_backend_tensor_set(positions.tensor, position_host.data(), 0, position_host.size() * sizeof(int32_t));
    // Suffix causal mask: query row i sees keys [0, reuse_rows + i]. With reuse_rows == 0 this is
    // the plain causal mask.
    const auto mask_values = modules::causal_suffix_mask_values(1, steps, reuse_rows);
    ggml_backend_tensor_set(
        attention_mask.tensor, mask_values.data(), 0, mask_values.size() * sizeof(ggml_fp16_t));
    const auto kv_indices_host = modules::decoder_position_ids(steps, reuse_rows);
    ggml_backend_tensor_set(
        kv_indices.tensor, kv_indices_host.data(), 0, kv_indices_host.size() * sizeof(int32_t));
    for (int64_t q = 0; q < n_vq; ++q) {
        ggml_backend_tensor_set(
            audio_ids[static_cast<size_t>(q)].tensor,
            audio_id_host[static_cast<size_t>(q)].data(),
            0,
            audio_id_host[static_cast<size_t>(q)].size() * sizeof(int32_t));
        ggml_backend_tensor_set(
            audio_masks[static_cast<size_t>(q)].tensor,
            audio_mask_host[static_cast<size_t>(q)].data(),
            0,
            audio_mask_host[static_cast<size_t>(q)].size() * sizeof(float));
    }

    core::set_backend_threads(execution_context_.backend(), std::max(1, execution_context_.config().threads));
    const ggml_status status = ggml_backend_graph_compute(execution_context_.backend(), graph);
    ggml_backend_synchronize(execution_context_.backend());
    if (status != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(gallocr);
        throw std::runtime_error("MOSS global prefill graph compute failed");
    }

    std::vector<float> last_hidden(static_cast<size_t>(hidden_size));
    ggml_backend_tensor_get(
        output.tensor,
        last_hidden.data(),
        static_cast<size_t>((steps - 1) * hidden_size) * sizeof(float),
        last_hidden.size() * sizeof(float));
    ggml_gallocr_free(gallocr);
    decode_->advance_cache(steps);
    return last_hidden;
}

void MossTTSNanoGlobalTransformerRuntime::step_into(
    int32_t text_token,
    const std::vector<int32_t> & code_row,
    std::vector<float> & hidden_state) {
    if (decode_ == nullptr) {
        throw std::runtime_error("MOSS global decode step requires begin_generation first");
    }
    const auto & config = assets_->config.global_transformer;
    decode_->forward(
        text_token,
        code_row,
        assets_->config.n_vq,
        config.hidden_size,
        hidden_state);
}

void MossTTSNanoGlobalTransformerRuntime::reset_cache() {
    if (decode_ != nullptr) {
        decode_->reset_cache();
    }
}

int64_t MossTTSNanoGlobalTransformerRuntime::cached_positions() const noexcept {
    return decode_ == nullptr ? 0 : decode_->valid_steps();
}

int64_t MossTTSNanoGlobalTransformerRuntime::cache_capacity() const noexcept {
    return decode_ == nullptr ? 0 : decode_->cache_steps();
}

int64_t MossTTSNanoGlobalTransformerRuntime::graph_builds() const noexcept {
    return graph_builds_;
}

}  // namespace engine::models::moss_tts_nano
