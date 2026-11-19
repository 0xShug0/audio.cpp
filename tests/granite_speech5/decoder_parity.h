#pragma once

// Test-only logits and teacher-forcing harness. Production inference uses
// GreedyCausalDecoderRuntime directly; no capture/readback API is added there.
#include "engine/community_models/granite_speech5/decoder.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <utility>

namespace engine::community_models::granite_speech5 {
namespace parity_detail {
using Clock = std::chrono::steady_clock;
void require_prompt_shapes(
    const runtime::GreedyCausalDecoderRuntime::Prompt & prompt,
    int64_t hidden_size) {
    const int64_t steps = static_cast<int64_t>(prompt.input_ids.size());
    const auto & injection = prompt.injection;
    if (injection.tokens < 0 || injection.tokens > steps ||
        static_cast<int64_t>(injection.positions.size()) != injection.tokens ||
        static_cast<int64_t>(injection.values.size()) != injection.tokens * hidden_size) {
        throw std::runtime_error("granite_speech5 decoder injection shape does not match the prompt");
    }
    for (const int32_t position : injection.positions) {
        if (position < 0 || position >= steps) {
            throw std::runtime_error("granite_speech5 decoder injection position is out of range");
        }
    }
}

int32_t argmax_index(const std::vector<float> & values) {
    if (values.empty()) {
        throw std::runtime_error("granite_speech5 decoder cannot select from empty logits");
    }
    size_t best = 0;
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return static_cast<int32_t>(best);
}

// Device-side token embedding lookup in 64-step blocks (same pattern as the
// shared greedy runtime: a small fixed-width graph also handles non-F32
// embedding storage, padding the final block and discarding it).
class PromptEmbeddingGraph {
public:
    static constexpr int64_t kSteps = 64;
    PromptEmbeddingGraph(
        core::ExecutionContext & execution,
        const core::TensorValue & token_embedding,
        int64_t hidden_size,
        int64_t vocab_size)
        : graph_(execution, {"granite_speech5.decoder.embedding", 1024 * 1024, 64}),
          hidden_size_(hidden_size) {
        auto & ctx = graph_.build_context();
        ids_ = ggml_new_tensor_1d(ctx.ggml, GGML_TYPE_I32, kSteps);
        auto ids = core::wrap_tensor(ids_, core::TensorShape::from_dims({kSteps}), GGML_TYPE_I32);
        output_ = modules::EmbeddingModule({vocab_size, hidden_size}).build(ctx, ids, token_embedding).tensor;
        ggml_set_output(output_);
        ggml_set_input(ids_);
        ggml_build_forward_expand(graph_.graph(), output_);
        graph_.allocate();
    }
    std::vector<float> run(const std::vector<int32_t> & ids) {
        std::vector<float> result(ids.size() * static_cast<size_t>(hidden_size_));
        std::vector<int32_t> block(kSteps, 0);
        for (size_t offset = 0; offset < ids.size(); offset += kSteps) {
            const size_t count = std::min<size_t>(kSteps, ids.size() - offset);
            std::fill(block.begin(), block.end(), 0);
            std::copy_n(ids.begin() + offset, count, block.begin());
            ggml_backend_tensor_set(ids_, block.data(), 0, block.size() * sizeof(int32_t));
            graph_.compute();
            ggml_backend_tensor_get(
                output_,
                result.data() + offset * hidden_size_,
                0,
                count * hidden_size_ * sizeof(float));
        }
        return result;
    }

private:
    core::PreparedGraph graph_;
    int64_t hidden_size_ = 0;
    ggml_tensor * ids_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

struct DecoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    std::vector<modules::DecoderLayerWeights> layers;
    modules::NormWeights final_norm;
    modules::LinearWeights lm_head;
};

DecoderWeights load_decoder_weights(
    const assets::TensorSource & source,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    DecoderWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend, backend_type, "granite_speech5.decoder.weights", weight_context_bytes);
    auto & store = *weights.store;
    weights.token_embedding =
        store.load_tensor(source, "language_model.model.embed_tokens.weight", storage_type, {100353, 2048});
    weights.layers.reserve(static_cast<size_t>(kGraniteSpeech5DecoderLayers));
    for (int64_t layer = 0; layer < kGraniteSpeech5DecoderLayers; ++layer) {
        const std::string prefix = "language_model.model.layers." + std::to_string(layer);
        modules::DecoderLayerWeights w;
        w.input_norm = {store.load_f32_tensor(source, prefix + ".input_layernorm.weight", {2048}),
                        std::nullopt};
        // Separate bias-free Q/K/V projections; no Q/K norms on Granite.
        w.self_attention.q_weight =
            store.load_tensor(source, prefix + ".self_attn.q_proj.weight", storage_type, {2048, 2048});
        w.self_attention.k_weight =
            store.load_tensor(source, prefix + ".self_attn.k_proj.weight", storage_type, {512, 2048});
        w.self_attention.v_weight =
            store.load_tensor(source, prefix + ".self_attn.v_proj.weight", storage_type, {512, 2048});
        w.self_attention.out_weight =
            store.load_tensor(source, prefix + ".self_attn.o_proj.weight", storage_type, {2048, 2048});
        w.post_norm = {store.load_f32_tensor(source, prefix + ".post_attention_layernorm.weight", {2048}),
                       std::nullopt};
        w.mlp.gate_proj = {store.load_tensor(
                                source, prefix + ".mlp.gate_proj.weight", storage_type, {4096, 2048}),
                            std::nullopt};
        w.mlp.up_proj = {store.load_tensor(
                              source, prefix + ".mlp.up_proj.weight", storage_type, {4096, 2048}),
                          std::nullopt};
        w.mlp.down_proj = {store.load_tensor(
                                source, prefix + ".mlp.down_proj.weight", storage_type, {2048, 4096}),
                            std::nullopt};
        weights.layers.push_back(std::move(w));
    }
    weights.final_norm = {store.load_f32_tensor(source, "language_model.model.norm.weight", {2048}),
                          std::nullopt};
    // Untied LM head (no bias in the pinned checkpoint).
    weights.lm_head = {store.load_tensor(
                            source, "language_model.lm_head.weight", storage_type, {100353, 2048}),
                        std::nullopt};
    store.upload();
    return weights;
}

modules::CausalDecoderRuntimeWeights bind_runtime_weights(const DecoderWeights & weights) {
    modules::CausalDecoderRuntimeWeights out;
    out.token_embedding = weights.token_embedding;
    out.stack.layers = weights.layers;
    out.final_norm = weights.final_norm;
    out.lm_head = weights.lm_head;
    return out;
}


}  // namespace parity_detail

// Logits-capable decoder over CausalDecoderRuntime (probe + C2 units).
class GraniteSpeech5Decoder {
public:
    GraniteSpeech5Decoder(
        std::shared_ptr<const assets::TensorSource> weights_source,
        core::ExecutionContext & execution,
        bool eager_attention = false,
        size_t prefill_graph_arena_bytes = 2ull * 1024ull * 1024ull * 1024ull,
        size_t decode_graph_arena_bytes = 1024ull * 1024ull * 1024ull,
        size_t weight_context_bytes = 4ull * 1024ull * 1024ull,
        assets::TensorStorageType weight_storage_type = assets::TensorStorageType::F32);
    ~GraniteSpeech5Decoder();

    GraniteSpeech5Decoder(const GraniteSpeech5Decoder &) = delete;
    GraniteSpeech5Decoder & operator=(const GraniteSpeech5Decoder &) = delete;

    struct PrefillResult {
        int64_t steps = 0;
        // Merged token-major [steps, 2048] embeddings AFTER audio
        // substitution, BEFORE the x12 scale (matches merged_prefill.f32).
        std::vector<float> merged;
        // Host-side x12 of merged (matches scaled_prefill.f32); the graph
        // applies the same factor once at the hidden-module entry, which
        // last-logit parity proves.
        std::vector<float> scaled;
        // Last-position logits (already /8).
        std::vector<float> last_logits;
    };

    // Device embedding lookup + host audio injection + bounded-block prefill
    // into the token-decode cache (keep_prefix_steps = 0: the full prompt is
    // always recomputed). cache capacity covers steps + max_new_tokens.
    PrefillResult prefill(
        const runtime::GreedyCausalDecoderRuntime::Prompt & prompt,
        int64_t max_new_tokens,
        int64_t chunk_steps = 64);

    // Single forced-decode step continuing the prefilled cache. Returns the
    // full [100353] logit vector (already /8).
    std::vector<float> decode_token(int32_t token);

    // Greedy continuation with EOS/budget stop (EOS excluded, mirrors the
    // shared greedy runtime loop).
    std::vector<int32_t> generate_greedy(
        const runtime::GreedyCausalDecoderRuntime::Prompt & prompt,
        int64_t max_new_tokens);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct GraniteSpeech5Decoder::Impl {
    Impl(
        std::shared_ptr<const assets::TensorSource> weights_source,
        core::ExecutionContext & execution,
        bool eager_attention,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type)
        : execution_(&execution),
          weights_(parity_detail::load_decoder_weights(
              *weights_source,
              execution.backend(),
              execution.backend_type(),
              weight_context_bytes,
              weight_storage_type)) {
        modules::CausalDecoderRuntimeConfig config;
        config.trace_name = "granite_speech5.decoder";
        config.decoder = make_granite_speech5_causal_decoder_config(execution.backend_type());
        if (eager_attention) config.decoder.stack.runtime.attention.allow_flash_attention = false;
        config.prefill_graph_arena_bytes = prefill_graph_arena_bytes;
        config.decode_graph_arena_bytes = decode_graph_arena_bytes;
        runtime_ =
            std::make_unique<modules::CausalDecoderRuntime>(execution, config, parity_detail::bind_runtime_weights(weights_));
        embedding_graph_ = std::make_unique<parity_detail::PromptEmbeddingGraph>(
            execution,
            weights_.token_embedding,
            2048,
            100353);
    }

    static constexpr int64_t kHiddenSize = 2048;

    GraniteSpeech5Decoder::PrefillResult prefill(
        const runtime::GreedyCausalDecoderRuntime::Prompt & prompt,
        int64_t max_new_tokens,
        int64_t chunk_steps) {
        const int64_t steps = static_cast<int64_t>(prompt.input_ids.size());
        granite_speech5_require_generation_budget(steps, max_new_tokens);
        parity_detail::require_prompt_shapes(prompt, kHiddenSize);
        if (chunk_steps <= 0) {
            throw std::runtime_error("granite_speech5 decoder chunk_steps must be positive");
        }
        const auto start = parity_detail::Clock::now();
        // Device lookup, then host audio substitution at the injected rows.
        std::vector<float> merged = embedding_graph_->run(prompt.input_ids);
        for (size_t i = 0; i < prompt.injection.positions.size(); ++i) {
            std::copy_n(
                prompt.injection.values.data() + i * kHiddenSize,
                kHiddenSize,
                merged.data() + prompt.injection.positions[i] * kHiddenSize);
        }
        const int64_t required = steps + max_new_tokens;
        const int64_t capacity = std::min<int64_t>(
            kGraniteSpeech5MaxPositionEmbeddings, (required + 127) / 128 * 128);
        auto step = runtime_->prefill_embeddings_into_cache(merged, steps, capacity, chunk_steps, 0);
        engine::debug::timing_log_scalar(
            "granite_speech5.decoder.prefill_ms", engine::debug::elapsed_ms(start, parity_detail::Clock::now()));
        GraniteSpeech5Decoder::PrefillResult out;
        out.steps = steps;
        out.merged = merged;
        out.scaled = merged;
        for (float & value : out.scaled) {
            value *= kGraniteSpeech5EmbedScale;
        }
        out.last_logits = std::move(step.logits);
        return out;
    }

    core::ExecutionContext * execution_ = nullptr;
    parity_detail::DecoderWeights weights_;
    std::unique_ptr<modules::CausalDecoderRuntime> runtime_;
    std::unique_ptr<parity_detail::PromptEmbeddingGraph> embedding_graph_;
};

inline GraniteSpeech5Decoder::GraniteSpeech5Decoder(
    std::shared_ptr<const assets::TensorSource> weights_source,
    core::ExecutionContext & execution,
    bool eager_attention,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(weights_source),
          execution,
          eager_attention,
          prefill_graph_arena_bytes,
          decode_graph_arena_bytes,
          weight_context_bytes,
          weight_storage_type)) {}

inline GraniteSpeech5Decoder::~GraniteSpeech5Decoder() = default;

inline GraniteSpeech5Decoder::PrefillResult GraniteSpeech5Decoder::prefill(
    const runtime::GreedyCausalDecoderRuntime::Prompt & prompt,
    int64_t max_new_tokens,
    int64_t chunk_steps) {
    return impl_->prefill(prompt, max_new_tokens, chunk_steps);
}

inline std::vector<float> GraniteSpeech5Decoder::decode_token(int32_t token) {
    return impl_->runtime_->decode_token(token).logits;
}

inline std::vector<int32_t> GraniteSpeech5Decoder::generate_greedy(
    const runtime::GreedyCausalDecoderRuntime::Prompt & prompt,
    int64_t max_new_tokens) {
    const auto prefill = impl_->prefill(prompt, max_new_tokens, 64);
    std::vector<int32_t> out;
    std::vector<float> logits = prefill.last_logits;
    for (int64_t step = 0; step < max_new_tokens; ++step) {
        const int32_t token = parity_detail::argmax_index(logits);
        if (token == kGraniteSpeech5EosTokenId) {
            break;
        }
        out.push_back(token);
        if (step + 1 < max_new_tokens) {
            logits = impl_->runtime_->decode_token(token).logits;
        }
    }
    return out;
}


}  // namespace engine::community_models::granite_speech5
