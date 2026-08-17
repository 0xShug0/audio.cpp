#include "engine/framework/modules/transformers/qwen_causal_decode_runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/lookup_modules.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace engine::modules {
namespace {

using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

void validate_runtime_config(const QwenCausalDecodeRuntimeConfig & config) {
    if (config.prefill_graph_arena_bytes == 0 || config.decode_graph_arena_bytes == 0) {
        throw std::runtime_error("QwenCausalDecodeRuntime requires positive graph arena sizes");
    }
    if (config.decoder.stack.hidden_size <= 0) {
        throw std::runtime_error("QwenCausalDecodeRuntime requires positive hidden size");
    }
    if (config.output_mode == QwenCausalDecodeOutputMode::Logits && config.decoder.logits_size <= 0) {
        throw std::runtime_error("QwenCausalDecodeRuntime logits mode requires positive logits size");
    }
    if (config.readback_round_type.has_value() && *config.readback_round_type != GGML_TYPE_BF16) {
        throw std::runtime_error("QwenCausalDecodeRuntime readback rounding currently supports only bf16");
    }
    if (!config.logits_readback_token_ids.empty()) {
        if (config.output_mode != QwenCausalDecodeOutputMode::Logits) {
            throw std::runtime_error("QwenCausalDecodeRuntime compact logits readback requires logits mode");
        }
        for (const int32_t token : config.logits_readback_token_ids) {
            if (token < 0 || token >= config.decoder.logits_size) {
                throw std::runtime_error("QwenCausalDecodeRuntime compact logits token id is outside logits size");
            }
        }
    }
}

core::TensorValue token_embedding_input(
    core::ModuleBuildContext & ctx,
    const QwenCausalDecodeRuntimeWeights & weights,
    const QwenCausalDecoderConfig & config,
    ggml_tensor * token_ids,
    int64_t steps) {
    auto ids = core::wrap_tensor(
        token_ids,
        core::TensorShape::from_dims({steps}),
        GGML_TYPE_I32);
    auto x = EmbeddingModule({weights.token_embedding.shape.dims[0], config.stack.hidden_size})
                 .build(ctx, ids, weights.token_embedding);
    return core::reshape_tensor(
        ctx,
        x,
        core::TensorShape::from_dims({1, steps, config.stack.hidden_size}));
}

core::TensorValue token_embedding_input_batched(
    core::ModuleBuildContext & ctx,
    const QwenCausalDecodeRuntimeWeights & weights,
    const QwenCausalDecoderConfig & config,
    ggml_tensor * token_ids,
    int64_t batch_size,
    int64_t steps) {
    auto ids = core::wrap_tensor(
        token_ids,
        core::TensorShape::from_dims({batch_size, steps}),
        GGML_TYPE_I32);
    return EmbeddingModule({weights.token_embedding.shape.dims[0], config.stack.hidden_size})
        .build(ctx, ids, weights.token_embedding);
}

QwenDecoderHiddenConfig hidden_config_from_runtime(const QwenCausalDecodeRuntimeConfig & config) {
    QwenDecoderHiddenConfig out;
    out.stack = config.decoder.stack;
    out.hidden_mode = config.decoder.logits_mode;
    out.static_cache_type = config.decoder.static_cache_type;
    return out;
}

QwenDecoderHiddenWeights hidden_weights_from_runtime(const QwenCausalDecodeRuntimeWeights & weights) {
    QwenDecoderHiddenWeights out;
    out.stack = weights.stack;
    out.final_norm = weights.final_norm;
    return out;
}

QwenCausalDecoderWeights causal_decoder_weights(const QwenCausalDecodeRuntimeWeights & weights) {
    if (!weights.lm_head.has_value()) {
        throw std::runtime_error("QwenCausalDecodeRuntime logits mode requires lm_head weights");
    }
    QwenCausalDecoderWeights out;
    out.stack = weights.stack;
    out.final_norm = weights.final_norm;
    out.lm_head = *weights.lm_head;
    return out;
}

void round_readback(std::vector<float> & values, const QwenCausalDecodeRuntimeConfig & config) {
    if (!config.readback_round_type.has_value()) {
        return;
    }
    if (*config.readback_round_type == GGML_TYPE_BF16) {
        core::round_f32_to_bf16_in_place(values);
    }
}

core::TensorValue compact_logits_readback(
    core::ModuleBuildContext & ctx,
    const QwenCausalDecodeRuntimeConfig & config,
    const core::TensorValue & logits,
    const core::TensorValue & token_ids) {
    if (logits.shape.last_dim() != config.decoder.logits_size) {
        throw std::runtime_error("QwenCausalDecodeRuntime compact logits requires logits on the last dimension");
    }

    const int64_t rows = logits.shape.prefix_elements();
    const int64_t compact_size = static_cast<int64_t>(config.logits_readback_token_ids.size());
    const auto flat = core::reshape_tensor(
        ctx,
        logits,
        core::TensorShape::from_dims({rows, config.decoder.logits_size}));
    const auto transposed = core::wrap_tensor(
        ggml_transpose(ctx.ggml, flat.tensor),
        core::TensorShape::from_dims({config.decoder.logits_size, rows}),
        flat.type);
    const auto source = core::wrap_tensor(
        ggml_cont(ctx.ggml, transposed.tensor),
        transposed.shape,
        transposed.type);
    const auto selected = core::wrap_tensor(
        ggml_get_rows(ctx.ggml, source.tensor, token_ids.tensor),
        core::TensorShape::from_dims({compact_size, rows}),
        GGML_TYPE_F32);
    const auto selected_t = core::wrap_tensor(
        ggml_transpose(ctx.ggml, selected.tensor),
        core::TensorShape::from_dims({rows, compact_size}),
        GGML_TYPE_F32);
    const auto contiguous = core::wrap_tensor(
        ggml_cont(ctx.ggml, selected_t.tensor),
        selected_t.shape,
        selected_t.type);
    return core::reshape_tensor(ctx, contiguous, logits.shape.with_last_dim(compact_size));
}

ggml_tensor * make_logits_readback_token_ids(
    ggml_context * ctx,
    const QwenCausalDecodeRuntimeConfig & config) {
    if (config.logits_readback_token_ids.empty()) {
        return nullptr;
    }
    return ggml_new_tensor_1d(
        ctx,
        GGML_TYPE_I32,
        static_cast<int64_t>(config.logits_readback_token_ids.size()));
}

core::TensorValue wrap_logits_readback_token_ids(
    ggml_tensor * tensor,
    const QwenCausalDecodeRuntimeConfig & config) {
    return core::wrap_tensor(
        tensor,
        core::TensorShape::from_dims({static_cast<int64_t>(config.logits_readback_token_ids.size())}),
        GGML_TYPE_I32);
}

void upload_logits_readback_token_ids(
    ggml_tensor * tensor,
    const QwenCausalDecodeRuntimeConfig & config) {
    ggml_backend_tensor_set(
        tensor,
        config.logits_readback_token_ids.data(),
        0,
        config.logits_readback_token_ids.size() * sizeof(int32_t));
}

QwenCausalDecoderOutputs build_causal_prefill(
    core::ModuleBuildContext & ctx,
    const QwenCausalDecodeRuntimeConfig & config,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const QwenCausalDecodeRuntimeWeights & weights,
    const core::TensorValue & attention_mask) {
    if (config.output_mode == QwenCausalDecodeOutputMode::Logits) {
        auto causal_weights = causal_decoder_weights(weights);
        return QwenCausalDecoderModule(config.decoder)
            .build(ctx, input, positions, causal_weights, std::nullopt, attention_mask);
    }

    auto hidden_out = QwenDecoderHiddenModule(hidden_config_from_runtime(config))
                          .build(
                              ctx,
                              input,
                              positions,
                              hidden_weights_from_runtime(weights),
                              std::nullopt,
                              attention_mask);
    return {
        std::move(hidden_out.sequence),
        hidden_out.hidden,
        {},
        std::move(hidden_out.state),
    };
}

QwenCausalDecoderStaticCacheOutputs build_causal_decode(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const QwenCausalDecodeRuntimeConfig & config,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const QwenCausalDecodeRuntimeWeights & weights,
    int64_t cache_steps,
    const core::TensorValue & attention_mask,
    const core::TensorValue & cache_slot) {
    if (config.output_mode == QwenCausalDecodeOutputMode::Logits) {
        auto causal_weights = causal_decoder_weights(weights);
        return QwenCausalDecoderModule(config.decoder)
            .build_static_cache_tail(
                ctx,
                graph,
                input,
                positions,
                causal_weights,
                cache_steps,
                attention_mask,
                cache_slot);
    }

    auto hidden_out = QwenDecoderHiddenModule(hidden_config_from_runtime(config))
                          .build_static_cache_tail(
                              ctx,
                              graph,
                              input,
                              positions,
                              hidden_weights_from_runtime(weights),
                              cache_steps,
                              attention_mask,
                              cache_slot);
    return {
        std::move(hidden_out.sequence),
        hidden_out.hidden,
        {},
        std::move(hidden_out.cache),
    };
}

QwenCausalDecoderBatchedStaticCacheOutputs build_causal_decode_batched(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const QwenCausalDecodeRuntimeConfig & config,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const QwenCausalDecodeRuntimeWeights & weights,
    int64_t cache_steps,
    const core::TensorValue & attention_mask,
    const core::TensorValue & cache_slot) {
    if (config.output_mode == QwenCausalDecodeOutputMode::Logits) {
        auto causal_weights = causal_decoder_weights(weights);
        return QwenCausalDecoderModule(config.decoder)
            .build_static_cache_tail_batched(
                ctx,
                graph,
                input,
                positions,
                causal_weights,
                cache_steps,
                attention_mask,
                cache_slot);
    }

    auto hidden_out = QwenDecoderHiddenModule(hidden_config_from_runtime(config))
                          .build_static_cache_tail_batched(
                              ctx,
                              graph,
                              input,
                              positions,
                              hidden_weights_from_runtime(weights),
                              cache_steps,
                              attention_mask,
                              cache_slot);
    return {
        std::move(hidden_out.sequence),
        hidden_out.hidden,
        {},
        std::move(hidden_out.cache),
    };
}

}  // namespace

class QwenCausalDecodeRuntime::Impl {
public:
    Impl(
        core::ExecutionContext & execution,
        QwenCausalDecodeRuntimeConfig config,
        QwenCausalDecodeRuntimeWeights weights)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          config_(std::move(config)),
          weights_(std::move(weights)) {
        validate_runtime_config(config_);
        if (backend_ == nullptr) {
            throw std::runtime_error("QwenCausalDecodeRuntime backend is not initialized");
        }
    }

    ~Impl() {
        release_runtime_graphs();
    }

    QwenCausalPrefillResult prefill_tokens(const std::vector<int32_t> & token_ids) {
        if (token_ids.empty()) {
            throw std::runtime_error("QwenCausalDecodeRuntime prefill requires tokens");
        }
        ensure_prefill_token_graph(static_cast<int64_t>(token_ids.size()));
        ggml_backend_tensor_set(
            prefill_input_,
            token_ids.data(),
            0,
            token_ids.size() * sizeof(int32_t));
        return run_prefill();
    }

    QwenCausalPrefillResult prefill_embeddings(const std::vector<float> & embeddings, int64_t steps) {
        if (steps <= 0) {
            throw std::runtime_error("QwenCausalDecodeRuntime prefill requires positive embedding steps");
        }
        const size_t expected = static_cast<size_t>(steps * config_.decoder.stack.hidden_size);
        if (embeddings.size() != expected) {
            throw std::runtime_error("QwenCausalDecodeRuntime prefill embedding size mismatch");
        }
        ensure_prefill_embedding_graph(steps);
        ggml_backend_tensor_set(
            prefill_input_,
            embeddings.data(),
            0,
            embeddings.size() * sizeof(float));
        return run_prefill();
    }

    QwenCausalBatchedPrefillResult prefill_tokens_batched(
        const std::vector<int32_t> & token_ids,
        int64_t batch_size,
        int64_t steps) {
        if (batch_size <= 0 || steps <= 0) {
            throw std::runtime_error("QwenCausalDecodeRuntime batched prefill requires positive batch and steps");
        }
        if (token_ids.size() != static_cast<size_t>(batch_size * steps)) {
            throw std::runtime_error("QwenCausalDecodeRuntime batched prefill token size mismatch");
        }
        ensure_batched_prefill_token_graph(batch_size, steps);
        ggml_backend_tensor_set(
            batched_prefill_input_,
            token_ids.data(),
            0,
            token_ids.size() * sizeof(int32_t));
        return run_batched_prefill();
    }

    QwenCausalBatchedPrefillResult prefill_embeddings_batched(
        const std::vector<float> & embeddings,
        int64_t batch_size,
        int64_t steps) {
        if (batch_size <= 0 || steps <= 0) {
            throw std::runtime_error("QwenCausalDecodeRuntime batched prefill requires positive batch and steps");
        }
        const size_t expected = static_cast<size_t>(batch_size * steps * config_.decoder.stack.hidden_size);
        if (embeddings.size() != expected) {
            throw std::runtime_error("QwenCausalDecodeRuntime batched prefill embedding size mismatch");
        }
        ensure_batched_prefill_embedding_graph(batch_size, steps);
        ggml_backend_tensor_set(
            batched_prefill_input_,
            embeddings.data(),
            0,
            embeddings.size() * sizeof(float));
        return run_batched_prefill();
    }

    void start_decode_tokens(const runtime::TransformerKVState & state, int64_t required_cache_steps) {
        if (required_cache_steps <= 0) {
            throw std::runtime_error("QwenCausalDecodeRuntime decode requires positive cache capacity");
        }
        ensure_decode_token_graph(required_cache_steps);
        decode_cache_.import_state(state);
    }

    void start_decode_embeddings(const runtime::TransformerKVState & state, int64_t required_cache_steps) {
        if (required_cache_steps <= 0) {
            throw std::runtime_error("QwenCausalDecodeRuntime decode requires positive cache capacity");
        }
        ensure_decode_embedding_graph(required_cache_steps);
        decode_cache_.import_state(state);
    }

    QwenCausalDecodeStepResult decode_token(int32_t token) {
        ensure_decode_started();
        if (decode_input_kind_ != InputKind::Token) {
            throw std::runtime_error("QwenCausalDecodeRuntime decode graph expects embeddings");
        }
        ggml_backend_tensor_set(decode_input_, &token, 0, sizeof(int32_t));
        return run_decode_step();
    }

    QwenCausalDecodeStepResult decode_embedding(const std::vector<float> & embedding) {
        ensure_decode_started();
        if (decode_input_kind_ != InputKind::Embedding) {
            throw std::runtime_error("QwenCausalDecodeRuntime decode graph expects tokens");
        }
        if (embedding.size() != static_cast<size_t>(config_.decoder.stack.hidden_size)) {
            throw std::runtime_error("QwenCausalDecodeRuntime decode embedding size mismatch");
        }
        ggml_backend_tensor_set(decode_input_, embedding.data(), 0, embedding.size() * sizeof(float));
        return run_decode_step();
    }

    void start_decode_tokens_batched(
        const runtime::TransformerBatchedKVState & state,
        int64_t required_cache_steps) {
        if (required_cache_steps <= 0) {
            throw std::runtime_error("QwenCausalDecodeRuntime batched decode requires positive cache capacity");
        }
        ensure_batched_decode_token_graph(required_cache_steps, state.batch_size);
        batched_decode_cache_.import_state(state);
    }

    void start_decode_embeddings_batched(
        const runtime::TransformerBatchedKVState & state,
        int64_t required_cache_steps) {
        if (required_cache_steps <= 0) {
            throw std::runtime_error("QwenCausalDecodeRuntime batched decode requires positive cache capacity");
        }
        ensure_batched_decode_embedding_graph(required_cache_steps, state.batch_size);
        batched_decode_cache_.import_state(state);
    }

    QwenCausalDecodeStepResult decode_tokens_batched(const std::vector<int32_t> & tokens) {
        ensure_batched_decode_started();
        if (batched_decode_input_kind_ != InputKind::Token) {
            throw std::runtime_error("QwenCausalDecodeRuntime batched decode graph expects tokens");
        }
        if (tokens.size() != static_cast<size_t>(batched_decode_batch_size_)) {
            throw std::runtime_error("QwenCausalDecodeRuntime batched decode token size mismatch");
        }
        ggml_backend_tensor_set(batched_decode_input_, tokens.data(), 0, tokens.size() * sizeof(int32_t));
        return run_batched_decode_step();
    }

    QwenCausalDecodeStepResult decode_embeddings_batched(
        const std::vector<float> & embeddings,
        int64_t batch_size) {
        ensure_batched_decode_started();
        if (batched_decode_input_kind_ != InputKind::Embedding) {
            throw std::runtime_error("QwenCausalDecodeRuntime batched decode graph expects embeddings");
        }
        if (batch_size != batched_decode_batch_size_) {
            throw std::runtime_error("QwenCausalDecodeRuntime batched decode embedding batch mismatch");
        }
        if (embeddings.size() != static_cast<size_t>(batch_size * config_.decoder.stack.hidden_size)) {
            throw std::runtime_error("QwenCausalDecodeRuntime batched decode embedding size mismatch");
        }
        ggml_backend_tensor_set(batched_decode_input_, embeddings.data(), 0, embeddings.size() * sizeof(float));
        return run_batched_decode_step();
    }

    int64_t decode_cache_steps() const noexcept {
        return decode_cache_steps_;
    }

    int64_t decode_current_end() const noexcept {
        return decode_cache_.current_end();
    }

    int64_t decode_valid_steps() const noexcept {
        return decode_cache_.valid_steps();
    }

    void release_runtime_graphs() {
        release_prefill_graph();
        release_decode_graph();
        release_batched_prefill_graph();
        release_batched_decode_graph();
    }

private:
    enum class InputKind {
        None,
        Token,
        Embedding,
    };

    void ensure_prefill_token_graph(int64_t steps) {
        if (prefill_graph_ != nullptr && prefill_input_kind_ == InputKind::Token && prefill_steps_ == steps) {
            debug::timing_log_scalar(config_.trace_name + ".prefill.graph.build_ms", 0.0);
            debug::trace_log_scalar(config_.trace_name + ".prefill.steps", steps);
            return;
        }
        release_prefill_graph();
        build_prefill_graph(InputKind::Token, steps);
    }

    void ensure_prefill_embedding_graph(int64_t steps) {
        if (prefill_graph_ != nullptr && prefill_input_kind_ == InputKind::Embedding && prefill_steps_ == steps) {
            debug::timing_log_scalar(config_.trace_name + ".prefill.graph.build_ms", 0.0);
            debug::trace_log_scalar(config_.trace_name + ".prefill.steps", steps);
            return;
        }
        release_prefill_graph();
        build_prefill_graph(InputKind::Embedding, steps);
    }

    void build_prefill_graph(InputKind input_kind, int64_t steps) {
        const auto build_start = Clock::now();
        ggml_init_params params{config_.prefill_graph_arena_bytes, nullptr, true};
        prefill_ctx_.reset(ggml_init(params));
        if (prefill_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize QwenCausalDecodeRuntime prefill graph context");
        }
        core::ModuleBuildContext ctx{prefill_ctx_.get(), config_.trace_name.c_str(), backend_type_};
        core::TensorValue x;
        if (input_kind == InputKind::Token) {
            prefill_input_ = ggml_new_tensor_1d(prefill_ctx_.get(), GGML_TYPE_I32, steps);
            x = token_embedding_input(ctx, weights_, config_.decoder, prefill_input_, steps);
        } else {
            auto input = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, steps, config_.decoder.stack.hidden_size}));
            prefill_input_ = input.tensor;
            x = input;
        }
        prefill_positions_ = ggml_new_tensor_1d(prefill_ctx_.get(), GGML_TYPE_I32, steps);
        auto positions = core::wrap_tensor(
            prefill_positions_,
            core::TensorShape::from_dims({steps}),
            GGML_TYPE_I32);
        prefill_attention_mask_ = ggml_new_tensor_4d(prefill_ctx_.get(), GGML_TYPE_F16, steps, steps, 1, 1);
        auto attention_mask = core::wrap_tensor(
            prefill_attention_mask_,
            core::TensorShape::from_dims({1, 1, steps, steps}),
            GGML_TYPE_F16);
        auto decoder_out = build_causal_prefill(ctx, config_, x, positions, weights_, attention_mask);
        prefill_logits_readback_token_ids_ = make_logits_readback_token_ids(prefill_ctx_.get(), config_);
        for (const auto & layer : decoder_out.state.layers) {
            if (!layer.key.has_value() || !layer.value.has_value()) {
                throw std::runtime_error("QwenCausalDecodeRuntime prefill decoder did not return K/V state");
            }
            auto * key = ggml_cpy(
                prefill_ctx_.get(),
                layer.key->tensor,
                ggml_dup_tensor(prefill_ctx_.get(), layer.key->tensor));
            auto * value = ggml_cpy(
                prefill_ctx_.get(),
                layer.value->tensor,
                ggml_dup_tensor(prefill_ctx_.get(), layer.value->tensor));
            ggml_set_output(key);
            ggml_set_output(value);
            prefill_keys_.push_back(key);
            prefill_values_.push_back(value);
        }
        if (config_.output_mode == QwenCausalDecodeOutputMode::Logits) {
            auto logits = decoder_out.logits;
            if (prefill_logits_readback_token_ids_ != nullptr) {
                logits = compact_logits_readback(
                    ctx,
                    config_,
                    logits,
                    wrap_logits_readback_token_ids(prefill_logits_readback_token_ids_, config_));
            }
            prefill_logits_ = ggml_cpy(
                prefill_ctx_.get(),
                logits.tensor,
                ggml_dup_tensor(prefill_ctx_.get(), logits.tensor));
            ggml_set_output(prefill_logits_);
        }
        if (config_.return_hidden || config_.output_mode == QwenCausalDecodeOutputMode::Hidden) {
            prefill_hidden_ = ggml_cpy(
                prefill_ctx_.get(),
                decoder_out.hidden.tensor,
                ggml_dup_tensor(prefill_ctx_.get(), decoder_out.hidden.tensor));
            ggml_set_output(prefill_hidden_);
        }
        prefill_graph_ = ggml_new_graph_custom(prefill_ctx_.get(), 65536, false);
        for (auto * key : prefill_keys_) {
            ggml_build_forward_expand(prefill_graph_, key);
        }
        for (auto * value : prefill_values_) {
            ggml_build_forward_expand(prefill_graph_, value);
        }
        if (prefill_logits_ != nullptr) {
            ggml_build_forward_expand(prefill_graph_, prefill_logits_);
        }
        if (prefill_hidden_ != nullptr) {
            ggml_build_forward_expand(prefill_graph_, prefill_hidden_);
        }
        prefill_gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (prefill_gallocr_ == nullptr ||
            !ggml_gallocr_reserve(prefill_gallocr_, prefill_graph_) ||
            !ggml_gallocr_alloc_graph(prefill_gallocr_, prefill_graph_)) {
            throw std::runtime_error("failed to allocate QwenCausalDecodeRuntime prefill graph");
        }
        const auto positions_values = qwen_position_ids(steps);
        ggml_backend_tensor_set(
            prefill_positions_,
            positions_values.data(),
            0,
            positions_values.size() * sizeof(int32_t));
        const auto mask = qwen_causal_prefill_mask_values(1, steps);
        ggml_backend_tensor_set(
            prefill_attention_mask_,
            mask.data(),
            0,
            mask.size() * sizeof(ggml_fp16_t));
        if (prefill_logits_readback_token_ids_ != nullptr) {
            upload_logits_readback_token_ids(prefill_logits_readback_token_ids_, config_);
        }
        prefill_input_kind_ = input_kind;
        prefill_steps_ = steps;
        debug::timing_log_scalar(
            config_.trace_name + ".prefill.graph.build_ms",
            engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar(config_.trace_name + ".prefill.steps", steps);
    }

    QwenCausalPrefillResult run_prefill() {
        core::set_backend_threads(backend_, threads_);
        const ggml_status status = core::compute_backend_graph(backend_, prefill_graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("QwenCausalDecodeRuntime prefill graph compute failed");
        }
        QwenCausalPrefillResult out;
        if (prefill_logits_ != nullptr) {
            out.logits.resize(static_cast<size_t>(ggml_nelements(prefill_logits_)));
            ggml_backend_tensor_get(prefill_logits_, out.logits.data(), 0, out.logits.size() * sizeof(float));
        }
        if (prefill_hidden_ != nullptr) {
            out.hidden.resize(static_cast<size_t>(ggml_nelements(prefill_hidden_)));
            ggml_backend_tensor_get(prefill_hidden_, out.hidden.data(), 0, out.hidden.size() * sizeof(float));
            round_readback(out.hidden, config_);
        }
        out.state.current_end = prefill_steps_;
        out.state.layers.resize(prefill_keys_.size());
        const size_t layer_values = static_cast<size_t>(
            prefill_steps_ * config_.decoder.stack.num_key_value_heads * config_.decoder.stack.head_dim);
        for (size_t layer = 0; layer < prefill_keys_.size(); ++layer) {
            auto & state = out.state.layers[layer];
            state.valid_steps = prefill_steps_;
            state.key.resize(layer_values);
            state.value.resize(layer_values);
            ggml_backend_tensor_get(prefill_keys_[layer], state.key.data(), 0, state.key.size() * sizeof(float));
            ggml_backend_tensor_get(prefill_values_[layer], state.value.data(), 0, state.value.size() * sizeof(float));
            round_readback(state.key, config_);
            round_readback(state.value, config_);
        }
        return out;
    }

    void ensure_batched_prefill_token_graph(int64_t batch_size, int64_t steps) {
        if (batched_prefill_graph_ != nullptr && batched_prefill_input_kind_ == InputKind::Token &&
            batched_prefill_batch_size_ == batch_size && batched_prefill_steps_ == steps) {
            debug::timing_log_scalar(config_.trace_name + ".batched_prefill.graph.build_ms", 0.0);
            return;
        }
        release_batched_prefill_graph();
        build_batched_prefill_graph(InputKind::Token, batch_size, steps);
    }

    void ensure_batched_prefill_embedding_graph(int64_t batch_size, int64_t steps) {
        if (batched_prefill_graph_ != nullptr && batched_prefill_input_kind_ == InputKind::Embedding &&
            batched_prefill_batch_size_ == batch_size && batched_prefill_steps_ == steps) {
            debug::timing_log_scalar(config_.trace_name + ".batched_prefill.graph.build_ms", 0.0);
            return;
        }
        release_batched_prefill_graph();
        build_batched_prefill_graph(InputKind::Embedding, batch_size, steps);
    }

    void build_batched_prefill_graph(InputKind input_kind, int64_t batch_size, int64_t steps) {
        const auto build_start = Clock::now();
        ggml_init_params params{config_.prefill_graph_arena_bytes, nullptr, true};
        batched_prefill_ctx_.reset(ggml_init(params));
        if (batched_prefill_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize QwenCausalDecodeRuntime batched prefill graph context");
        }
        core::ModuleBuildContext ctx{batched_prefill_ctx_.get(), config_.trace_name.c_str(), backend_type_};
        core::TensorValue x;
        if (input_kind == InputKind::Token) {
            auto input = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({batch_size, steps}));
            batched_prefill_input_ = input.tensor;
            x = token_embedding_input_batched(
                ctx,
                weights_,
                config_.decoder,
                batched_prefill_input_,
                batch_size,
                steps);
        } else {
            auto input = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({batch_size, steps, config_.decoder.stack.hidden_size}));
            batched_prefill_input_ = input.tensor;
            x = input;
        }
        batched_prefill_positions_ = ggml_new_tensor_1d(batched_prefill_ctx_.get(), GGML_TYPE_I32, steps);
        auto positions = core::wrap_tensor(
            batched_prefill_positions_,
            core::TensorShape::from_dims({steps}),
            GGML_TYPE_I32);
        auto attention = core::make_tensor(
            ctx,
            GGML_TYPE_F16,
            core::TensorShape::from_dims({batch_size, 1, steps, steps}));
        batched_prefill_attention_mask_ = attention.tensor;
        auto decoder_out = build_causal_prefill(ctx, config_, x, positions, weights_, attention);
        batched_prefill_logits_readback_token_ids_ =
            make_logits_readback_token_ids(batched_prefill_ctx_.get(), config_);
        for (const auto & layer : decoder_out.state.layers) {
            if (!layer.key.has_value() || !layer.value.has_value()) {
                throw std::runtime_error("QwenCausalDecodeRuntime batched prefill decoder did not return K/V state");
            }
            auto * key = ggml_cpy(
                batched_prefill_ctx_.get(),
                layer.key->tensor,
                ggml_dup_tensor(batched_prefill_ctx_.get(), layer.key->tensor));
            auto * value = ggml_cpy(
                batched_prefill_ctx_.get(),
                layer.value->tensor,
                ggml_dup_tensor(batched_prefill_ctx_.get(), layer.value->tensor));
            ggml_set_output(key);
            ggml_set_output(value);
            batched_prefill_keys_.push_back(key);
            batched_prefill_values_.push_back(value);
        }
        if (config_.output_mode == QwenCausalDecodeOutputMode::Logits) {
            auto logits = decoder_out.logits;
            if (batched_prefill_logits_readback_token_ids_ != nullptr) {
                logits = compact_logits_readback(
                    ctx,
                    config_,
                    logits,
                    wrap_logits_readback_token_ids(batched_prefill_logits_readback_token_ids_, config_));
            }
            batched_prefill_logits_ = ggml_cpy(
                batched_prefill_ctx_.get(),
                logits.tensor,
                ggml_dup_tensor(batched_prefill_ctx_.get(), logits.tensor));
            ggml_set_output(batched_prefill_logits_);
        }
        if (config_.return_hidden || config_.output_mode == QwenCausalDecodeOutputMode::Hidden) {
            batched_prefill_hidden_ = ggml_cpy(
                batched_prefill_ctx_.get(),
                decoder_out.hidden.tensor,
                ggml_dup_tensor(batched_prefill_ctx_.get(), decoder_out.hidden.tensor));
            ggml_set_output(batched_prefill_hidden_);
        }
        batched_prefill_graph_ = ggml_new_graph_custom(batched_prefill_ctx_.get(), 65536, false);
        for (auto * key : batched_prefill_keys_) {
            ggml_build_forward_expand(batched_prefill_graph_, key);
        }
        for (auto * value : batched_prefill_values_) {
            ggml_build_forward_expand(batched_prefill_graph_, value);
        }
        if (batched_prefill_logits_ != nullptr) {
            ggml_build_forward_expand(batched_prefill_graph_, batched_prefill_logits_);
        }
        if (batched_prefill_hidden_ != nullptr) {
            ggml_build_forward_expand(batched_prefill_graph_, batched_prefill_hidden_);
        }
        batched_prefill_gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (batched_prefill_gallocr_ == nullptr ||
            !ggml_gallocr_reserve(batched_prefill_gallocr_, batched_prefill_graph_) ||
            !ggml_gallocr_alloc_graph(batched_prefill_gallocr_, batched_prefill_graph_)) {
            throw std::runtime_error("failed to allocate QwenCausalDecodeRuntime batched prefill graph");
        }
        const auto positions_values = qwen_position_ids(steps);
        ggml_backend_tensor_set(
            batched_prefill_positions_,
            positions_values.data(),
            0,
            positions_values.size() * sizeof(int32_t));
        const auto mask = qwen_causal_prefill_mask_values(batch_size, steps);
        ggml_backend_tensor_set(
            batched_prefill_attention_mask_,
            mask.data(),
            0,
            mask.size() * sizeof(ggml_fp16_t));
        if (batched_prefill_logits_readback_token_ids_ != nullptr) {
            upload_logits_readback_token_ids(batched_prefill_logits_readback_token_ids_, config_);
        }
        batched_prefill_input_kind_ = input_kind;
        batched_prefill_batch_size_ = batch_size;
        batched_prefill_steps_ = steps;
        debug::timing_log_scalar(
            config_.trace_name + ".batched_prefill.graph.build_ms",
            engine::debug::elapsed_ms(build_start, Clock::now()));
    }

    QwenCausalBatchedPrefillResult run_batched_prefill() {
        core::set_backend_threads(backend_, threads_);
        const ggml_status status = core::compute_backend_graph(backend_, batched_prefill_graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("QwenCausalDecodeRuntime batched prefill graph compute failed");
        }
        QwenCausalBatchedPrefillResult out;
        if (batched_prefill_logits_ != nullptr) {
            out.logits.resize(static_cast<size_t>(ggml_nelements(batched_prefill_logits_)));
            ggml_backend_tensor_get(
                batched_prefill_logits_,
                out.logits.data(),
                0,
                out.logits.size() * sizeof(float));
        }
        if (batched_prefill_hidden_ != nullptr) {
            out.hidden.resize(static_cast<size_t>(ggml_nelements(batched_prefill_hidden_)));
            ggml_backend_tensor_get(
                batched_prefill_hidden_,
                out.hidden.data(),
                0,
                out.hidden.size() * sizeof(float));
            round_readback(out.hidden, config_);
        }
        out.state.batch_size = batched_prefill_batch_size_;
        out.state.current_end = batched_prefill_steps_;
        out.state.layers.resize(batched_prefill_keys_.size());
        const size_t layer_values = static_cast<size_t>(
            batched_prefill_batch_size_ *
            batched_prefill_steps_ *
            config_.decoder.stack.num_key_value_heads *
            config_.decoder.stack.head_dim);
        for (size_t layer = 0; layer < batched_prefill_keys_.size(); ++layer) {
            auto & state = out.state.layers[layer];
            state.valid_steps = batched_prefill_steps_;
            state.key.resize(layer_values);
            state.value.resize(layer_values);
            ggml_backend_tensor_get(
                batched_prefill_keys_[layer],
                state.key.data(),
                0,
                state.key.size() * sizeof(float));
            ggml_backend_tensor_get(
                batched_prefill_values_[layer],
                state.value.data(),
                0,
                state.value.size() * sizeof(float));
            round_readback(state.key, config_);
            round_readback(state.value, config_);
        }
        return out;
    }

    void ensure_decode_token_graph(int64_t cache_steps) {
        if (decode_graph_ != nullptr && decode_input_kind_ == InputKind::Token && decode_cache_steps_ >= cache_steps) {
            debug::timing_log_scalar(config_.trace_name + ".decode.graph.build_ms", 0.0);
            debug::trace_log_scalar(config_.trace_name + ".decode.cache_steps", cache_steps);
            return;
        }
        release_decode_graph();
        build_decode_graph(InputKind::Token, cache_steps);
    }

    void ensure_decode_embedding_graph(int64_t cache_steps) {
        if (decode_graph_ != nullptr && decode_input_kind_ == InputKind::Embedding && decode_cache_steps_ >= cache_steps) {
            debug::timing_log_scalar(config_.trace_name + ".decode.graph.build_ms", 0.0);
            debug::trace_log_scalar(config_.trace_name + ".decode.cache_steps", cache_steps);
            return;
        }
        release_decode_graph();
        build_decode_graph(InputKind::Embedding, cache_steps);
    }

    void build_decode_graph(InputKind input_kind, int64_t cache_steps) {
        const auto build_start = Clock::now();
        ggml_init_params params{config_.decode_graph_arena_bytes, nullptr, true};
        decode_ctx_.reset(ggml_init(params));
        if (decode_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize QwenCausalDecodeRuntime decode graph context");
        }
        core::ModuleBuildContext ctx{decode_ctx_.get(), config_.trace_name.c_str(), backend_type_};
        core::TensorValue x;
        if (input_kind == InputKind::Token) {
            decode_input_ = ggml_new_tensor_1d(decode_ctx_.get(), GGML_TYPE_I32, 1);
            x = token_embedding_input(ctx, weights_, config_.decoder, decode_input_, 1);
        } else {
            auto input = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, 1, config_.decoder.stack.hidden_size}));
            decode_input_ = input.tensor;
            x = input;
        }
        decode_positions_ = ggml_new_tensor_1d(decode_ctx_.get(), GGML_TYPE_I32, 1);
        auto positions = core::wrap_tensor(decode_positions_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        decode_cache_slot_ = ggml_new_tensor_1d(decode_ctx_.get(), GGML_TYPE_I32, 1);
        auto cache_slot = core::wrap_tensor(decode_cache_slot_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        decode_attention_mask_ = ggml_new_tensor_4d(decode_ctx_.get(), GGML_TYPE_F16, cache_steps, 1, 1, 1);
        auto attention_mask = core::wrap_tensor(
            decode_attention_mask_,
            core::TensorShape::from_dims({1, 1, 1, cache_steps}),
            GGML_TYPE_F16);
        decode_graph_ = ggml_new_graph_custom(decode_ctx_.get(), 65536, false);
        auto decoder_out = build_causal_decode(
            ctx,
            decode_graph_,
            config_,
            x,
            positions,
            weights_,
            cache_steps,
            attention_mask,
            cache_slot);
        decode_cache_ = std::move(decoder_out.cache);
        decode_logits_readback_token_ids_ = make_logits_readback_token_ids(decode_ctx_.get(), config_);
        if (config_.output_mode == QwenCausalDecodeOutputMode::Logits) {
            auto logits = decoder_out.logits;
            if (decode_logits_readback_token_ids_ != nullptr) {
                logits = compact_logits_readback(
                    ctx,
                    config_,
                    logits,
                    wrap_logits_readback_token_ids(decode_logits_readback_token_ids_, config_));
            }
            decode_logits_ = ggml_cpy(
                decode_ctx_.get(),
                logits.tensor,
                ggml_dup_tensor(decode_ctx_.get(), logits.tensor));
            ggml_set_output(decode_logits_);
        }
        if (config_.return_hidden || config_.output_mode == QwenCausalDecodeOutputMode::Hidden) {
            decode_hidden_ = ggml_cpy(
                decode_ctx_.get(),
                decoder_out.hidden.tensor,
                ggml_dup_tensor(decode_ctx_.get(), decoder_out.hidden.tensor));
            ggml_set_output(decode_hidden_);
        }
        if (decode_logits_ != nullptr) {
            ggml_build_forward_expand(decode_graph_, decode_logits_);
        }
        if (decode_hidden_ != nullptr) {
            ggml_build_forward_expand(decode_graph_, decode_hidden_);
        }
        decode_buffer_ = ggml_backend_alloc_ctx_tensors(decode_ctx_.get(), backend_);
        if (decode_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate QwenCausalDecodeRuntime decode graph");
        }
        if (decode_logits_readback_token_ids_ != nullptr) {
            upload_logits_readback_token_ids(decode_logits_readback_token_ids_, config_);
        }
        decode_attention_mask_values_.assign(
            static_cast<size_t>(cache_steps),
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        decode_cache_steps_ = cache_steps;
        decode_input_kind_ = input_kind;
        debug::timing_log_scalar(
            config_.trace_name + ".decode.graph.build_ms",
            engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar(config_.trace_name + ".decode.cache_steps", cache_steps);
    }

    void ensure_batched_decode_token_graph(int64_t cache_steps, int64_t batch_size) {
        if (batched_decode_graph_ != nullptr && batched_decode_input_kind_ == InputKind::Token &&
            batched_decode_cache_steps_ >= cache_steps && batched_decode_batch_size_ == batch_size) {
            debug::timing_log_scalar(config_.trace_name + ".batched_decode.graph.build_ms", 0.0);
            return;
        }
        release_batched_decode_graph();
        build_batched_decode_graph(InputKind::Token, batch_size, cache_steps);
    }

    void ensure_batched_decode_embedding_graph(int64_t cache_steps, int64_t batch_size) {
        if (batched_decode_graph_ != nullptr && batched_decode_input_kind_ == InputKind::Embedding &&
            batched_decode_cache_steps_ >= cache_steps && batched_decode_batch_size_ == batch_size) {
            debug::timing_log_scalar(config_.trace_name + ".batched_decode.graph.build_ms", 0.0);
            return;
        }
        release_batched_decode_graph();
        build_batched_decode_graph(InputKind::Embedding, batch_size, cache_steps);
    }

    void build_batched_decode_graph(InputKind input_kind, int64_t batch_size, int64_t cache_steps) {
        if (batch_size <= 0) {
            throw std::runtime_error("QwenCausalDecodeRuntime batched decode requires positive batch size");
        }
        if (config_.decoder.stack.runtime.static_cache.update_mode != QwenDecoderStaticCacheUpdateMode::DirectSetRows) {
            throw std::runtime_error("QwenCausalDecodeRuntime batched decode supports only DirectSetRows cache update");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{config_.decode_graph_arena_bytes, nullptr, true};
        batched_decode_ctx_.reset(ggml_init(params));
        if (batched_decode_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize QwenCausalDecodeRuntime batched decode graph context");
        }
        core::ModuleBuildContext ctx{batched_decode_ctx_.get(), config_.trace_name.c_str(), backend_type_};
        core::TensorValue x;
        if (input_kind == InputKind::Token) {
            auto input = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({batch_size, 1}));
            batched_decode_input_ = input.tensor;
            x = token_embedding_input_batched(ctx, weights_, config_.decoder, batched_decode_input_, batch_size, 1);
        } else {
            auto input = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({batch_size, 1, config_.decoder.stack.hidden_size}));
            batched_decode_input_ = input.tensor;
            x = input;
        }
        batched_decode_positions_ = ggml_new_tensor_1d(batched_decode_ctx_.get(), GGML_TYPE_I32, 1);
        auto positions = core::wrap_tensor(
            batched_decode_positions_,
            core::TensorShape::from_dims({1}),
            GGML_TYPE_I32);
        auto slot = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({batch_size}));
        batched_decode_cache_slot_ = slot.tensor;
        auto attention = core::make_tensor(
            ctx,
            GGML_TYPE_F16,
            core::TensorShape::from_dims({batch_size, 1, 1, cache_steps}));
        batched_decode_attention_mask_ = attention.tensor;
        batched_decode_graph_ = ggml_new_graph_custom(batched_decode_ctx_.get(), 65536, false);
        auto decoder_out = build_causal_decode_batched(
            ctx,
            batched_decode_graph_,
            config_,
            x,
            positions,
            weights_,
            cache_steps,
            attention,
            slot);
        batched_decode_cache_ = std::move(decoder_out.cache);
        batched_decode_logits_readback_token_ids_ =
            make_logits_readback_token_ids(batched_decode_ctx_.get(), config_);
        if (config_.output_mode == QwenCausalDecodeOutputMode::Logits) {
            auto logits = decoder_out.logits;
            if (batched_decode_logits_readback_token_ids_ != nullptr) {
                logits = compact_logits_readback(
                    ctx,
                    config_,
                    logits,
                    wrap_logits_readback_token_ids(batched_decode_logits_readback_token_ids_, config_));
            }
            batched_decode_logits_ = ggml_cpy(
                batched_decode_ctx_.get(),
                logits.tensor,
                ggml_dup_tensor(batched_decode_ctx_.get(), logits.tensor));
            ggml_set_output(batched_decode_logits_);
        }
        if (config_.return_hidden || config_.output_mode == QwenCausalDecodeOutputMode::Hidden) {
            batched_decode_hidden_ = ggml_cpy(
                batched_decode_ctx_.get(),
                decoder_out.hidden.tensor,
                ggml_dup_tensor(batched_decode_ctx_.get(), decoder_out.hidden.tensor));
            ggml_set_output(batched_decode_hidden_);
        }
        if (batched_decode_logits_ != nullptr) {
            ggml_build_forward_expand(batched_decode_graph_, batched_decode_logits_);
        }
        if (batched_decode_hidden_ != nullptr) {
            ggml_build_forward_expand(batched_decode_graph_, batched_decode_hidden_);
        }
        batched_decode_buffer_ = ggml_backend_alloc_ctx_tensors(batched_decode_ctx_.get(), backend_);
        if (batched_decode_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate QwenCausalDecodeRuntime batched decode graph");
        }
        if (batched_decode_logits_readback_token_ids_ != nullptr) {
            upload_logits_readback_token_ids(batched_decode_logits_readback_token_ids_, config_);
        }
        batched_decode_attention_mask_values_.assign(
            static_cast<size_t>(batch_size * cache_steps),
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        batched_decode_cache_slots_.resize(static_cast<size_t>(batch_size));
        batched_decode_batch_size_ = batch_size;
        batched_decode_cache_steps_ = cache_steps;
        batched_decode_input_kind_ = input_kind;
        debug::timing_log_scalar(
            config_.trace_name + ".batched_decode.graph.build_ms",
            engine::debug::elapsed_ms(build_start, Clock::now()));
    }

    void ensure_decode_started() const {
        if (decode_graph_ == nullptr) {
            throw std::runtime_error("QwenCausalDecodeRuntime decode graph has not been started");
        }
    }

    void ensure_batched_decode_started() const {
        if (batched_decode_graph_ == nullptr) {
            throw std::runtime_error("QwenCausalDecodeRuntime batched decode graph has not been started");
        }
    }

    QwenCausalDecodeStepResult run_decode_step() {
        if (decode_cache_.valid_steps() >= decode_cache_steps_) {
            throw std::runtime_error("QwenCausalDecodeRuntime decode cache exhausted");
        }
        const int32_t position = static_cast<int32_t>(decode_cache_.current_end());
        ggml_backend_tensor_set(decode_positions_, &position, 0, sizeof(int32_t));
        const int32_t cache_slot = static_cast<int32_t>(decode_cache_.valid_steps());
        ggml_backend_tensor_set(decode_cache_slot_, &cache_slot, 0, sizeof(int32_t));
        write_qwen_cached_step_mask(
            decode_attention_mask_,
            decode_attention_mask_values_,
            decode_cache_steps_,
            decode_cache_.valid_steps(),
            cache_slot);
        core::set_backend_threads(backend_, threads_);
        const ggml_status status = core::compute_backend_graph(backend_, decode_graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("QwenCausalDecodeRuntime decode graph compute failed");
        }
        QwenCausalDecodeStepResult out;
        if (decode_logits_ != nullptr) {
            out.logits.resize(static_cast<size_t>(ggml_nelements(decode_logits_)));
            ggml_backend_tensor_get(decode_logits_, out.logits.data(), 0, out.logits.size() * sizeof(float));
        }
        if (decode_hidden_ != nullptr) {
            out.hidden.resize(static_cast<size_t>(ggml_nelements(decode_hidden_)));
            ggml_backend_tensor_get(decode_hidden_, out.hidden.data(), 0, out.hidden.size() * sizeof(float));
            round_readback(out.hidden, config_);
        }
        decode_cache_.advance_after_direct_append(1);
        return out;
    }

    QwenCausalDecodeStepResult run_batched_decode_step() {
        if (batched_decode_cache_.valid_steps() >= batched_decode_cache_steps_) {
            throw std::runtime_error("QwenCausalDecodeRuntime batched decode cache exhausted");
        }
        const int32_t position = static_cast<int32_t>(batched_decode_cache_.current_end());
        ggml_backend_tensor_set(batched_decode_positions_, &position, 0, sizeof(int32_t));
        const int32_t cache_slot = static_cast<int32_t>(batched_decode_cache_.valid_steps());
        for (int64_t batch = 0; batch < batched_decode_batch_size_; ++batch) {
            batched_decode_cache_slots_[static_cast<size_t>(batch)] =
                static_cast<int32_t>(batch * batched_decode_cache_steps_ + cache_slot);
        }
        ggml_backend_tensor_set(
            batched_decode_cache_slot_,
            batched_decode_cache_slots_.data(),
            0,
            batched_decode_cache_slots_.size() * sizeof(int32_t));
        write_qwen_batched_cached_step_mask(
            batched_decode_attention_mask_,
            batched_decode_attention_mask_values_,
            batched_decode_batch_size_,
            batched_decode_cache_steps_,
            batched_decode_cache_.valid_steps(),
            cache_slot);
        core::set_backend_threads(backend_, threads_);
        const ggml_status status = core::compute_backend_graph(backend_, batched_decode_graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("QwenCausalDecodeRuntime batched decode graph compute failed");
        }
        QwenCausalDecodeStepResult out;
        if (batched_decode_logits_ != nullptr) {
            out.logits.resize(static_cast<size_t>(ggml_nelements(batched_decode_logits_)));
            ggml_backend_tensor_get(
                batched_decode_logits_,
                out.logits.data(),
                0,
                out.logits.size() * sizeof(float));
        }
        if (batched_decode_hidden_ != nullptr) {
            out.hidden.resize(static_cast<size_t>(ggml_nelements(batched_decode_hidden_)));
            ggml_backend_tensor_get(
                batched_decode_hidden_,
                out.hidden.data(),
                0,
                out.hidden.size() * sizeof(float));
            round_readback(out.hidden, config_);
        }
        batched_decode_cache_.advance_after_direct_append(1);
        return out;
    }

    void release_prefill_graph() {
        if (prefill_graph_ != nullptr) {
            core::release_backend_graph_resources(backend_, prefill_graph_);
        }
        if (prefill_gallocr_ != nullptr) {
            ggml_gallocr_free(prefill_gallocr_);
            prefill_gallocr_ = nullptr;
        }
        prefill_ctx_.reset();
        prefill_input_ = nullptr;
        prefill_positions_ = nullptr;
        prefill_attention_mask_ = nullptr;
        prefill_logits_readback_token_ids_ = nullptr;
        prefill_logits_ = nullptr;
        prefill_hidden_ = nullptr;
        prefill_keys_.clear();
        prefill_values_.clear();
        prefill_graph_ = nullptr;
        prefill_steps_ = 0;
        prefill_input_kind_ = InputKind::None;
    }

    void release_batched_prefill_graph() {
        if (batched_prefill_graph_ != nullptr) {
            core::release_backend_graph_resources(backend_, batched_prefill_graph_);
        }
        if (batched_prefill_gallocr_ != nullptr) {
            ggml_gallocr_free(batched_prefill_gallocr_);
            batched_prefill_gallocr_ = nullptr;
        }
        batched_prefill_ctx_.reset();
        batched_prefill_input_ = nullptr;
        batched_prefill_positions_ = nullptr;
        batched_prefill_attention_mask_ = nullptr;
        batched_prefill_logits_readback_token_ids_ = nullptr;
        batched_prefill_logits_ = nullptr;
        batched_prefill_hidden_ = nullptr;
        batched_prefill_keys_.clear();
        batched_prefill_values_.clear();
        batched_prefill_graph_ = nullptr;
        batched_prefill_batch_size_ = 0;
        batched_prefill_steps_ = 0;
        batched_prefill_input_kind_ = InputKind::None;
    }

    void release_decode_graph() {
        if (decode_graph_ != nullptr) {
            core::release_backend_graph_resources(backend_, decode_graph_);
        }
        if (decode_buffer_ != nullptr) {
            ggml_backend_buffer_free(decode_buffer_);
            decode_buffer_ = nullptr;
        }
        decode_ctx_.reset();
        decode_input_ = nullptr;
        decode_positions_ = nullptr;
        decode_cache_slot_ = nullptr;
        decode_attention_mask_ = nullptr;
        decode_logits_readback_token_ids_ = nullptr;
        decode_logits_ = nullptr;
        decode_hidden_ = nullptr;
        decode_graph_ = nullptr;
        decode_cache_ = runtime::TransformerKVCache();
        decode_attention_mask_values_.clear();
        decode_cache_steps_ = 0;
        decode_input_kind_ = InputKind::None;
    }

    void release_batched_decode_graph() {
        if (batched_decode_graph_ != nullptr) {
            core::release_backend_graph_resources(backend_, batched_decode_graph_);
        }
        if (batched_decode_buffer_ != nullptr) {
            ggml_backend_buffer_free(batched_decode_buffer_);
            batched_decode_buffer_ = nullptr;
        }
        batched_decode_ctx_.reset();
        batched_decode_input_ = nullptr;
        batched_decode_positions_ = nullptr;
        batched_decode_cache_slot_ = nullptr;
        batched_decode_attention_mask_ = nullptr;
        batched_decode_logits_readback_token_ids_ = nullptr;
        batched_decode_logits_ = nullptr;
        batched_decode_hidden_ = nullptr;
        batched_decode_graph_ = nullptr;
        batched_decode_cache_ = runtime::TransformerBatchedKVCache();
        batched_decode_attention_mask_values_.clear();
        batched_decode_cache_slots_.clear();
        batched_decode_batch_size_ = 0;
        batched_decode_cache_steps_ = 0;
        batched_decode_input_kind_ = InputKind::None;
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    QwenCausalDecodeRuntimeConfig config_;
    QwenCausalDecodeRuntimeWeights weights_;

    std::unique_ptr<ggml_context, GgmlContextDeleter> prefill_ctx_;
    ggml_tensor * prefill_input_ = nullptr;
    ggml_tensor * prefill_positions_ = nullptr;
    ggml_tensor * prefill_attention_mask_ = nullptr;
    ggml_tensor * prefill_logits_readback_token_ids_ = nullptr;
    ggml_tensor * prefill_logits_ = nullptr;
    ggml_tensor * prefill_hidden_ = nullptr;
    std::vector<ggml_tensor *> prefill_keys_;
    std::vector<ggml_tensor *> prefill_values_;
    ggml_cgraph * prefill_graph_ = nullptr;
    ggml_gallocr_t prefill_gallocr_ = nullptr;
    int64_t prefill_steps_ = 0;
    InputKind prefill_input_kind_ = InputKind::None;

    std::unique_ptr<ggml_context, GgmlContextDeleter> batched_prefill_ctx_;
    ggml_tensor * batched_prefill_input_ = nullptr;
    ggml_tensor * batched_prefill_positions_ = nullptr;
    ggml_tensor * batched_prefill_attention_mask_ = nullptr;
    ggml_tensor * batched_prefill_logits_readback_token_ids_ = nullptr;
    ggml_tensor * batched_prefill_logits_ = nullptr;
    ggml_tensor * batched_prefill_hidden_ = nullptr;
    std::vector<ggml_tensor *> batched_prefill_keys_;
    std::vector<ggml_tensor *> batched_prefill_values_;
    ggml_cgraph * batched_prefill_graph_ = nullptr;
    ggml_gallocr_t batched_prefill_gallocr_ = nullptr;
    int64_t batched_prefill_batch_size_ = 0;
    int64_t batched_prefill_steps_ = 0;
    InputKind batched_prefill_input_kind_ = InputKind::None;

    std::unique_ptr<ggml_context, GgmlContextDeleter> decode_ctx_;
    ggml_tensor * decode_input_ = nullptr;
    ggml_tensor * decode_positions_ = nullptr;
    ggml_tensor * decode_cache_slot_ = nullptr;
    ggml_tensor * decode_attention_mask_ = nullptr;
    ggml_tensor * decode_logits_readback_token_ids_ = nullptr;
    ggml_tensor * decode_logits_ = nullptr;
    ggml_tensor * decode_hidden_ = nullptr;
    ggml_cgraph * decode_graph_ = nullptr;
    ggml_backend_buffer_t decode_buffer_ = nullptr;
    std::vector<ggml_fp16_t> decode_attention_mask_values_;
    runtime::TransformerKVCache decode_cache_;
    int64_t decode_cache_steps_ = 0;
    InputKind decode_input_kind_ = InputKind::None;

    std::unique_ptr<ggml_context, GgmlContextDeleter> batched_decode_ctx_;
    ggml_tensor * batched_decode_input_ = nullptr;
    ggml_tensor * batched_decode_positions_ = nullptr;
    ggml_tensor * batched_decode_cache_slot_ = nullptr;
    ggml_tensor * batched_decode_attention_mask_ = nullptr;
    ggml_tensor * batched_decode_logits_readback_token_ids_ = nullptr;
    ggml_tensor * batched_decode_logits_ = nullptr;
    ggml_tensor * batched_decode_hidden_ = nullptr;
    ggml_cgraph * batched_decode_graph_ = nullptr;
    ggml_backend_buffer_t batched_decode_buffer_ = nullptr;
    std::vector<ggml_fp16_t> batched_decode_attention_mask_values_;
    std::vector<int32_t> batched_decode_cache_slots_;
    runtime::TransformerBatchedKVCache batched_decode_cache_;
    int64_t batched_decode_batch_size_ = 0;
    int64_t batched_decode_cache_steps_ = 0;
    InputKind batched_decode_input_kind_ = InputKind::None;
};

QwenCausalDecodeRuntime::QwenCausalDecodeRuntime(
    core::ExecutionContext & execution,
    QwenCausalDecodeRuntimeConfig config,
    QwenCausalDecodeRuntimeWeights weights)
    : impl_(std::make_unique<Impl>(execution, std::move(config), std::move(weights))) {}

QwenCausalDecodeRuntime::~QwenCausalDecodeRuntime() = default;

QwenCausalPrefillResult QwenCausalDecodeRuntime::prefill_tokens(const std::vector<int32_t> & token_ids) {
    return impl_->prefill_tokens(token_ids);
}

QwenCausalPrefillResult QwenCausalDecodeRuntime::prefill_embeddings(
    const std::vector<float> & embeddings,
    int64_t steps) {
    return impl_->prefill_embeddings(embeddings, steps);
}

QwenCausalBatchedPrefillResult QwenCausalDecodeRuntime::prefill_tokens_batched(
    const std::vector<int32_t> & token_ids,
    int64_t batch_size,
    int64_t steps) {
    return impl_->prefill_tokens_batched(token_ids, batch_size, steps);
}

QwenCausalBatchedPrefillResult QwenCausalDecodeRuntime::prefill_embeddings_batched(
    const std::vector<float> & embeddings,
    int64_t batch_size,
    int64_t steps) {
    return impl_->prefill_embeddings_batched(embeddings, batch_size, steps);
}

void QwenCausalDecodeRuntime::start_decode_tokens(
    const runtime::TransformerKVState & state,
    int64_t required_cache_steps) {
    impl_->start_decode_tokens(state, required_cache_steps);
}

void QwenCausalDecodeRuntime::start_decode_embeddings(
    const runtime::TransformerKVState & state,
    int64_t required_cache_steps) {
    impl_->start_decode_embeddings(state, required_cache_steps);
}

QwenCausalDecodeStepResult QwenCausalDecodeRuntime::decode_token(int32_t token) {
    return impl_->decode_token(token);
}

QwenCausalDecodeStepResult QwenCausalDecodeRuntime::decode_embedding(const std::vector<float> & embedding) {
    return impl_->decode_embedding(embedding);
}

void QwenCausalDecodeRuntime::start_decode_tokens_batched(
    const runtime::TransformerBatchedKVState & state,
    int64_t required_cache_steps) {
    impl_->start_decode_tokens_batched(state, required_cache_steps);
}

void QwenCausalDecodeRuntime::start_decode_embeddings_batched(
    const runtime::TransformerBatchedKVState & state,
    int64_t required_cache_steps) {
    impl_->start_decode_embeddings_batched(state, required_cache_steps);
}

QwenCausalDecodeStepResult QwenCausalDecodeRuntime::decode_tokens_batched(const std::vector<int32_t> & tokens) {
    return impl_->decode_tokens_batched(tokens);
}

QwenCausalDecodeStepResult QwenCausalDecodeRuntime::decode_embeddings_batched(
    const std::vector<float> & embeddings,
    int64_t batch_size) {
    return impl_->decode_embeddings_batched(embeddings, batch_size);
}

int64_t QwenCausalDecodeRuntime::decode_cache_steps() const noexcept {
    return impl_->decode_cache_steps();
}

int64_t QwenCausalDecodeRuntime::decode_current_end() const noexcept {
    return impl_->decode_current_end();
}

int64_t QwenCausalDecodeRuntime::decode_valid_steps() const noexcept {
    return impl_->decode_valid_steps();
}

void QwenCausalDecodeRuntime::release_runtime_graphs() {
    impl_->release_runtime_graphs();
}

}  // namespace engine::modules
