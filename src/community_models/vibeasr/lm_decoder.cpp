#include "engine/community_models/vibeasr/lm_decoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/transformers/qwen_causal_decoder.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/kv_cache.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace engine::community_models::vibeasr {
namespace {

namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

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

struct LmLayerWeights {
    core::TensorValue input_norm;
    core::TensorValue q_proj;
    core::TensorValue q_bias;
    core::TensorValue k_proj;
    core::TensorValue k_bias;
    core::TensorValue v_proj;
    core::TensorValue v_bias;
    core::TensorValue o_proj;
    core::TensorValue post_norm;
    core::TensorValue gate_proj;
    core::TensorValue up_proj;
    core::TensorValue down_proj;
};

struct LmWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    std::vector<LmLayerWeights> layers;
    core::TensorValue norm;
    core::TensorValue lm_head;
};

struct PrefillOutput {
    std::vector<float> logits;
    runtime::TransformerKVState kv_state;
};

// I2_S is a whole-tensor quantization whose in-band F32 scale sits after the
// packed codes, which is exactly what ggml_nbytes() accounts for, so the GGUF
// payload goes to the backend byte for byte. Same contract as the encoder's
// load_i8_s_tensor().
core::TensorValue load_i2_s_tensor(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & name,
    const std::vector<int64_t> & expected_shape) {
    const auto metadata = source.require_metadata(name);
    if (metadata.dtype != "i2_s") {
        throw std::runtime_error("VibeASR LM tensor " + name + " is " + metadata.dtype + ", expected i2_s");
    }
    if (metadata.shape != expected_shape) {
        throw std::runtime_error("VibeASR LM tensor " + name + " has an unexpected shape");
    }

    core::TensorShape shape;
    shape.rank = expected_shape.size();
    for (size_t i = 0; i < shape.rank; ++i) {
        shape.dims[i] = expected_shape[i];
    }

    const auto raw = source.require_tensor_data(name);
    return store.make_tensor(shape, GGML_TYPE_I2_S, raw.bytes.data(), raw.bytes.size());
}

modules::QwenDecoderLayerWeights to_qwen_layer_weights(const LmLayerWeights & weights) {
    modules::QwenDecoderLayerWeights out;
    out.input_norm = {weights.input_norm, std::nullopt};
    out.self_attention.q_weight = weights.q_proj;
    out.self_attention.q_bias = weights.q_bias;
    out.self_attention.k_weight = weights.k_proj;
    out.self_attention.k_bias = weights.k_bias;
    out.self_attention.v_weight = weights.v_proj;
    out.self_attention.v_bias = weights.v_bias;
    out.self_attention.out_weight = weights.o_proj;
    out.post_norm = {weights.post_norm, std::nullopt};
    out.mlp.gate_proj = {weights.gate_proj, std::nullopt};
    out.mlp.up_proj = {weights.up_proj, std::nullopt};
    out.mlp.down_proj = {weights.down_proj, std::nullopt};
    return out;
}

// Plain Qwen2: attention biases, no per-head Q/K norms. Nothing here depends on
// the weight type, which is why the framework's decoder runs unmodified on I2_S
// projections -- ggml_mul_mat dispatches on the tensor type.
modules::QwenCausalDecoderConfig make_qwen_decoder_config(const VibeASRLmConfig & config) {
    modules::QwenCausalDecoderConfig out;
    out.stack.hidden_size = config.hidden_size;
    out.stack.num_attention_heads = config.num_attention_heads;
    out.stack.num_key_value_heads = config.num_key_value_heads;
    out.stack.head_dim = config.head_dim;
    out.stack.intermediate_size = config.intermediate_size;
    out.stack.layers = config.num_hidden_layers;
    out.stack.rms_norm_eps = config.rms_norm_eps;
    out.stack.rope_theta = config.rope_theta;
    out.stack.use_qk_norm = false;
    out.stack.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    out.logits_size = config.vocab_size;
    out.logits_mode = modules::QwenCausalDecoderLogitsMode::LastStep;
    return out;
}

modules::QwenCausalDecoderWeights make_qwen_decoder_weights(const LmWeights & weights) {
    modules::QwenCausalDecoderWeights out;
    out.stack.layers.reserve(weights.layers.size());
    for (const auto & layer : weights.layers) {
        out.stack.layers.push_back(to_qwen_layer_weights(layer));
    }
    out.final_norm = {weights.norm, std::nullopt};
    out.lm_head = {weights.lm_head, std::nullopt};
    return out;
}

// Token embeddings with the encoder's speech features written over the
// <|speech_pad|> slots. Doing the overwrite in-graph with ggml_set_rows keeps
// the prompt a single I32 upload instead of a host-side embedding matrix.
core::TensorValue prompt_embeddings(
    core::ModuleBuildContext & ctx,
    const LmWeights & weights,
    const VibeASRLmConfig & config,
    ggml_tensor * token_ids,
    ggml_tensor * speech_embeddings,
    ggml_tensor * speech_positions,
    int64_t prompt_steps,
    int64_t speech_tokens) {
    auto ids = core::wrap_tensor(token_ids, core::TensorShape::from_dims({prompt_steps}), GGML_TYPE_I32);
    auto x = modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                 .build(ctx, ids, weights.token_embedding);
    if (speech_tokens > 0) {
        auto speech = core::wrap_tensor(
            speech_embeddings,
            core::TensorShape::from_dims({speech_tokens, config.hidden_size}),
            GGML_TYPE_F32);
        auto positions = core::wrap_tensor(
            speech_positions,
            core::TensorShape::from_dims({speech_tokens}),
            GGML_TYPE_I64);
        x = core::wrap_tensor(
            ggml_set_rows(ctx.ggml, x.tensor, speech.tensor, positions.tensor),
            x.shape,
            GGML_TYPE_F32);
    }
    return core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, prompt_steps, config.hidden_size}));
}

LmWeights load_weights(
    const assets::TensorSource & source,
    const VibeASRLmConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes) {
    LmWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "vibeasr.lm.weights",
        weight_context_bytes);

    // The embedding table and the output projection are the two tensors VibeASR
    // leaves unternarized -- Q6_K and F16 in the published checkpoint -- so they
    // load through the framework's normal path.
    weights.token_embedding = weights.store->load_tensor(
        source,
        "token_embd.weight",
        assets::TensorStorageType::Native,
        {config.vocab_size, config.hidden_size});

    const int64_t dim = config.head_dim;
    const int64_t q_dim = config.num_attention_heads * dim;
    const int64_t kv_dim = config.num_key_value_heads * dim;
    weights.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        LmLayerWeights w;
        w.input_norm = weights.store->load_f32_tensor(source, prefix + "attn_norm.weight", {config.hidden_size});
        w.q_proj = load_i2_s_tensor(*weights.store, source, prefix + "attn_q.weight", {q_dim, config.hidden_size});
        w.q_bias = weights.store->load_f32_tensor(source, prefix + "attn_q.bias", {q_dim});
        w.k_proj = load_i2_s_tensor(*weights.store, source, prefix + "attn_k.weight", {kv_dim, config.hidden_size});
        w.k_bias = weights.store->load_f32_tensor(source, prefix + "attn_k.bias", {kv_dim});
        w.v_proj = load_i2_s_tensor(*weights.store, source, prefix + "attn_v.weight", {kv_dim, config.hidden_size});
        w.v_bias = weights.store->load_f32_tensor(source, prefix + "attn_v.bias", {kv_dim});
        w.o_proj = load_i2_s_tensor(*weights.store, source, prefix + "attn_output.weight", {config.hidden_size, q_dim});
        w.post_norm = weights.store->load_f32_tensor(source, prefix + "ffn_norm.weight", {config.hidden_size});
        w.gate_proj = load_i2_s_tensor(
            *weights.store, source, prefix + "ffn_gate.weight", {config.intermediate_size, config.hidden_size});
        w.up_proj = load_i2_s_tensor(
            *weights.store, source, prefix + "ffn_up.weight", {config.intermediate_size, config.hidden_size});
        w.down_proj = load_i2_s_tensor(
            *weights.store, source, prefix + "ffn_down.weight", {config.hidden_size, config.intermediate_size});
        weights.layers.push_back(std::move(w));
    }

    weights.norm = weights.store->load_f32_tensor(source, "output_norm.weight", {config.hidden_size});
    weights.lm_head = weights.store->load_tensor(
        source,
        "output.weight",
        assets::TensorStorageType::Native,
        {config.vocab_size, config.hidden_size});
    weights.store->upload();
    return weights;
}

int32_t argmax_index(const std::vector<float> & values) {
    if (values.empty()) {
        throw std::runtime_error("VibeASR LM cannot select from empty logits");
    }
    size_t best = 0;
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return static_cast<int32_t>(best);
}

class LmWeightsRuntime {
public:
    LmWeightsRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        VibeASRLmConfig config,
        core::ExecutionContext & execution,
        size_t weight_context_bytes)
        : source_(std::move(source)),
          config_(std::make_shared<const VibeASRLmConfig>(config)),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          weights_(std::make_shared<LmWeights>(load_weights(
              *source_,
              *config_,
              backend_,
              backend_type_,
              weight_context_bytes))) {}

    const VibeASRLmConfig & config() const noexcept { return *config_; }
    const LmWeights & weights() const noexcept { return *weights_; }
    ggml_backend_t backend() const noexcept { return backend_; }
    core::BackendType backend_type() const noexcept { return backend_type_; }
    int threads() const noexcept { return threads_; }

private:
    std::shared_ptr<const assets::TensorSource> source_;
    std::shared_ptr<const VibeASRLmConfig> config_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const LmWeights> weights_;
};

class PrefillGraph {
public:
    PrefillGraph(
        std::shared_ptr<LmWeightsRuntime> runtime,
        int64_t prompt_steps,
        int64_t speech_tokens,
        size_t graph_arena_bytes)
        : runtime_(std::move(runtime)),
          prompt_steps_(prompt_steps),
          speech_tokens_(speech_tokens) {
        if (prompt_steps_ <= 0) {
            throw std::runtime_error("VibeASR LM prefill requires positive prompt length");
        }
        if (speech_tokens_ < 0 || speech_tokens_ > prompt_steps_) {
            throw std::runtime_error("VibeASR LM prefill speech token count is invalid");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VibeASR LM prefill graph context");
        }
        const auto & config = runtime_->config();
        const auto & weights = runtime_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), "vibeasr.lm.prefill", runtime_->backend_type()};
        token_ids_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
        speech_embeddings_ = ggml_new_tensor_2d(
            ctx_.get(), GGML_TYPE_F32, config.hidden_size, std::max<int64_t>(speech_tokens_, 1));
        speech_positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I64, std::max<int64_t>(speech_tokens_, 1));
        auto x = prompt_embeddings(
            ctx,
            weights,
            config,
            token_ids_,
            speech_embeddings_,
            speech_positions_,
            prompt_steps_,
            speech_tokens_);
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({prompt_steps_}), GGML_TYPE_I32);

        auto decoder_out = modules::QwenCausalDecoderModule(make_qwen_decoder_config(config))
                               .build(ctx, x, positions, make_qwen_decoder_weights(weights));
        for (const auto & layer : decoder_out.state.layers) {
            if (!layer.key.has_value() || !layer.value.has_value()) {
                throw std::runtime_error("VibeASR LM prefill decoder did not return K/V state");
            }
            // Copy K/V out of the graph-allocated intermediates and mark them as
            // outputs so the allocator cannot recycle them before run() reads
            // them back.
            auto * key = ggml_cpy(ctx_.get(), layer.key->tensor, ggml_dup_tensor(ctx_.get(), layer.key->tensor));
            auto * value = ggml_cpy(ctx_.get(), layer.value->tensor, ggml_dup_tensor(ctx_.get(), layer.value->tensor));
            ggml_set_output(key);
            ggml_set_output(value);
            keys_.push_back(key);
            values_.push_back(value);
        }
        logits_ = decoder_out.logits.tensor;
        ggml_set_output(logits_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, logits_);
        for (auto * key : keys_) {
            ggml_build_forward_expand(graph_, key);
        }
        for (auto * value : values_) {
            ggml_build_forward_expand(graph_, value);
        }
        const auto try_alloc = [&]() {
            gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend())));
            return gallocr_ != nullptr &&
                ggml_gallocr_reserve(gallocr_.get(), graph_) &&
                ggml_gallocr_alloc_graph(gallocr_.get(), graph_);
        };
        if (!try_alloc() && (engine::core::trim_backend_pools(runtime_->backend()), !try_alloc())) {
            throw engine::runtime::CapacityError(
                "VibeASR LM prefill graph does not fit in device memory at this size ("
                + std::to_string(prompt_steps_) + " prompt steps, of which "
                + std::to_string(speech_tokens_) + " are speech tokens)");
        }
        position_ids_ = modules::qwen_position_ids(prompt_steps_);
        debug::timing_log_scalar("vibeasr.lm.prefill.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("vibeasr.lm.prefill_prompt_steps", prompt_steps_);
    }

    ~PrefillGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_, true);
    }

    bool matches(const LmWeightsRuntime & runtime, int64_t prompt_steps, int64_t speech_tokens) const {
        return runtime_.get() == &runtime && prompt_steps_ == prompt_steps && speech_tokens_ == speech_tokens;
    }

    PrefillOutput run(
        const std::vector<int32_t> & token_ids,
        const std::vector<float> & speech_embeddings,
        const std::vector<int32_t> & speech_positions) {
        const auto & config = runtime_->config();
        if (static_cast<int64_t>(token_ids.size()) != prompt_steps_) {
            throw std::runtime_error("VibeASR LM prefill token id count mismatch");
        }
        if (static_cast<int64_t>(speech_embeddings.size()) != speech_tokens_ * config.hidden_size) {
            throw std::runtime_error("VibeASR LM prefill speech embedding size mismatch");
        }
        if (static_cast<int64_t>(speech_positions.size()) != speech_tokens_) {
            throw std::runtime_error("VibeASR LM prefill speech position count mismatch");
        }
        // Re-uploaded on every run: leaves are not pinned by the graph allocator.
        ggml_backend_tensor_set(positions_, position_ids_.data(), 0, position_ids_.size() * sizeof(int32_t));
        ggml_backend_tensor_set(token_ids_, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
        if (speech_tokens_ > 0) {
            const std::vector<int64_t> positions(speech_positions.begin(), speech_positions.end());
            ggml_backend_tensor_set(
                speech_embeddings_, speech_embeddings.data(), 0, speech_embeddings.size() * sizeof(float));
            ggml_backend_tensor_set(speech_positions_, positions.data(), 0, positions.size() * sizeof(int64_t));
        }
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        debug::timing_log_scalar("vibeasr.lm.prefill.graph.compute_ms", engine::debug::elapsed_ms(compute_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VibeASR LM prefill graph compute failed");
        }
        PrefillOutput out;
        out.logits.resize(static_cast<size_t>(config.vocab_size));
        ggml_backend_tensor_get(logits_, out.logits.data(), 0, out.logits.size() * sizeof(float));
        out.kv_state.current_end = prompt_steps_;
        out.kv_state.layers.resize(keys_.size());
        const size_t layer_values =
            static_cast<size_t>(prompt_steps_ * config.num_key_value_heads * config.head_dim);
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
    std::shared_ptr<LmWeightsRuntime> runtime_;
    int64_t prompt_steps_ = 0;
    int64_t speech_tokens_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_ids_ = nullptr;
    ggml_tensor * speech_embeddings_ = nullptr;
    ggml_tensor * speech_positions_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    std::vector<int32_t> position_ids_;
    ggml_cgraph * graph_ = nullptr;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
};

class DecodeGraph {
public:
    DecodeGraph(std::shared_ptr<LmWeightsRuntime> runtime, int64_t cache_steps, size_t graph_arena_bytes)
        : runtime_(std::move(runtime)),
          cache_steps_(cache_steps) {
        if (cache_steps_ <= 0) {
            throw std::runtime_error("VibeASR LM decode requires positive cache length");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VibeASR LM decode graph context");
        }
        const auto & config = runtime_->config();
        const auto & weights = runtime_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), "vibeasr.lm.decode", runtime_->backend_type()};
        token_id_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto token_id = core::wrap_tensor(token_id_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto x = modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                     .build(ctx, token_id, weights.token_embedding);
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, 1, config.hidden_size}));
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        cache_slot_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto cache_slot = core::wrap_tensor(cache_slot_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, cache_steps_, 1, 1, 1);
        auto attention_mask = core::wrap_tensor(
            attention_mask_, core::TensorShape::from_dims({1, 1, 1, cache_steps_}), GGML_TYPE_F16);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        auto decoder_out = modules::QwenCausalDecoderModule(make_qwen_decoder_config(config))
                               .build_static_cache_tail(
                                   ctx,
                                   graph_,
                                   x,
                                   positions,
                                   make_qwen_decoder_weights(weights),
                                   cache_steps_,
                                   attention_mask,
                                   cache_slot);
        step_cache_ = std::move(decoder_out.cache);
        logits_ = decoder_out.logits.tensor;
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, logits_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            engine::core::trim_backend_pools(runtime_->backend());
            buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        }
        if (buffer_ == nullptr) {
            throw engine::runtime::CapacityError(
                "VibeASR LM decode graph does not fit in device memory at "
                + std::to_string(cache_steps_) + " cache steps");
        }
        attention_mask_values_.assign(static_cast<size_t>(cache_steps_), ggml_fp32_to_fp16(-INFINITY));
        debug::timing_log_scalar("vibeasr.lm.decode.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("vibeasr.lm.decode_cache_steps", cache_steps_);
    }

    ~DecodeGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_, true);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool can_run(const LmWeightsRuntime & runtime, int64_t required_steps) const {
        return runtime_.get() == &runtime && cache_steps_ >= required_steps;
    }

    void import_state(const runtime::TransformerKVState & state) {
        step_cache_.import_state(state);
    }

    std::vector<float> run_step(int32_t token) {
        const auto & config = runtime_->config();
        if (step_cache_.valid_steps() >= cache_steps_) {
            throw std::runtime_error("VibeASR LM decode cache exhausted");
        }
        ggml_backend_tensor_set(token_id_, &token, 0, sizeof(int32_t));
        const int32_t position = static_cast<int32_t>(step_cache_.current_end());
        ggml_backend_tensor_set(positions_, &position, 0, sizeof(int32_t));
        const int32_t cache_slot = static_cast<int32_t>(step_cache_.valid_steps());
        ggml_backend_tensor_set(cache_slot_, &cache_slot, 0, sizeof(int32_t));
        modules::write_qwen_cached_step_mask(
            attention_mask_,
            attention_mask_values_,
            cache_steps_,
            step_cache_.valid_steps(),
            step_cache_.valid_steps());
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VibeASR LM decode graph compute failed");
        }
        logits_buffer_.resize(static_cast<size_t>(config.vocab_size));
        ggml_backend_tensor_get(logits_, logits_buffer_.data(), 0, logits_buffer_.size() * sizeof(float));
        step_cache_.advance_after_direct_append(1);
        // The caller moves out of this buffer before the next step.
        return std::move(logits_buffer_);
    }

private:
    std::shared_ptr<LmWeightsRuntime> runtime_;
    int64_t cache_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_id_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * cache_slot_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_fp16_t> attention_mask_values_;
    std::vector<float> logits_buffer_;
    runtime::TransformerKVCache step_cache_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

}  // namespace

struct VibeASRLmRuntime::Impl {
    Impl(
        std::shared_ptr<const assets::TensorSource> weights_source,
        const VibeASRLmConfig & config,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes)
        : weights(std::make_shared<LmWeightsRuntime>(
              std::move(weights_source),
              config,
              execution,
              weight_context_bytes)),
          prefill_graph_arena_bytes(prefill_graph_arena_bytes),
          decode_graph_arena_bytes(decode_graph_arena_bytes) {}

    void validate_speech(const VibeASRLmPrompt & prompt, const VibeASRSpeechEmbeddings & speech) const {
        const auto & config = weights->config();
        if (speech.tokens > 0 && speech.hidden_size != config.hidden_size) {
            throw std::runtime_error("VibeASR speech embedding hidden size mismatch");
        }
        if (speech.tokens != static_cast<int64_t>(prompt.speech_positions.size())) {
            throw std::runtime_error("VibeASR speech embedding count does not match the prompt's speech pads");
        }
        if (static_cast<int64_t>(speech.values.size()) != speech.tokens * speech.hidden_size) {
            throw std::runtime_error("VibeASR speech embedding value count mismatch");
        }
        for (const int32_t position : prompt.speech_positions) {
            if (position < 0 || position >= static_cast<int32_t>(prompt.input_ids.size())) {
                throw std::runtime_error("VibeASR speech pad position out of range");
            }
        }
    }

    std::shared_ptr<LmWeightsRuntime> weights;
    size_t prefill_graph_arena_bytes = 0;
    size_t decode_graph_arena_bytes = 0;
    std::unique_ptr<PrefillGraph> prefill_graph;
    std::unique_ptr<DecodeGraph> decode_graph;
};

VibeASRLmRuntime::VibeASRLmRuntime(
    std::shared_ptr<const assets::TensorSource> weights_source,
    const VibeASRLmConfig & config,
    core::ExecutionContext & execution,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes,
    size_t weight_context_bytes)
    : impl_(std::make_unique<Impl>(
          std::move(weights_source),
          config,
          execution,
          prefill_graph_arena_bytes,
          decode_graph_arena_bytes,
          weight_context_bytes)) {}

VibeASRLmRuntime::~VibeASRLmRuntime() = default;

std::vector<int32_t> VibeASRLmRuntime::generate(
    const VibeASRLmPrompt & prompt,
    const VibeASRSpeechEmbeddings & speech,
    const VibeASRGenerationOptions & options) {
    const auto & config = impl_->weights->config();
    if (prompt.input_ids.empty()) {
        throw std::runtime_error("VibeASR LM prompt is empty");
    }
    if (options.max_new_tokens <= 0) {
        throw std::runtime_error("VibeASR max_new_tokens must be positive");
    }
    const int64_t prompt_steps = static_cast<int64_t>(prompt.input_ids.size());
    if (prompt_steps + options.max_new_tokens > config.max_position_embeddings) {
        throw std::runtime_error("VibeASR request exceeds the decoder context length");
    }
    impl_->validate_speech(prompt, speech);

    if (impl_->prefill_graph == nullptr ||
        !impl_->prefill_graph->matches(*impl_->weights, prompt_steps, speech.tokens)) {
        impl_->prefill_graph.reset();
        impl_->prefill_graph = std::make_unique<PrefillGraph>(
            impl_->weights, prompt_steps, speech.tokens, impl_->prefill_graph_arena_bytes);
    }
    auto prefill = impl_->prefill_graph->run(prompt.input_ids, speech.values, prompt.speech_positions);

    const int64_t required_cache_steps = prompt_steps + options.max_new_tokens;
    if (impl_->decode_graph == nullptr || !impl_->decode_graph->can_run(*impl_->weights, required_cache_steps)) {
        impl_->decode_graph.reset();
        impl_->decode_graph =
            std::make_unique<DecodeGraph>(impl_->weights, required_cache_steps, impl_->decode_graph_arena_bytes);
    }
    impl_->decode_graph->import_state(prefill.kv_state);

    const auto is_eos = [&options](int32_t token) {
        return std::find(options.eos_token_ids.begin(), options.eos_token_ids.end(), token) !=
            options.eos_token_ids.end();
    };

    std::vector<int32_t> out;
    std::vector<float> logits = std::move(prefill.logits);
    const auto decode_start = Clock::now();
    for (int64_t step = 0; step < options.max_new_tokens; ++step) {
        const int32_t token = argmax_index(logits);
        if (is_eos(token)) {
            break;
        }
        out.push_back(token);
        logits = impl_->decode_graph->run_step(token);
    }
    debug::timing_log_scalar("vibeasr.lm.decode_total_ms", engine::debug::elapsed_ms(decode_start, Clock::now()));
    return out;
}

}  // namespace engine::community_models::vibeasr
