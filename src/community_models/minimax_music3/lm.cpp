#include "engine/community_models/minimax_music3/lm.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/transformers/qwen_causal_decoder.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml.h>
#include "ggml-alloc.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::minimax_music3 {
namespace {

namespace assets = engine::assets;
namespace core = engine::core;
namespace binding = engine::modules::binding;

// The conditional and unconditional CFG sequences always have equal length and decode in
// lockstep, so the decode step runs both as one batch-2 graph: the weights stream once
// per frame instead of once per branch.
constexpr int64_t kBatch = 2;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

modules::QwenDecoderActivationCastPolicy lm_activation_cast_policy(core::BackendType backend_type) {
    modules::QwenDecoderActivationCastPolicy policy;
    if (backend_type == core::BackendType::Cpu || backend_type == core::BackendType::Vulkan ||
        backend_type == core::BackendType::Metal) {
        return policy;
    }
    policy.enabled = true;
    policy.type = GGML_TYPE_BF16;
    policy.after_input_norm = true;
    policy.after_qkv_projection = true;
    policy.after_qk_norm = true;
    policy.after_rope = true;
    policy.after_static_cache_update = true;
    policy.after_attention = true;
    policy.after_attention_output = true;
    policy.after_residual = true;
    policy.after_ffn_norm = true;
    policy.after_mlp_projection = true;
    policy.after_mlp_silu = true;
    policy.after_mlp_mul = true;
    policy.after_output = true;
    return policy;
}

modules::QwenDecoderLayerWeights load_layer_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const MiniMaxMusic3Config & config,
    int64_t layer) {
    const std::string prefix = "model.layers." + std::to_string(layer);
    const auto storage = assets::TensorStorageType::Native;
    modules::QwenDecoderLayerWeights out;
    out.input_norm = binding::norm_weight_from_source(store, source, prefix + ".input_layernorm", config.lm_hidden);
    out.self_attention.q_weight = store.load_tensor(
        source, prefix + ".self_attn.q_proj.weight", storage, {config.lm_heads * config.lm_head_dim, config.lm_hidden});
    out.self_attention.k_weight = store.load_tensor(
        source, prefix + ".self_attn.k_proj.weight", storage, {config.lm_kv_heads * config.lm_head_dim, config.lm_hidden});
    out.self_attention.v_weight = store.load_tensor(
        source, prefix + ".self_attn.v_proj.weight", storage, {config.lm_kv_heads * config.lm_head_dim, config.lm_hidden});
    out.self_attention.out_weight = store.load_tensor(
        source, prefix + ".self_attn.o_proj.weight", storage, {config.lm_hidden, config.lm_heads * config.lm_head_dim});
    out.q_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.q_norm", config.lm_head_dim);
    out.k_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.k_norm", config.lm_head_dim);
    out.post_norm = binding::norm_weight_from_source(store, source, prefix + ".post_attention_layernorm", config.lm_hidden);
    out.mlp.gate_proj = binding::linear_from_source(
        store, source, prefix + ".mlp.gate_proj", storage, config.lm_intermediate, config.lm_hidden, false);
    out.mlp.up_proj = binding::linear_from_source(
        store, source, prefix + ".mlp.up_proj", storage, config.lm_intermediate, config.lm_hidden, false);
    out.mlp.down_proj = binding::linear_from_source(
        store, source, prefix + ".mlp.down_proj", storage, config.lm_hidden, config.lm_intermediate, false);
    return out;
}

modules::QwenDecoderStackConfig make_stack_config(
    const MiniMaxMusic3Config & config,
    core::BackendType backend_type) {
    modules::QwenDecoderStackConfig out;
    out.hidden_size = config.lm_hidden;
    out.num_attention_heads = config.lm_heads;
    out.num_key_value_heads = config.lm_kv_heads;
    out.head_dim = config.lm_head_dim;
    out.intermediate_size = config.lm_intermediate;
    out.layers = config.lm_layers;
    out.rms_norm_eps = config.lm_rms_eps;
    out.rope_theta = config.lm_rope_theta;
    out.rope_type = GGML_ROPE_TYPE_NEOX;
    out.attention_precision = GGML_PREC_F32;
    out.projection_precision = GGML_PREC_DEFAULT;
    out.activation_cast = lm_activation_cast_policy(backend_type);
    out.use_qk_norm = true;
    out.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    return out;
}

modules::QwenDecoderLayerConfig make_layer_config(const modules::QwenDecoderStackConfig & stack) {
    modules::QwenDecoderLayerConfig out;
    out.hidden_size = stack.hidden_size;
    out.num_attention_heads = stack.num_attention_heads;
    out.num_key_value_heads = stack.num_key_value_heads;
    out.head_dim = stack.head_dim;
    out.intermediate_size = stack.intermediate_size;
    out.rms_norm_eps = stack.rms_norm_eps;
    out.rope_theta = stack.rope_theta;
    out.rope_type = stack.rope_type;
    out.attention_precision = stack.attention_precision;
    out.projection_precision = stack.projection_precision;
    out.use_qk_norm = stack.use_qk_norm;
    out.activation_cast = stack.activation_cast;
    out.runtime = stack.runtime;
    return out;
}

modules::QwenCausalDecodeRuntimeConfig make_prefill_runtime_config(
    const modules::QwenDecoderStackConfig & stack,
    const MiniMaxMusic3Config & config,
    core::BackendType backend_type,
    const char * trace_name) {
    modules::QwenCausalDecodeRuntimeConfig out;
    out.trace_name = trace_name;
    out.prefill_graph_arena_bytes = 512ull * 1024ull * 1024ull;
    out.decode_graph_arena_bytes = 256ull * 1024ull * 1024ull;
    out.decoder.stack = stack;
    out.decoder.logits_size = config.lm_logits;
    out.decoder.logits_mode = modules::QwenCausalDecoderLogitsMode::LastStep;
    out.decoder.lm_head_precision = GGML_PREC_DEFAULT;
    if (backend_type == core::BackendType::Vulkan || backend_type == core::BackendType::Metal) {
        out.decoder.lm_head_input_type = GGML_TYPE_F16;
    } else if (backend_type != core::BackendType::Cpu) {
        out.decoder.lm_head_input_type = GGML_TYPE_BF16;
    }
    out.return_hidden = true;
    out.readback_round_type = GGML_TYPE_BF16;
    return out;
}

}  // namespace

struct MiniMaxMusic3LmRuntime::Impl {
    core::ExecutionContext & execution;
    MiniMaxMusic3Config config;
    modules::QwenDecoderStackConfig stack_config;
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    modules::QwenDecoderStackWeights stack;
    modules::NormWeights final_norm;
    core::TensorValue lm_head;
    std::unique_ptr<modules::QwenCausalDecodeRuntime> cond_prefill;
    std::unique_ptr<modules::QwenCausalDecodeRuntime> uncond_prefill;

    // Batched decode state.
    std::unique_ptr<ggml_context, GgmlContextDeleter> decode_ctx;
    ggml_backend_buffer_t decode_buffer = nullptr;
    std::vector<ggml_tensor *> cache_keys;
    std::vector<ggml_tensor *> cache_values;
    ggml_tensor * decode_input = nullptr;      // f32 [batch, 1, hidden]
    ggml_tensor * decode_positions = nullptr;  // i32 [1]
    ggml_tensor * decode_slots = nullptr;      // i32 [batch], flat cache rows b * capacity + slot
    ggml_tensor * decode_mask = nullptr;       // f16 [capacity, 1, 1, 1]
    ggml_tensor * decode_logits = nullptr;
    ggml_tensor * decode_hidden = nullptr;
    ggml_cgraph * decode_graph = nullptr;
    std::vector<ggml_fp16_t> mask_scratch;
    int64_t capacity = 0;
    int64_t valid_steps = 0;

    Impl(
        core::ExecutionContext & execution_context,
        const assets::TensorSource & source,
        const MiniMaxMusic3Config & cfg,
        size_t weight_context_bytes)
        : execution(execution_context),
          config(cfg),
          stack_config(make_stack_config(cfg, execution_context.backend_type())),
          store(std::make_shared<core::BackendWeightStore>(
              execution.backend(),
              execution.backend_type(),
              "minimax_music3.lm.weights",
              weight_context_bytes)) {
        token_embedding = store->load_tensor(
            source,
            "model.embed_tokens.weight",
            assets::TensorStorageType::Native,
            {config.lm_vocab_size, config.lm_hidden});
        stack.layers.reserve(static_cast<size_t>(config.lm_layers));
        for (int64_t layer = 0; layer < config.lm_layers; ++layer) {
            stack.layers.push_back(load_layer_weights(*store, source, config, layer));
        }
        final_norm = binding::norm_weight_from_source(*store, source, "model.norm", config.lm_hidden);
        lm_head = store->load_tensor(
            source,
            "lm_head_sliced.weight",
            assets::TensorStorageType::Native,
            {config.lm_logits, config.lm_hidden});
        store->upload();
        source.release_storage();

        modules::QwenCausalDecodeRuntimeWeights weights;
        weights.token_embedding = token_embedding;
        weights.stack = stack;
        weights.final_norm = final_norm;
        weights.lm_head = modules::LinearWeights{lm_head, std::nullopt};
        cond_prefill = std::make_unique<modules::QwenCausalDecodeRuntime>(
            execution,
            make_prefill_runtime_config(stack_config, config, execution.backend_type(), "minimax_music3.lm.cond"),
            weights);
        uncond_prefill = std::make_unique<modules::QwenCausalDecodeRuntime>(
            execution,
            make_prefill_runtime_config(stack_config, config, execution.backend_type(), "minimax_music3.lm.uncond"),
            weights);
    }

    ~Impl() {
        release_batched_decode();
    }

    void release_batched_decode() {
        if (decode_graph != nullptr) {
            core::release_backend_graph_resources(execution.backend(), decode_graph);
            decode_graph = nullptr;
        }
        if (decode_buffer != nullptr) {
            ggml_backend_buffer_free(decode_buffer);
            decode_buffer = nullptr;
        }
        decode_ctx.reset();
        cache_keys.clear();
        cache_values.clear();
        decode_input = nullptr;
        decode_positions = nullptr;
        decode_slots = nullptr;
        decode_mask = nullptr;
        decode_logits = nullptr;
        decode_hidden = nullptr;
        mask_scratch.clear();
        capacity = 0;
        valid_steps = 0;
    }

    void build_batched_decode(int64_t cache_capacity) {
        release_batched_decode();
        const int64_t hidden = config.lm_hidden;
        decode_ctx.reset(ggml_init({768ull * 1024ull * 1024ull, nullptr, true}));
        if (decode_ctx == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-Music3 batched LM decode context");
        }
        core::ModuleBuildContext ctx{decode_ctx.get(), "minimax_music3.lm.batched", execution.backend_type()};

        auto input = core::make_tensor(
            ctx, GGML_TYPE_F32, core::TensorShape::from_dims({kBatch, 1, hidden}));
        decode_input = input.tensor;
        decode_positions = ggml_new_tensor_1d(decode_ctx.get(), GGML_TYPE_I32, 1);
        decode_slots = ggml_new_tensor_1d(decode_ctx.get(), GGML_TYPE_I32, kBatch);
        decode_mask = ggml_new_tensor_4d(decode_ctx.get(), GGML_TYPE_F16, cache_capacity, 1, 1, 1);
        for (ggml_tensor * tensor : {decode_input, decode_positions, decode_slots, decode_mask}) {
            ggml_set_input(tensor);
        }
        auto positions = core::wrap_tensor(decode_positions, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto slots = core::wrap_tensor(decode_slots, core::TensorShape::from_dims({kBatch}), GGML_TYPE_I32);
        auto mask = core::wrap_tensor(
            decode_mask, core::TensorShape::from_dims({1, 1, 1, cache_capacity}), GGML_TYPE_F16);

        decode_graph = ggml_new_graph_custom(decode_ctx.get(), 65536, false);
        auto x = input;
        auto layer_config = make_layer_config(stack_config);
        // The BF16 activation-cast policy helps the batch-1 GEMV decode path but slows
        // the batch-2 matmuls; the batched graph runs F32 activations.
        layer_config.activation_cast = modules::QwenDecoderActivationCastPolicy{};
        cache_keys.reserve(stack.layers.size());
        cache_values.reserve(stack.layers.size());
        for (const auto & layer : stack.layers) {
            auto cache_key = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({kBatch, cache_capacity, config.lm_kv_heads, config.lm_head_dim}));
            auto cache_value = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({kBatch, cache_capacity, config.lm_kv_heads, config.lm_head_dim}));
            cache_keys.push_back(cache_key.tensor);
            cache_values.push_back(cache_value.tensor);
            auto outs = modules::QwenDecoderLayerModule(layer_config).build_with_static_cache_tail(
                ctx,
                decode_graph,
                x,
                positions,
                layer,
                cache_key,
                cache_value,
                slots,
                mask);
            x = outs.output;
        }
        auto normed = modules::RMSNormModule({hidden, config.lm_rms_eps, true, false})
                          .build(ctx, x, {*final_norm.weight, std::nullopt});
        auto head_input = normed;
        if (execution.backend_type() == core::BackendType::Cuda) {
            head_input = core::wrap_tensor(
                ggml_cast(ctx.ggml, normed.tensor, GGML_TYPE_BF16), normed.shape, GGML_TYPE_BF16);
        }
        auto logits = modules::LinearModule({hidden, config.lm_logits, false, GGML_PREC_DEFAULT})
                          .build(ctx, head_input, {lm_head, std::nullopt});
        decode_logits = ggml_cpy(
            decode_ctx.get(), logits.tensor, ggml_dup_tensor(decode_ctx.get(), logits.tensor));
        decode_hidden = ggml_cpy(
            decode_ctx.get(), normed.tensor, ggml_dup_tensor(decode_ctx.get(), normed.tensor));
        ggml_set_output(decode_logits);
        ggml_set_output(decode_hidden);
        ggml_build_forward_expand(decode_graph, decode_logits);
        ggml_build_forward_expand(decode_graph, decode_hidden);

        decode_buffer = ggml_backend_alloc_ctx_tensors(decode_ctx.get(), execution.backend());
        if (decode_buffer == nullptr) {
            throw std::runtime_error("failed to allocate MiniMax-Music3 batched LM decode graph");
        }
        capacity = cache_capacity;
        valid_steps = 0;
    }

    void import_prefill_state(
        const runtime::TransformerKVState & cond_state,
        const runtime::TransformerKVState & uncond_state) {
        if (cond_state.layers.size() != cache_keys.size() ||
            uncond_state.layers.size() != cache_keys.size()) {
            throw std::runtime_error("MiniMax-Music3 batched LM state layer count mismatch");
        }
        if (cond_state.current_end != uncond_state.current_end) {
            throw std::runtime_error("MiniMax-Music3 batched LM requires aligned CFG prompt lengths");
        }
        const int64_t steps = cond_state.current_end;
        const size_t row_floats = static_cast<size_t>(config.lm_kv_heads * config.lm_head_dim);
        const size_t branch_bytes = static_cast<size_t>(capacity) * row_floats * sizeof(float);
        for (size_t layer = 0; layer < cache_keys.size(); ++layer) {
            for (int64_t branch = 0; branch < kBatch; ++branch) {
                const auto & state =
                    branch == 0 ? cond_state.layers[layer] : uncond_state.layers[layer];
                if (state.key.size() != static_cast<size_t>(steps) * row_floats ||
                    state.value.size() != static_cast<size_t>(steps) * row_floats) {
                    throw std::runtime_error("MiniMax-Music3 batched LM state size mismatch");
                }
                ggml_backend_tensor_set(
                    cache_keys[layer],
                    state.key.data(),
                    static_cast<size_t>(branch) * branch_bytes,
                    state.key.size() * sizeof(float));
                ggml_backend_tensor_set(
                    cache_values[layer],
                    state.value.data(),
                    static_cast<size_t>(branch) * branch_bytes,
                    state.value.size() * sizeof(float));
            }
        }
        valid_steps = steps;
    }

    MiniMaxMusic3LmRuntime::StepResult decode(const std::vector<float> & embedding) {
        const int64_t hidden = config.lm_hidden;
        if (decode_graph == nullptr) {
            throw std::runtime_error("MiniMax-Music3 batched LM decode has not been started");
        }
        if (embedding.size() != static_cast<size_t>(hidden)) {
            throw std::runtime_error("MiniMax-Music3 batched LM embedding size mismatch");
        }
        if (valid_steps >= capacity) {
            throw std::runtime_error("MiniMax-Music3 batched LM cache exhausted");
        }
        // Both CFG branches consume the same frame feedback embedding.
        for (int64_t branch = 0; branch < kBatch; ++branch) {
            ggml_backend_tensor_set(
                decode_input,
                embedding.data(),
                static_cast<size_t>(branch * hidden) * sizeof(float),
                embedding.size() * sizeof(float));
        }
        const int32_t position = static_cast<int32_t>(valid_steps);
        ggml_backend_tensor_set(decode_positions, &position, 0, sizeof(int32_t));
        const int32_t slots[kBatch] = {
            static_cast<int32_t>(valid_steps),
            static_cast<int32_t>(capacity + valid_steps),
        };
        ggml_backend_tensor_set(decode_slots, slots, 0, sizeof(slots));
        modules::write_qwen_cached_step_mask(decode_mask, mask_scratch, capacity, valid_steps, valid_steps);

        core::set_backend_threads(execution.backend(), std::max(1, execution.config().threads));
        const ggml_status status = core::compute_backend_graph(execution.backend(), decode_graph);
        ggml_backend_synchronize(execution.backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax-Music3 batched LM decode graph compute failed");
        }

        MiniMaxMusic3LmRuntime::StepResult out;
        out.cond_logits.resize(static_cast<size_t>(config.lm_logits));
        out.uncond_logits.resize(static_cast<size_t>(config.lm_logits));
        ggml_backend_tensor_get(
            decode_logits, out.cond_logits.data(), 0, out.cond_logits.size() * sizeof(float));
        ggml_backend_tensor_get(
            decode_logits,
            out.uncond_logits.data(),
            static_cast<size_t>(config.lm_logits) * sizeof(float),
            out.uncond_logits.size() * sizeof(float));
        out.last_hidden.resize(static_cast<size_t>(kBatch * hidden));
        ggml_backend_tensor_get(
            decode_hidden, out.last_hidden.data(), 0, out.last_hidden.size() * sizeof(float));
        core::round_f32_to_bf16_in_place(out.last_hidden);
        ++valid_steps;
        return out;
    }
};

MiniMaxMusic3LmRuntime::MiniMaxMusic3LmRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const assets::TensorSource> source,
    const MiniMaxMusic3Config & config,
    size_t weight_context_bytes) {
    if (source == nullptr) {
        throw std::runtime_error("MiniMax-Music3 LM tensor source is missing");
    }
    impl_ = std::make_unique<Impl>(execution, *source, config, weight_context_bytes);
}

MiniMaxMusic3LmRuntime::~MiniMaxMusic3LmRuntime() = default;

MiniMaxMusic3LmRuntime::StepResult MiniMaxMusic3LmRuntime::prefill(
    const std::vector<int32_t> & cond_ids,
    const std::vector<int32_t> & uncond_ids,
    int64_t required_cache_steps) {
    if (cond_ids.empty() || cond_ids.size() != uncond_ids.size()) {
        throw std::runtime_error("MiniMax-Music3 LM prompt id pair is invalid");
    }
    auto & impl = *impl_;
    StepResult out;
    auto cond = impl.cond_prefill->prefill_tokens(cond_ids);
    auto uncond = impl.uncond_prefill->prefill_tokens(uncond_ids);
    impl.cond_prefill->release_runtime_graphs();
    impl.uncond_prefill->release_runtime_graphs();
    impl.build_batched_decode(required_cache_steps);
    impl.import_prefill_state(cond.state, uncond.state);
    out.cond_logits = std::move(cond.logits);
    out.uncond_logits = std::move(uncond.logits);
    out.last_hidden.reserve(static_cast<size_t>(2 * impl.config.lm_hidden));
    out.last_hidden.insert(out.last_hidden.end(), cond.hidden.begin(), cond.hidden.end());
    out.last_hidden.insert(out.last_hidden.end(), uncond.hidden.begin(), uncond.hidden.end());
    return out;
}

MiniMaxMusic3LmRuntime::StepResult MiniMaxMusic3LmRuntime::decode_embedding(const std::vector<float> & embedding) {
    return impl_->decode(embedding);
}

core::TensorValue MiniMaxMusic3LmRuntime::token_embedding() const {
    return impl_->token_embedding;
}

void MiniMaxMusic3LmRuntime::release_runtime_graphs() {
    if (impl_ != nullptr) {
        if (impl_->cond_prefill != nullptr) {
            impl_->cond_prefill->release_runtime_graphs();
        }
        if (impl_->uncond_prefill != nullptr) {
            impl_->uncond_prefill->release_runtime_graphs();
        }
        impl_->release_batched_decode();
    }
}

}  // namespace engine::models::minimax_music3
