#include "engine/models/personaplex/lm_runtime.h"

#include "engine/framework/core/module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/transformers/qwen_causal_decoder.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/bounded_static_kv_decode.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace engine::models::personaplex {
namespace {

modules::NormWeights load_rms_alpha(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & name,
    int64_t hidden_size) {
    return {
        store.make_f32(core::TensorShape::from_dims({hidden_size}), source.require_f32(name, {1, 1, hidden_size})),
        std::nullopt,
    };
}

modules::QwenDecoderLayerWeights load_main_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const PersonaPlexLMConfig & config,
    int64_t layer,
    assets::TensorStorageType storage_type) {
    const std::string prefix = "transformer.layers." + std::to_string(layer) + ".";
    modules::QwenDecoderLayerWeights weights;
    weights.input_norm = load_rms_alpha(store, source, prefix + "norm1.alpha", config.hidden_size);
    weights.self_attention.qkv_weight = store.load_tensor(
        source,
        prefix + "self_attn.in_proj_weight",
        storage_type,
        {3 * config.hidden_size, config.hidden_size});
    weights.self_attention.out_weight = store.load_tensor(
        source,
        prefix + "self_attn.out_proj.weight",
        storage_type,
        {config.hidden_size, config.hidden_size});
    weights.post_norm = load_rms_alpha(store, source, prefix + "norm2.alpha", config.hidden_size);
    weights.mlp.gate_up_proj = modules::LinearWeights{
        store.load_tensor(
            source,
            prefix + "gating.linear_in.weight",
            storage_type,
            {2 * config.intermediate_size, config.hidden_size}),
        std::nullopt,
    };
    weights.mlp.down_proj = modules::LinearWeights{
        store.load_tensor(
            source,
            prefix + "gating.linear_out.weight",
            storage_type,
            {config.hidden_size, config.intermediate_size}),
        std::nullopt,
    };
    return weights;
}

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

}  // namespace

modules::QwenCausalDecoderConfig personaplex_lm_decoder_config(
    const PersonaPlexConfig & config,
    core::BackendType backend_type) {
    modules::QwenCausalDecoderConfig out;
    out.stack.hidden_size = config.lm.hidden_size;
    out.stack.num_attention_heads = config.lm.num_attention_heads;
    out.stack.num_key_value_heads = config.lm.num_key_value_heads;
    out.stack.head_dim = config.lm.head_dim;
    out.stack.intermediate_size = config.lm.intermediate_size;
    out.stack.layers = config.lm.num_layers;
    out.stack.rms_norm_eps = config.lm.rms_norm_eps;
    out.stack.rope_theta = config.lm.rope_theta;
    out.stack.rope_type = GGML_ROPE_TYPE_NORMAL;
    out.stack.attention_precision = GGML_PREC_DEFAULT;
    out.stack.projection_precision = GGML_PREC_DEFAULT;
    out.stack.activation_cast.enabled = backend_type != core::BackendType::Vulkan;
    out.stack.activation_cast.type = GGML_TYPE_BF16;
    out.stack.activation_cast.after_input_norm = true;
    out.stack.activation_cast.after_qkv_projection = true;
    out.stack.activation_cast.after_rope = true;
    out.stack.activation_cast.after_attention = true;
    out.stack.activation_cast.after_attention_output = true;
    out.stack.activation_cast.after_residual = true;
    out.stack.activation_cast.after_ffn_norm = true;
    out.stack.activation_cast.after_mlp_projection = true;
    out.stack.activation_cast.after_mlp_silu = true;
    out.stack.activation_cast.after_mlp_mul = true;
    out.stack.activation_cast.after_output = true;
    out.stack.qkv_layout = modules::QwenDecoderQKVLayout::PackedQKV;
    out.stack.use_qk_norm = false;
    out.stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.attention.prefix_mode = modules::QwenDecoderPrefixAttentionMode::FlashWithPrefix;
    out.stack.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    out.stack.runtime.static_cache.set_rows_mode = modules::QwenDecoderStaticCacheSetRowsMode::BackendViewOptimized;
    out.stack.runtime.mlp.mode = modules::QwenDecoderMLPMode::PackedGateUp;
    out.logits_size = config.lm.text_vocab_size;
    out.logits_mode = modules::QwenCausalDecoderLogitsMode::LastStep;
    out.use_lm_head_bias = false;
    return out;
}

std::shared_ptr<const PersonaPlexLMWeights> load_personaplex_lm_weights(
    const PersonaPlexAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t context_bytes,
    assets::TensorStorageType storage_type) {
    if (assets.lm_weights == nullptr) {
        throw std::runtime_error("PersonaPlex LM weights source is not loaded");
    }
    auto weights = std::make_shared<PersonaPlexLMWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "PersonaPlex.lm.weights",
        context_bytes);
    auto & store = *weights->store;
    const auto & source = *assets.lm_weights;
    const auto & config = assets.config.lm;

    weights->text_embedding = store.load_tensor(
        source,
        "text_emb.weight",
        storage_type,
        {config.text_vocab_size + 1, config.hidden_size});
    weights->audio_embeddings.reserve(static_cast<size_t>(config.lm_codebooks));
    for (int64_t codebook = 0; codebook < config.lm_codebooks; ++codebook) {
        weights->audio_embeddings.push_back(store.load_tensor(
            source,
            "emb." + std::to_string(codebook) + ".weight",
            storage_type,
            {config.audio_codebook_size + 1, config.hidden_size}));
    }
    weights->main.stack.layers.reserve(static_cast<size_t>(config.num_layers));
    for (int64_t layer = 0; layer < config.num_layers; ++layer) {
        weights->main.stack.layers.push_back(load_main_layer(store, source, config, layer, storage_type));
    }
    weights->main.final_norm = load_rms_alpha(store, source, "out_norm.alpha", config.hidden_size);
    weights->main.lm_head = modules::LinearWeights{
        store.load_tensor(
            source,
            "text_linear.weight",
            storage_type,
            {config.text_vocab_size, config.hidden_size}),
        std::nullopt,
    };
    weights->store->upload();
    return weights;
}

class PersonaPlexMainStepDecodeGraph {
public:
    PersonaPlexMainStepDecodeGraph(
        std::shared_ptr<const PersonaPlexLMWeights> graph_weights,
        PersonaPlexConfig graph_config,
        ggml_backend_t graph_backend,
        core::BackendType graph_backend_type,
        int graph_threads,
        size_t graph_arena_bytes)
        : weights(std::move(graph_weights)),
          config(std::move(graph_config)),
          backend(graph_backend),
          backend_type(graph_backend_type),
          threads(graph_threads),
          cache_steps_(graph_config.lm.context) {
        if (weights == nullptr) {
            throw std::runtime_error("PersonaPlex main step graph requires LM weights");
        }
        if (backend == nullptr) {
            throw std::runtime_error("PersonaPlex main step graph requires backend");
        }
        if (threads <= 0) {
            throw std::runtime_error("PersonaPlex main step graph requires positive thread count");
        }
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx.reset(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize PersonaPlex main step graph context");
        }
        core::ModuleBuildContext build_ctx{ctx.get(), "personaplex.main_step", backend_type};
        input = core::make_tensor(
                    build_ctx,
                    GGML_TYPE_F32,
                    core::TensorShape::from_dims({1, 1, config.lm.hidden_size}))
                    .tensor;
        text_token_id = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
        ggml_set_input(text_token_id);
        audio_token_ids.reserve(static_cast<size_t>(config.lm.lm_codebooks));
        for (int64_t codebook = 0; codebook < config.lm.lm_codebooks; ++codebook) {
            auto * id = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
            ggml_set_input(id);
            audio_token_ids.push_back(id);
        }
        embedding_scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        ggml_set_input(embedding_scale);
        token_scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        ggml_set_input(token_scale);
        positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
        ggml_set_input(positions);
        auto positions_value = core::wrap_tensor(positions, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        cache_slot = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
        ggml_set_input(cache_slot);
        auto cache_slot_value = core::wrap_tensor(cache_slot, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        attention_mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, config.lm.context, 1, 1, 1);
        ggml_set_input(attention_mask);
        auto attention_mask_value = core::wrap_tensor(
            attention_mask,
            core::TensorShape::from_dims({1, 1, 1, config.lm.context}),
            GGML_TYPE_F16);

        graph = ggml_new_graph_custom(ctx.get(), 65536, false);
        auto token_embedding = build_token_embedding(build_ctx);
        auto embedding_scaled = core::wrap_tensor(
            ggml_mul(ctx.get(), input, embedding_scale),
            core::TensorShape::from_dims({1, 1, config.lm.hidden_size}),
            GGML_TYPE_F32);
        auto token_scaled = core::wrap_tensor(
            ggml_mul(ctx.get(), token_embedding.tensor, token_scale),
            core::TensorShape::from_dims({1, 1, config.lm.hidden_size}),
            GGML_TYPE_F32);
        auto step_input = core::wrap_tensor(
            ggml_add(ctx.get(), embedding_scaled.tensor, token_scaled.tensor),
            core::TensorShape::from_dims({1, 1, config.lm.hidden_size}),
            GGML_TYPE_F32);
        auto decoder_out = modules::QwenCausalDecoderModule(personaplex_lm_decoder_config(config, backend_type))
                               .build_static_cache_tail(
                                   build_ctx,
                                   graph,
                                   step_input,
                                   positions_value,
                                   weights->main,
                                   cache_steps_,
                                   attention_mask_value,
                                   cache_slot_value);
        step_cache = std::move(decoder_out.cache);
        hidden_output = decoder_out.hidden.tensor;
        logits_output = decoder_out.logits.tensor;
        ggml_set_output(hidden_output);
        ggml_set_output(logits_output);
        ggml_build_forward_expand(graph, hidden_output);
        ggml_build_forward_expand(graph, logits_output);
        buffer = ggml_backend_alloc_ctx_tensors(ctx.get(), backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate PersonaPlex main step graph");
        }
        mask_values.assign(
            static_cast<size_t>(cache_steps_),
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
    }

    ~PersonaPlexMainStepDecodeGraph() {
        engine::core::release_backend_graph_resources(backend, graph);
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
    }

    int64_t cache_steps() const noexcept {
        return cache_steps_;
    }

    runtime::TransformerKVState export_state() const {
        return step_cache.export_state();
    }

    void reset_state() {
        step_cache.retain_prefix(0);
    }

    void import_state(const runtime::TransformerKVState & state) {
        step_cache.import_state(state);
    }

    void advance_cache_after_direct_append(int64_t steps) {
        step_cache.advance_after_direct_append(steps);
    }

    PersonaPlexMainStepOutput run_embedding_step(
        const runtime::CachedDecodeStep & step,
        const std::vector<float> & embedding) {
        if (static_cast<int64_t>(embedding.size()) != config.lm.hidden_size) {
            throw std::runtime_error("PersonaPlex main step embedding size mismatch");
        }
        ggml_backend_tensor_set(input, embedding.data(), 0, embedding.size() * sizeof(float));
        const float use_embedding = 1.0F;
        const float skip_tokens = 0.0F;
        ggml_backend_tensor_set(embedding_scale, &use_embedding, 0, sizeof(float));
        ggml_backend_tensor_set(token_scale, &skip_tokens, 0, sizeof(float));
        return run_step(step);
    }

    PersonaPlexMainStepOutput run_token_step(
        const runtime::CachedDecodeStep & step,
        const std::array<int32_t, kPersonaPlexDelayedStreamCount> & tokens) {
        if (static_cast<int64_t>(tokens.size()) != config.lm.lm_codebooks + 1) {
            throw std::runtime_error("PersonaPlex main step token count mismatch");
        }
        ggml_backend_tensor_set(text_token_id, tokens.data(), 0, sizeof(int32_t));
        for (int64_t codebook = 0; codebook < config.lm.lm_codebooks; ++codebook) {
            ggml_backend_tensor_set(
                audio_token_ids[static_cast<size_t>(codebook)],
                tokens.data() + 1 + codebook,
                0,
                sizeof(int32_t));
        }
        const float skip_embedding = 0.0F;
        const float use_tokens = 1.0F;
        ggml_backend_tensor_set(embedding_scale, &skip_embedding, 0, sizeof(float));
        ggml_backend_tensor_set(token_scale, &use_tokens, 0, sizeof(float));
        return run_step(step);
    }

    core::TensorValue build_token_embedding(core::ModuleBuildContext & build_ctx) {
        const auto text_id = core::wrap_tensor(text_token_id, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto out = modules::EmbeddingModule({config.lm.text_vocab_size + 1, config.lm.hidden_size})
                       .build(build_ctx, text_id, weights->text_embedding);
        out = core::reshape_tensor(build_ctx, out, core::TensorShape::from_dims({1, 1, config.lm.hidden_size}));
        for (int64_t codebook = 0; codebook < config.lm.lm_codebooks; ++codebook) {
            const auto audio_id = core::wrap_tensor(
                audio_token_ids[static_cast<size_t>(codebook)],
                core::TensorShape::from_dims({1}),
                GGML_TYPE_I32);
            auto audio = modules::EmbeddingModule({config.lm.audio_codebook_size + 1, config.lm.hidden_size})
                             .build(build_ctx, audio_id, weights->audio_embeddings[static_cast<size_t>(codebook)]);
            audio = core::reshape_tensor(build_ctx, audio, core::TensorShape::from_dims({1, 1, config.lm.hidden_size}));
            out = core::wrap_tensor(
                ggml_add(ctx.get(), out.tensor, audio.tensor),
                out.shape,
                GGML_TYPE_F32);
        }
        return out;
    }

    PersonaPlexMainStepOutput run_step(const runtime::CachedDecodeStep & step) {
        run_step_no_readback(step);
        PersonaPlexMainStepOutput out;
        out.hidden.resize(static_cast<size_t>(config.lm.hidden_size));
        out.text_logits.resize(static_cast<size_t>(config.lm.text_vocab_size));
        ggml_backend_tensor_get_async(backend, hidden_output, out.hidden.data(), 0, out.hidden.size() * sizeof(float));
        ggml_backend_tensor_get_async(
            backend,
            logits_output,
            out.text_logits.data(),
            0,
            out.text_logits.size() * sizeof(float));
        ggml_backend_synchronize(backend);
        return out;
    }

    void run_step_no_readback(const runtime::CachedDecodeStep & step) {
        if (step.cache_steps != cache_steps_) {
            throw std::runtime_error("PersonaPlex main step cache capacity mismatch");
        }
        if (step.valid_steps >= cache_steps_) {
            throw std::runtime_error("PersonaPlex main step exceeded LM context");
        }
        const int32_t position = static_cast<int32_t>(step.position);
        ggml_backend_tensor_set(positions, &position, 0, sizeof(int32_t));
        ggml_backend_tensor_set(cache_slot, &step.cache_slot, 0, sizeof(int32_t));
        modules::write_qwen_cached_step_mask(
            attention_mask,
            mask_values,
            cache_steps_,
            step.valid_steps,
            step.cache_slot);
        core::set_backend_threads(backend, threads);
        const ggml_status status = core::compute_backend_graph(backend, graph, nullptr, "personaplex.main_step");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("PersonaPlex main step graph compute failed");
        }
    }

    std::shared_ptr<const PersonaPlexLMWeights> weights;
    PersonaPlexConfig config;
    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    int threads = 1;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    ggml_tensor * input = nullptr;
    ggml_tensor * text_token_id = nullptr;
    std::vector<ggml_tensor *> audio_token_ids;
    ggml_tensor * embedding_scale = nullptr;
    ggml_tensor * token_scale = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * cache_slot = nullptr;
    ggml_tensor * attention_mask = nullptr;
    ggml_tensor * hidden_output = nullptr;
    ggml_tensor * logits_output = nullptr;
    std::vector<ggml_fp16_t> mask_values;
    runtime::TransformerKVCache step_cache;
    ggml_cgraph * graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    int64_t cache_steps_ = 0;
};

struct PersonaPlexMainStepGraph::Impl {
    Impl(
        std::shared_ptr<const PersonaPlexLMWeights> graph_weights,
        PersonaPlexConfig graph_config,
        ggml_backend_t graph_backend,
        core::BackendType graph_backend_type,
        int graph_threads,
        size_t graph_arena_bytes)
        : weights(std::move(graph_weights)),
          config(std::move(graph_config)),
          backend(graph_backend),
          backend_type(graph_backend_type),
          threads(graph_threads),
          graph_arena_bytes(graph_arena_bytes),
          decode_runtime(runtime::BoundedStaticKVDecodeConfig{
              config.lm.context,
              config.lm.context,
              "PersonaPlex main step decode",
          }) {
        ensure_graph();
    }

    PersonaPlexMainStepOutput run_embedding_step(const std::vector<float> & embedding) {
        ensure_graph();
        const auto step = decode_runtime.next_step();
        auto out = decode_runtime.graph().run_embedding_step(step, embedding);
        decode_runtime.advance_after_direct_append(1);
        ++valid_steps_;
        return out;
    }

    PersonaPlexMainStepOutput run_token_step(
        const std::array<int32_t, kPersonaPlexDelayedStreamCount> & tokens) {
        ensure_graph();
        const auto step = decode_runtime.next_step();
        auto out = decode_runtime.graph().run_token_step(step, tokens);
        decode_runtime.advance_after_direct_append(1);
        ++valid_steps_;
        return out;
    }

    void reset() {
        ensure_graph(true);
    }

    int64_t valid_steps() const noexcept {
        return valid_steps_;
    }

    void ensure_graph(bool force_reset = false) {
        const auto factory = [this](int64_t cache_steps) {
            if (cache_steps != config.lm.context) {
                throw std::runtime_error("PersonaPlex main step decode requires fixed full-context cache");
            }
            return std::make_unique<PersonaPlexMainStepDecodeGraph>(
                weights,
                config,
                backend,
                backend_type,
                threads,
                graph_arena_bytes);
        };
        if (!decode_runtime.has_graph()) {
            decode_runtime.prepare_for_prefill(1, factory);
            decode_runtime.reset_to_empty_cache();
        }
        if (force_reset) {
            decode_runtime.reset_to_empty_cache();
            valid_steps_ = 0;
        }
    }

    std::shared_ptr<const PersonaPlexLMWeights> weights;
    PersonaPlexConfig config;
    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    int threads = 1;
    size_t graph_arena_bytes = 0;
    runtime::CachedDecodeRuntime<PersonaPlexMainStepDecodeGraph, runtime::BoundedStaticKVDecodePolicy> decode_runtime;
    int64_t valid_steps_ = 0;
};

PersonaPlexMainStepGraph::PersonaPlexMainStepGraph(
    std::shared_ptr<const PersonaPlexLMWeights> weights,
    PersonaPlexConfig config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    int threads,
    size_t graph_arena_bytes)
    : impl_(std::make_unique<Impl>(
          std::move(weights),
          std::move(config),
          backend,
          backend_type,
          threads,
          graph_arena_bytes)) {}

PersonaPlexMainStepGraph::~PersonaPlexMainStepGraph() = default;

PersonaPlexMainStepOutput PersonaPlexMainStepGraph::run_embedding_step(const std::vector<float> & embedding) {
    return impl_->run_embedding_step(embedding);
}

PersonaPlexMainStepOutput PersonaPlexMainStepGraph::run_token_step(
    const std::array<int32_t, kPersonaPlexDelayedStreamCount> & tokens) {
    return impl_->run_token_step(tokens);
}

void PersonaPlexMainStepGraph::reset() {
    impl_->reset();
}

int64_t PersonaPlexMainStepGraph::valid_steps() const noexcept {
    return impl_->valid_steps();
}

}  // namespace engine::models::personaplex
