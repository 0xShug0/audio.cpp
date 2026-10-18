#include "engine/models/plapre/generator.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/packed_linear_weights.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/hf_sampler.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace engine::models::plapre {

PlapreLlamaConfig PlapreLlamaConfig::from_json(const io::json::Value & value) {
    PlapreLlamaConfig c;
    c.hidden = value.require("hidden_size").as_i64();
    c.intermediate = value.require("intermediate_size").as_i64();
    c.layers = value.require("num_hidden_layers").as_i64();
    c.heads = value.require("num_attention_heads").as_i64();
    c.kv_heads = value.require("num_key_value_heads").as_i64();
    c.head_dim = value.require("head_dim").as_i64();
    c.vocabulary = value.require("vocab_size").as_i64();
    c.context_length = value.require("max_position_embeddings").as_i64();
    c.norm_eps = value.require("rms_norm_eps").as_f32();
    // Official vLLM resolves the legacy root theta, not the stale nested value.
    c.rope_theta = value.require("rope_theta").as_f32();
    if (c.hidden <= 0 || c.intermediate <= 0 || c.layers <= 0 || c.heads <= 0 ||
        c.kv_heads <= 0 || c.head_dim <= 0 || c.hidden != c.heads * c.head_dim ||
        c.heads % c.kv_heads != 0 || c.vocabulary <= 0 || c.context_length <= 0 ||
        value.require("model_type").as_string() != "llama" ||
        value.require("hidden_act").as_string() != "silu" ||
        !value.require("tie_word_embeddings").as_bool() ||
        value.require("attention_bias").as_bool() || value.require("mlp_bias").as_bool()) {
        throw std::runtime_error("unsupported Plapre Llama configuration");
    }
    return c;
}

struct PlapreLlamaGenerator::Impl {
    Impl(std::shared_ptr<const assets::TensorSource> source,
         std::shared_ptr<const assets::TensorSource> speaker_projection, PlapreLlamaConfig config,
         core::ExecutionContext & execution, assets::TensorStorageType storage)
        : source_(std::move(source)), speaker_source_(std::move(speaker_projection)), config_(config), execution_(execution),
          store_(execution.backend(), execution.backend_type(), "plapre.ar.weights", 4 * 1024 * 1024) {
        namespace binding = modules::binding;
        const auto & c = config_;
        weights_.token_embedding = store_.load_tensor(*source_, "model.embed_tokens.weight", storage,
            {c.vocabulary, c.hidden});
        // Python prepares prompt embeddings with a BF16 CPU projection even for F32 AR.
        const auto speaker_weight = speaker_source_->require_tensor("weight",
            assets::TensorStorageType::BF16, {c.hidden, c.speaker_dim});
        speaker_projection_.weight = store_.make_f32(speaker_weight.shape,
            assets::tensor_data_to_f32("speaker_proj.weight", speaker_weight));
        const auto speaker_bias = speaker_source_->require_tensor("bias",
            assets::TensorStorageType::BF16, {c.hidden});
        speaker_projection_.bias = store_.make_f32(speaker_bias.shape,
            assets::tensor_data_to_f32("speaker_proj.bias", speaker_bias));
        for (int64_t i = 0; i < c.layers; ++i) {
            const auto p = "model.layers." + std::to_string(i);
            modules::DecoderLayerWeights layer;
            layer.input_norm = binding::norm_weight_from_source(store_, *source_, p + ".input_layernorm", c.hidden);
            layer.post_norm = binding::norm_weight_from_source(store_, *source_, p + ".post_attention_layernorm", c.hidden);
            layer.self_attention.qkv_weight = modules::PackedLinearWeightsBuilder({c.hidden, {
                {p + ".self_attn.q_proj.weight", std::nullopt, c.heads * c.head_dim},
                {p + ".self_attn.k_proj.weight", std::nullopt, c.kv_heads * c.head_dim},
                {p + ".self_attn.v_proj.weight", std::nullopt, c.kv_heads * c.head_dim}}, false})
                .build(store_, *source_, storage).weight;
            layer.self_attention.out_weight = store_.load_tensor(*source_, p + ".self_attn.o_proj.weight", storage,
                {c.hidden, c.heads * c.head_dim});
            layer.mlp.gate_up_proj = modules::PackedLinearWeightsBuilder({c.hidden, {
                {p + ".mlp.gate_proj.weight", std::nullopt, c.intermediate},
                {p + ".mlp.up_proj.weight", std::nullopt, c.intermediate}}, false})
                .build(store_, *source_, storage);
            layer.mlp.down_proj = binding::linear_from_source(store_, *source_, p + ".mlp.down_proj", storage,
                c.hidden, c.intermediate, false);
            weights_.stack.layers.push_back(std::move(layer));
        }
        weights_.final_norm = binding::norm_weight_from_source(store_, *source_, "model.norm", c.hidden);
        weights_.lm_head = modules::LinearWeights{weights_.token_embedding, std::nullopt};
        store_.upload();

        modules::CausalDecoderRuntimeConfig runtime;
        runtime.trace_name = "plapre.ar";
        runtime.prefill_graph_arena_bytes = 4 * 1024 * 1024;
        runtime.decode_graph_arena_bytes = 4 * 1024 * 1024;
        auto & stack = runtime.decoder.stack;
        stack.hidden_size = c.hidden;
        stack.intermediate_size = c.intermediate;
        stack.layers = c.layers;
        stack.num_attention_heads = c.heads;
        stack.num_key_value_heads = c.kv_heads;
        stack.head_dim = c.head_dim;
        stack.rms_norm_eps = c.norm_eps;
        stack.rope_theta = c.rope_theta;
        stack.rope_type = GGML_ROPE_TYPE_NEOX;
        stack.use_qk_norm = false;
        stack.qkv_layout = modules::DecoderQKVLayout::PackedQKV;
        stack.runtime.mlp.mode = modules::DecoderMLPMode::PackedGateUp;
        const bool flash = execution.backend_type() == core::BackendType::Cuda &&
            core::resolve_flash_attention(execution.backend(), c.head_dim, core::AttentionPreference::Auto);
        flash_attention_ = flash;
        const auto attention = flash
            ? modules::DecoderAttentionMode::FlashGroupedViewKV
            : modules::DecoderAttentionMode::ManualRepeat;
        stack.runtime.attention.prefill_mode = attention;
        stack.runtime.attention.static_mode = attention;
        stack.runtime.static_cache.update_mode = modules::DecoderStaticCacheUpdateMode::DirectSetRows;
        runtime.decoder.logits_size = c.vocabulary;
        runtime.decoder.logits_mode = modules::CausalDecoderLogitsMode::LastStep;
        // CUDA flash already reads F16 K/V; avoid recasting the whole cache per step.
        runtime.decoder.static_cache_type = flash ? GGML_TYPE_F16 : GGML_TYPE_F32;
        if (flash) stack.runtime.static_cache.set_rows_mode = modules::DecoderStaticCacheSetRowsMode::BackendViewOptimized;
        decoder_ = std::make_unique<modules::CausalDecoderRuntime>(execution_, runtime, weights_);
    }

    ~Impl() { release_prompt(); }

    void release_prompt() {
        core::release_backend_graph_resources(execution_.backend(), prompt_graph_);
        prompt_graph_ = nullptr;
        prompt_plan_.reset();
        if (prompt_allocator_) ggml_gallocr_free(prompt_allocator_);
        prompt_allocator_ = nullptr;
        if (prompt_context_) ggml_free(prompt_context_);
        prompt_context_ = nullptr;
    }

    std::vector<float> embed_prompt(const std::vector<int32_t> & prompt, const std::vector<float> & speaker) {
        if (prompt.empty() || speaker.size() != static_cast<size_t>(config_.speaker_dim)) {
            throw std::runtime_error("Plapre requires a nonempty prompt and a 128-element speaker embedding");
        }
        for (const auto id : prompt) {
            if (id < 0 || id >= config_.vocabulary) throw std::runtime_error("Plapre prompt token is outside the vocabulary");
        }
        const int64_t steps = static_cast<int64_t>(prompt.size());
        if (!prompt_graph_ || prompt_steps_ != steps) {
            release_prompt();
            prompt_context_ = ggml_init({1024 * 1024, nullptr, true});
            core::ModuleBuildContext ctx{prompt_context_, "plapre.prompt", execution_.backend_type()};
            prompt_ids_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
            speaker_input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config_.speaker_dim}));
            ggml_set_input(prompt_ids_.tensor);
            ggml_set_input(speaker_input_.tensor);
            const auto tokens = modules::EmbeddingModule({config_.vocabulary, config_.hidden})
                .build(ctx, prompt_ids_, weights_.token_embedding);
            const auto projected = modules::LinearModule({config_.speaker_dim, config_.hidden, true})
                .build(ctx, speaker_input_, speaker_projection_);
            prompt_output_ = modules::ConcatModule({0}).build(ctx, projected, tokens);
            ggml_set_output(prompt_output_.tensor);
            prompt_graph_ = ggml_new_graph_custom(prompt_context_, 128, false);
            ggml_build_forward_expand(prompt_graph_, prompt_output_.tensor);
            prompt_allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
            if (!ggml_gallocr_alloc_graph(prompt_allocator_, prompt_graph_)) {
                throw std::runtime_error("Plapre prompt graph allocation failed");
            }
            core::prepare_host_graph_plan(execution_, prompt_graph_, prompt_plan_);
            prompt_steps_ = steps;
        }
        ggml_backend_tensor_set(prompt_ids_.tensor, prompt.data(), 0, prompt.size() * sizeof(int32_t));
        auto speaker_f32 = speaker;
        core::round_f32_to_bf16_in_place(speaker_f32);
        core::write_tensor_f32(speaker_input_, speaker_f32);
        if (core::compute_graph(execution_, prompt_graph_, prompt_plan_, "plapre.prompt") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Plapre prompt graph execution failed");
        }
        auto embeddings = core::read_tensor_f32(prompt_output_.tensor);
        core::round_f32_to_bf16_in_place(embeddings);
        return embeddings;
    }

    std::vector<int32_t> generate(const std::vector<float> & embeddings, const PlapreSamplingOptions & options) {
        if (embeddings.empty() || embeddings.size() % static_cast<size_t>(config_.hidden) != 0) {
            throw std::runtime_error("Plapre AR input embedding shape is invalid");
        }
        if (options.max_tokens <= 0 || options.min_tokens < 0 || options.min_tokens > options.max_tokens ||
            !std::isfinite(options.temperature) || options.temperature < 0 ||
            !std::isfinite(options.top_p) || options.top_p <= 0 || options.top_p > 1 ||
            options.top_k < -1 || options.stop_ids.empty()) {
            throw std::runtime_error("invalid Plapre sampling options");
        }
        for (const auto id : options.stop_ids) {
            if (id < 0 || id >= config_.vocabulary) throw std::runtime_error("Plapre stop token is outside the vocabulary");
        }
        const auto steps = static_cast<int64_t>(embeddings.size() / config_.hidden);
        if (options.max_tokens > config_.context_length - steps) {
            throw std::runtime_error("Plapre prompt and max_tokens exceed the model context length");
        }
        const auto prefill_start = std::chrono::steady_clock::now();
        // CUDA's single-query vector kernel requires a 256-aligned K/V extent.
        const auto cache_steps = GGML_PAD(steps + options.max_tokens, flash_attention_ ? 256 : 1);
        auto state = decoder_->prefill_embeddings_into_cache(embeddings, steps, cache_steps, 256);
        debug::timing_log_scalar("plapre.ar.prefill_ms", debug::elapsed_ms(prefill_start));
        sampling::HfSamplingOptions sampling;
        sampling.do_sample = options.temperature > 0;
        sampling.temperature = options.temperature;
        sampling.top_p = options.top_p;
        sampling.top_k = std::max<int64_t>(0, options.top_k);
        sampling::HfSampler sampler;
        sampling::HfSamplerScratch scratch;
        scratch.reserve_vocab(static_cast<size_t>(config_.vocabulary));
        std::mt19937 rng(static_cast<uint32_t>(options.seed));
        std::vector<int32_t> tokens;
        double sampling_ms = 0;
        const auto decode_start = std::chrono::steady_clock::now();
        for (int64_t step = 0; step < options.max_tokens; ++step) {
            if (step < options.min_tokens) {
                for (const auto id : options.stop_ids) state.logits[id] = -std::numeric_limits<float>::infinity();
            }
            const auto sample_start = std::chrono::steady_clock::now();
            const auto token = sampler.sample(state.logits, tokens, sampling, scratch, rng, nullptr, "Plapre AR");
            sampling_ms += debug::elapsed_ms(sample_start);
            if (std::find(options.stop_ids.begin(), options.stop_ids.end(), token) != options.stop_ids.end()) break;
            tokens.push_back(token);
            if (step + 1 < options.max_tokens) state = decoder_->decode_token(token);
        }
        debug::timing_log_scalar("plapre.ar.decode_ms", debug::elapsed_ms(decode_start));
        debug::timing_log_scalar("plapre.ar.sampling_ms", sampling_ms);
        debug::trace_log_scalar("plapre.ar.generated_tokens", static_cast<int64_t>(tokens.size()));
        return tokens;
    }

    std::shared_ptr<const assets::TensorSource> source_;
    std::shared_ptr<const assets::TensorSource> speaker_source_;
    PlapreLlamaConfig config_;
    core::ExecutionContext & execution_;
    core::BackendWeightStore store_;
    modules::CausalDecoderRuntimeWeights weights_;
    modules::LinearWeights speaker_projection_;
    std::unique_ptr<modules::CausalDecoderRuntime> decoder_;
    bool flash_attention_ = false;
    ggml_context * prompt_context_ = nullptr;
    ggml_cgraph * prompt_graph_ = nullptr;
    ggml_gallocr_t prompt_allocator_ = nullptr;
    core::HostGraphPlan prompt_plan_;
    core::TensorValue prompt_ids_, speaker_input_, prompt_output_;
    int64_t prompt_steps_ = 0;
};

PlapreLlamaGenerator::PlapreLlamaGenerator(std::shared_ptr<const assets::TensorSource> source,
    std::shared_ptr<const assets::TensorSource> speaker_projection,
    PlapreLlamaConfig config, core::ExecutionContext & execution, assets::TensorStorageType storage)
    : impl_(std::make_unique<Impl>(std::move(source), std::move(speaker_projection), config, execution, storage)) {}
PlapreLlamaGenerator::~PlapreLlamaGenerator() = default;
std::vector<float> PlapreLlamaGenerator::prepare_inputs(const std::vector<int32_t> & prompt, const std::vector<float> & speaker) {
    return impl_->embed_prompt(prompt, speaker);
}
std::vector<int32_t> PlapreLlamaGenerator::generate(const std::vector<float> & embeddings, const PlapreSamplingOptions & options) {
    return impl_->generate(embeddings, options);
}

}  // namespace engine::models::plapre
