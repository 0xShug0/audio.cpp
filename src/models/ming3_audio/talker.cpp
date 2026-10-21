#include "engine/models/ming3_audio/talker.h"

#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/packed_linear_weights.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/torch_random.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::ming3_audio {

struct Talker::Impl {
    Impl(core::ExecutionContext & execution, std::shared_ptr<const assets::TensorSource> source,
         const TalkerConfig & config)
        : execution(execution), source(std::move(source)), config(config),
          store(execution.backend(), execution.backend_type(), "ming3_audio.talker.weights", 4 * 1024 * 1024),
          flow(this->source, config.flow, execution) {
        namespace binding = modules::binding;
        const auto native = assets::TensorStorageType::Native;
        const auto & llm = config.backbone;
        modules::CausalDecoderRuntimeConfig runtime;
        runtime.trace_name = "ming3_audio.talker.ar";
        runtime.prefill_graph_arena_bytes = 8 * 1024 * 1024;
        runtime.decode_graph_arena_bytes = 8 * 1024 * 1024;
        runtime.evict_cuda_graph_cache_on_release = true;
        runtime.return_hidden = true;
        runtime.readback_round_type = GGML_TYPE_BF16;
        auto & c = runtime.decoder.stack;
        c.hidden_size = llm.hidden_size;
        c.intermediate_size = llm.intermediate_size;
        c.layers = llm.layers;
        c.num_attention_heads = llm.attention_heads;
        c.num_key_value_heads = llm.key_value_heads;
        c.head_dim = llm.hidden_size / llm.attention_heads;
        c.rms_norm_eps = llm.rms_norm_eps;
        c.rope_theta = llm.rope_theta;
        c.use_qk_norm = false;
        c.qkv_layout = modules::DecoderQKVLayout::PackedQKV;
        c.runtime.mlp.mode = modules::DecoderMLPMode::PackedGateUp;
        const bool flash = core::resolve_flash_attention(execution.backend(), c.head_dim, core::AttentionPreference::Auto);
        c.runtime.attention.allow_flash_attention = flash;
        c.runtime.attention.prefill_mode = c.runtime.attention.static_mode = flash
            ? modules::DecoderAttentionMode::FlashGroupedViewKV : modules::DecoderAttentionMode::ManualRepeat;
        c.runtime.static_cache.update_mode = modules::DecoderStaticCacheUpdateMode::DirectSetRows;
        if (flash) c.runtime.static_cache.set_rows_mode = modules::DecoderStaticCacheSetRowsMode::BackendViewOptimized;
        c.activation_cast.enabled = true;
        c.activation_cast.norm_intermediate_type = GGML_TYPE_BF16;
        c.activation_cast.rope_intermediate_type = GGML_TYPE_BF16;
        c.activation_cast.after_input_norm = true;
        c.activation_cast.after_qkv_projection = true;
        c.activation_cast.after_rope = true;
        c.activation_cast.after_attention = true;
        c.activation_cast.after_attention_output = true;
        c.activation_cast.after_residual = true;
        c.activation_cast.after_ffn_norm = true;
        c.activation_cast.after_mlp_projection = true;
        c.activation_cast.after_mlp_silu = true;
        c.activation_cast.after_mlp_mul = true;
        c.activation_cast.after_output = true;
        runtime.decoder.static_cache_type = flash ? GGML_TYPE_F16 : GGML_TYPE_F32;
        runtime.decoder.logits_size = 2;
        runtime.decoder.use_lm_head_bias = true;
        runtime.decoder.lm_head_input_type = (execution.backend_type() == core::BackendType::Vulkan ||
            execution.backend_type() == core::BackendType::Metal) ? GGML_TYPE_F32 : GGML_TYPE_BF16;
        embedding = store.load_tensor(*this->source, "model.embed_tokens.weight", native, {config.vocabulary_size, c.hidden_size});
        speaker_projection = binding::linear_from_source(store, *this->source, "spk_head", native, c.hidden_size, 192, true);
        std::vector<float> frequencies(c.head_dim / 2);
        for (size_t i = 0; i < frequencies.size(); ++i) frequencies[i] = 1.0f / std::pow(c.rope_theta, static_cast<float>(2 * i) / c.head_dim);
        auto rounded = frequencies;
        core::round_f32_to_bf16_in_place(rounded);
        for (size_t i = 0; i < frequencies.size(); ++i) frequencies[i] /= rounded[i];
        const auto factors = store.make_f32(core::TensorShape::from_dims({c.head_dim / 2}), frequencies);
        modules::CausalDecoderRuntimeWeights weights;
        for (int64_t i = 0; i < c.layers; ++i) {
            const auto p = "model.layers." + std::to_string(i);
            modules::DecoderLayerWeights layer;
            layer.input_norm = binding::norm_weight_from_source(store, *this->source, p + ".input_layernorm", c.hidden_size);
            layer.post_norm = binding::norm_weight_from_source(store, *this->source, p + ".post_attention_layernorm", c.hidden_size);
            const auto qkv = modules::PackedLinearWeightsBuilder({c.hidden_size, {
                {p + ".self_attn.q_proj.weight", p + ".self_attn.q_proj.bias", c.hidden_size},
                {p + ".self_attn.k_proj.weight", p + ".self_attn.k_proj.bias", c.num_key_value_heads * c.head_dim},
                {p + ".self_attn.v_proj.weight", p + ".self_attn.v_proj.bias", c.num_key_value_heads * c.head_dim}}, true})
                .build(store, *this->source, native);
            layer.self_attention.qkv_weight = qkv.weight;
            layer.self_attention.qkv_bias = qkv.bias;
            layer.self_attention.out_weight = store.load_tensor(*this->source, p + ".self_attn.o_proj.weight", native,
                {c.hidden_size, c.hidden_size});
            layer.mlp.gate_up_proj = modules::PackedLinearWeightsBuilder({c.hidden_size, {
                {p + ".mlp.gate_proj.weight", std::nullopt, c.intermediate_size},
                {p + ".mlp.up_proj.weight", std::nullopt, c.intermediate_size}}, false}).build(store, *this->source, native);
            layer.mlp.down_proj = binding::linear_from_source(store, *this->source, p + ".mlp.down_proj", native,
                c.hidden_size, c.intermediate_size, false);
            layer.rope_frequency_factors = factors;
            weights.stack.layers.push_back(std::move(layer));
        }
        weights.final_norm = binding::norm_weight_from_source(store, *this->source, "model.norm", c.hidden_size);
        weights.lm_head = binding::linear_from_source(store, *this->source, "stop_head", native, 2, c.hidden_size, true);
        store.upload();
        decoder = std::make_unique<modules::CausalDecoderRuntime>(execution, runtime, std::move(weights));
    }

    ~Impl() { release_prompt(); }

    void release_prompt() {
        core::release_backend_graph_resources(execution.backend(), graph, true);
        plan.reset();
        if (allocator) ggml_gallocr_free(allocator);
        if (context) ggml_free(context);
        graph = nullptr;
        allocator = nullptr;
        context = nullptr;
    }

    std::vector<float> prepare_inputs(const TalkerPrompt & prompt, const std::vector<float> & speaker,
                                      const std::vector<float> & reference_latents) {
        const int64_t steps = prompt.ids.size();
        const auto hidden = config.backbone.hidden_size;
        if (steps == 0 || (!speaker.empty() && (speaker.size() != 192 || prompt.speaker_position < 0))) {
            throw std::runtime_error("Ming3 Audio Talker requires prompt tokens and a valid speaker conditioning slot");
        }
        if (!graph || steps != prompt_steps || has_speaker != !speaker.empty()) {
            release_prompt();
            context = ggml_init({1024 * 1024, nullptr, true});
            core::ModuleBuildContext ctx{context, "ming3_audio.talker.prompt", execution.backend_type()};
            prompt_ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
            ggml_set_input(prompt_ids.tensor);
            auto value = modules::EmbeddingModule({config.vocabulary_size, hidden}).build(ctx, prompt_ids, embedding);
            prompt_output = core::wrap_tensor(ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F32), value.shape);
            ggml_set_output(prompt_output.tensor);
            graph = ggml_new_graph_custom(context, 128, false);
            ggml_build_forward_expand(graph, prompt_output.tensor);
            if (!speaker.empty()) {
                speaker_input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 192}));
                ggml_set_input(speaker_input.tensor);
                const auto rounded = modules::Bf16RoundModule{}.build(ctx, speaker_input);
                speaker_output = modules::Bf16RoundModule{}.build(ctx,
                    modules::LinearModule({192, hidden, true}).build(ctx, rounded, speaker_projection));
                ggml_set_output(speaker_output.tensor);
                ggml_build_forward_expand(graph, speaker_output.tensor);
            }
            allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
            if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) throw std::runtime_error("Ming3 Audio Talker prompt allocation failed");
            core::prepare_host_graph_plan(execution, graph, plan);
            prompt_steps = steps;
            has_speaker = !speaker.empty();
        }
        core::write_tensor_i32(prompt_ids, prompt.ids);
        if (!speaker.empty()) core::write_tensor_f32(speaker_input, speaker);
        if (core::compute_graph(execution, graph, plan, "ming3_audio.talker.prompt") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Ming3 Audio Talker prompt execution failed");
        }
        auto embeddings = core::read_tensor_f32(prompt_output.tensor);
        core::round_f32_to_bf16_in_place(embeddings);
        if (!speaker.empty()) {
            const auto projected = core::read_tensor_f32(speaker_output.tensor);
            std::copy(projected.begin(), projected.end(), embeddings.begin() + prompt.speaker_position * hidden);
        }
        if (!reference_latents.empty()) {
            auto rounded = reference_latents;
            core::round_f32_to_bf16_in_place(rounded);
            const auto projected = flow.aggregate(rounded);
            if (prompt.latent_position < 0 || prompt.latent_position * hidden + projected.size() != embeddings.size()) {
                throw std::runtime_error("Ming3 Audio acoustic reference does not match prompt slots");
            }
            std::copy(projected.begin(), projected.end(), embeddings.begin() + prompt.latent_position * hidden);
        }
        return embeddings;
    }

    std::vector<float> generate(const std::vector<float> & embeddings, const TalkerGenerationOptions & options,
                                uint64_t & offset, const std::vector<float> & initial_history) {
        const auto hidden = config.backbone.hidden_size;
        const auto latent_dim = config.flow.latent_dim;
        const int64_t prompt_steps = embeddings.size() / hidden;
        const int64_t budget = std::min(options.max_tokens, config.context_length - prompt_steps);
        if (budget <= 0) throw std::runtime_error("Ming3 Audio Talker prompt exhausts the checkpoint context");
        const auto policy = sampling::resolve_torch_cuda_sampling_policy(execution.backend_type(),
            execution.config().device, "ming3_audio", "Ming3 Audio Talker");
        const auto pref = decoder->prefill_embeddings_into_decode_cache(embeddings, prompt_steps, prompt_steps + budget);
        modules::CausalDecoderStepResult state{pref.logits, pref.hidden};
        auto history = initial_history.empty() ? std::vector<float>(latent_dim, 0.0f) : initial_history;
        core::round_f32_to_bf16_in_place(history);
        std::vector<float> latents;
        for (int64_t step = 0; step < budget; ++step) {
            const auto noise = sampling::generate_torch_cuda_tensor_iterator_randn(latent_dim, options.seed, offset,
                policy, sampling::TorchRandnPrecision::BFloat16);
            offset += sampling::torch_cuda_tensor_iterator_offset_blocks(latent_dim, policy);
            const auto sde_noise = sampling::generate_torch_cuda_tensor_iterator_randn(10 * latent_dim, options.seed, offset,
                policy, sampling::TorchRandnPrecision::BFloat16);
            offset += sampling::torch_cuda_tensor_iterator_offset_blocks(10 * latent_dim, policy);
            auto patch = flow.sample(state.hidden, history, noise, sde_noise, options.guidance_scale, options.sigma, options.temperature);
            latents.insert(latents.end(), patch.begin(), patch.end());
            core::round_f32_to_bf16_in_place(state.logits);
            std::vector<float> stop{1.0f / (1.0f + std::exp(state.logits[0] - state.logits[1]))};
            core::round_f32_to_bf16_in_place(stop);
            if (step > 3 && stop[0] > 0.5f) break;
            if (step + 1 < budget) state = decoder->decode_embedding(flow.aggregate(patch));
            history = std::move(patch);
        }
        return latents;
    }

    core::ExecutionContext & execution;
    std::shared_ptr<const assets::TensorSource> source;
    TalkerConfig config;
    core::BackendWeightStore store;
    modules::MingFlowRuntime flow;
    std::unique_ptr<modules::CausalDecoderRuntime> decoder;
    core::TensorValue embedding, prompt_ids, prompt_output, speaker_input, speaker_output;
    modules::LinearWeights speaker_projection;
    ggml_context * context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    core::HostGraphPlan plan;
    int64_t prompt_steps = 0;
    bool has_speaker = false;
};

Talker::Talker(core::ExecutionContext & execution, std::shared_ptr<const assets::TensorSource> source,
               const TalkerConfig & config) : impl_(std::make_unique<Impl>(execution, std::move(source), config)) {}
Talker::~Talker() = default;
std::vector<float> Talker::prepare_inputs(const TalkerPrompt & prompt, const std::vector<float> & speaker,
                                         const std::vector<float> & reference_latents) {
    return impl_->prepare_inputs(prompt, speaker, reference_latents);
}
std::vector<float> Talker::generate(const std::vector<float> & embeddings, const TalkerGenerationOptions & options,
                                   uint64_t & random_offset, const std::vector<float> & initial_history) {
    return impl_->generate(embeddings, options, random_offset, initial_history);
}

}  // namespace engine::models::ming3_audio
