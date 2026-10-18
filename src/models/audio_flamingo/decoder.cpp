#include "engine/models/audio_flamingo/decoder.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/hf_sampler.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace engine::models::audio_flamingo {
namespace {
namespace binding = modules::binding;
constexpr int64_t kLookupSteps = 256;
constexpr int64_t kPrefillSteps = 1024;
constexpr size_t kContextBytes = 4 * 1024 * 1024;

struct ContextDeleter {
    void operator()(ggml_context * ctx) const { ggml_free(ctx); }
};
struct AllocatorDeleter {
    void operator()(ggml_gallocr_t allocator) const { ggml_gallocr_free(allocator); }
};
}  // namespace

struct AudioFlamingoQwen2DecoderRuntime::Impl {
    Impl(std::shared_ptr<const AudioFlamingoAssets> assets, core::ExecutionContext & execution,
        assets::TensorStorageType storage)
        : assets_(std::move(assets)), execution_(execution),
          store_(execution.backend(), execution.backend_type(), "audio_flamingo.decoder.weights", kContextBytes) {
        const auto & c = assets_->config.text_decoder;
        const auto & source = *assets_->model_weights;
        bool packed_gate_up = true;
        bool packed_qkv = true;
        if (storage == assets::TensorStorageType::Native) {
            for (int64_t i = 0; i < c.num_hidden_layers; ++i) {
                const std::string p = "language_model.model.layers." + std::to_string(i) + ".mlp.";
                packed_gate_up = packed_gate_up &&
                    source.require_metadata(p + "gate_proj.weight").dtype ==
                    source.require_metadata(p + "up_proj.weight").dtype;
                const std::string a = "language_model.model.layers." + std::to_string(i) + ".self_attn.";
                const auto q_type = source.require_metadata(a + "q_proj.weight").dtype;
                packed_qkv = packed_qkv && q_type == source.require_metadata(a + "k_proj.weight").dtype &&
                    q_type == source.require_metadata(a + "v_proj.weight").dtype;
            }
        }
        modules::CausalDecoderRuntimeWeights weights;
        weights.token_embedding = store_.load_tensor(source, "language_model.model.embed_tokens.weight",
            storage, {c.vocab_size, c.hidden_size});
        for (int64_t i = 0; i < c.num_hidden_layers; ++i) {
            const std::string p = "language_model.model.layers." + std::to_string(i);
            modules::DecoderLayerWeights layer;
            layer.input_norm = binding::norm_weight_from_source(store_, source, p + ".input_layernorm", c.hidden_size);
            layer.post_norm = binding::norm_weight_from_source(store_, source, p + ".post_attention_layernorm", c.hidden_size);
            if (packed_qkv) {
                const int64_t q_rows = c.num_attention_heads * c.head_dim;
                const int64_t kv_rows = c.num_key_value_heads * c.head_dim;
                auto q = source.require_tensor(p + ".self_attn.q_proj.weight", storage, {q_rows, c.hidden_size});
                const auto k = source.require_tensor(p + ".self_attn.k_proj.weight", storage, {kv_rows, c.hidden_size});
                const auto v = source.require_tensor(p + ".self_attn.v_proj.weight", storage, {kv_rows, c.hidden_size});
                q.bytes.insert(q.bytes.end(), k.bytes.begin(), k.bytes.end());
                q.bytes.insert(q.bytes.end(), v.bytes.begin(), v.bytes.end());
                layer.self_attention.qkv_weight = store_.make_tensor(
                    core::TensorShape::from_dims({q_rows + 2 * kv_rows, c.hidden_size}),
                    q.type, q.bytes.data(), q.bytes.size());
                auto bias = source.require_f32(p + ".self_attn.q_proj.bias");
                const auto k_bias = source.require_f32(p + ".self_attn.k_proj.bias");
                const auto v_bias = source.require_f32(p + ".self_attn.v_proj.bias");
                bias.insert(bias.end(), k_bias.begin(), k_bias.end());
                bias.insert(bias.end(), v_bias.begin(), v_bias.end());
                layer.self_attention.qkv_bias = store_.make_tensor(
                    core::TensorShape::from_dims({q_rows + 2 * kv_rows}), GGML_TYPE_F32,
                    bias.data(), bias.size() * sizeof(float));
            } else {
                layer.self_attention.q_weight = store_.load_tensor(source, p + ".self_attn.q_proj.weight",
                    storage, {c.num_attention_heads * c.head_dim, c.hidden_size});
                layer.self_attention.k_weight = store_.load_tensor(source, p + ".self_attn.k_proj.weight",
                    storage, {c.num_key_value_heads * c.head_dim, c.hidden_size});
                layer.self_attention.v_weight = store_.load_tensor(source, p + ".self_attn.v_proj.weight",
                    storage, {c.num_key_value_heads * c.head_dim, c.hidden_size});
                layer.self_attention.q_bias = store_.load_f32_tensor(source, p + ".self_attn.q_proj.bias", {c.num_attention_heads * c.head_dim});
                layer.self_attention.k_bias = store_.load_f32_tensor(source, p + ".self_attn.k_proj.bias", {c.num_key_value_heads * c.head_dim});
                layer.self_attention.v_bias = store_.load_f32_tensor(source, p + ".self_attn.v_proj.bias", {c.num_key_value_heads * c.head_dim});
            }
            layer.self_attention.out_weight = store_.load_tensor(source, p + ".self_attn.o_proj.weight",
                storage, {c.hidden_size, c.num_attention_heads * c.head_dim});
            if (packed_gate_up) {
                auto gate = source.require_tensor(p + ".mlp.gate_proj.weight", storage,
                    {c.intermediate_size, c.hidden_size});
                const auto up = source.require_tensor(p + ".mlp.up_proj.weight", storage,
                    {c.intermediate_size, c.hidden_size});
                gate.bytes.insert(gate.bytes.end(), up.bytes.begin(), up.bytes.end());
                layer.mlp.gate_up_proj = modules::LinearWeights{
                    store_.make_tensor(core::TensorShape::from_dims({2 * c.intermediate_size, c.hidden_size}),
                        gate.type, gate.bytes.data(), gate.bytes.size()), std::nullopt};
            } else {
                layer.mlp.gate_proj = binding::linear_from_source(store_, source, p + ".mlp.gate_proj",
                    storage, c.intermediate_size, c.hidden_size, false);
                layer.mlp.up_proj = binding::linear_from_source(store_, source, p + ".mlp.up_proj",
                    storage, c.intermediate_size, c.hidden_size, false);
            }
            layer.mlp.down_proj = binding::linear_from_source(store_, source, p + ".mlp.down_proj",
                storage, c.hidden_size, c.intermediate_size, false);
            weights.stack.layers.push_back(std::move(layer));
        }
        weights.final_norm = binding::norm_weight_from_source(store_, source, "language_model.model.norm", c.hidden_size);
        weights.lm_head = binding::linear_from_source(store_, source, "language_model.lm_head",
            storage, c.vocab_size, c.hidden_size, false);
        store_.upload();

        modules::CausalDecoderRuntimeConfig config;
        config.trace_name = "audio_flamingo.decoder";
        config.prefill_graph_arena_bytes = kContextBytes;
        config.decode_graph_arena_bytes = kContextBytes;
        config.decoder.logits_size = c.vocab_size;
        config.decoder.static_cache_type = GGML_TYPE_F16;
        auto & stack = config.decoder.stack;
        stack.hidden_size = c.hidden_size;
        stack.num_attention_heads = c.num_attention_heads;
        stack.num_key_value_heads = c.num_key_value_heads;
        stack.head_dim = c.head_dim;
        stack.intermediate_size = c.intermediate_size;
        stack.layers = c.num_hidden_layers;
        stack.rms_norm_eps = c.rms_norm_eps;
        stack.rope_theta = c.rope_theta;
        stack.use_qk_norm = false;
        stack.qkv_layout = packed_qkv ? modules::DecoderQKVLayout::PackedQKV : modules::DecoderQKVLayout::Separate;
        stack.runtime.attention.prefill_mode = modules::DecoderAttentionMode::FlashGroupedViewKV;
        stack.runtime.attention.static_mode = modules::DecoderAttentionMode::FlashGroupedViewKV;
        stack.runtime.static_cache.update_mode = modules::DecoderStaticCacheUpdateMode::DirectSetRows;
        stack.runtime.static_cache.set_rows_mode = modules::DecoderStaticCacheSetRowsMode::BackendViewOptimized;
        stack.runtime.mlp.mode = packed_gate_up ? modules::DecoderMLPMode::PackedGateUp
                                              : modules::DecoderMLPMode::FusedSwiGLU;
        qwen2_ = std::make_unique<modules::CausalDecoderRuntime>(execution_, config, weights);

        lookup_context_.reset(ggml_init({kContextBytes, nullptr, true}));
        if (!lookup_context_) {
            throw std::runtime_error("Audio Flamingo embedding context allocation failed");
        }
        core::ModuleBuildContext ctx{lookup_context_.get(), "audio_flamingo.embedding", execution.backend_type()};
        ids_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({kLookupSteps}));
        ggml_set_input(ids_.tensor);
        embeddings_ = modules::EmbeddingModule({c.vocab_size, c.hidden_size}).build(ctx, ids_, weights.token_embedding);
        ggml_set_output(embeddings_.tensor);
        lookup_graph_ = ggml_new_graph_custom(ctx.ggml, 32, false);
        ggml_build_forward_expand(lookup_graph_, embeddings_.tensor);
        lookup_allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!ggml_gallocr_alloc_graph(lookup_allocator_.get(), lookup_graph_)) {
            throw std::runtime_error("Audio Flamingo embedding graph allocation failed");
        }
        core::prepare_host_graph_plan(execution_, lookup_graph_, lookup_plan_);
    }

    ~Impl() {
        core::release_backend_graph_resources(execution_.backend(), lookup_graph_);
    }

    AudioFlamingoGeneratedTokens generate(const AudioFlamingoPrompt & prompt,
        const AudioFlamingoAudioProjectorOutput & audio, const AudioFlamingoGenerationOptions & options) {
        const auto & c = assets_->config.text_decoder;
        const int64_t steps = static_cast<int64_t>(prompt.input_ids.size());
        if (steps == 0 || options.max_new_tokens <= 0 || steps + options.max_new_tokens > c.max_position_embeddings) {
            throw std::runtime_error("Audio Flamingo prompt plus max_tokens exceeds the text context or is empty");
        }
        if (audio.hidden_size != c.hidden_size || audio.tokens != static_cast<int64_t>(prompt.audio_token_positions.size()) ||
            audio.values.size() != static_cast<size_t>(audio.tokens * c.hidden_size)) {
            throw std::runtime_error("Audio Flamingo audio embeddings do not match prompt placeholders");
        }
        const auto started = std::chrono::steady_clock::now();
        std::vector<float> input(static_cast<size_t>(steps * c.hidden_size));
        std::vector<int64_t> text_positions;
        for (int64_t pos = 0; pos < steps; ++pos) {
            if (prompt.input_ids[static_cast<size_t>(pos)] != c.audio_token_id) {
                text_positions.push_back(pos);
            }
        }
        std::vector<int32_t> ids(kLookupSteps, 0);
        std::vector<float> text_embeddings(static_cast<size_t>(kLookupSteps * c.hidden_size));
        for (int64_t offset = 0; offset < static_cast<int64_t>(text_positions.size()); offset += kLookupSteps) {
            const auto count = std::min(kLookupSteps, static_cast<int64_t>(text_positions.size()) - offset);
            for (int64_t i = 0; i < count; ++i) {
                ids[static_cast<size_t>(i)] = prompt.input_ids[static_cast<size_t>(text_positions[offset + i])];
            }
            ggml_backend_tensor_set(ids_.tensor, ids.data(), 0, ids.size() * sizeof(int32_t));
            if (core::compute_graph(execution_, lookup_graph_, lookup_plan_, "audio_flamingo.embedding") != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Audio Flamingo embedding lookup failed");
            }
            ggml_backend_tensor_get(embeddings_.tensor, text_embeddings.data(),
                0, static_cast<size_t>(count * c.hidden_size) * sizeof(float));
            for (int64_t i = 0; i < count; ++i) {
                std::copy_n(text_embeddings.data() + i * c.hidden_size, c.hidden_size,
                    input.data() + text_positions[offset + i] * c.hidden_size);
            }
        }
        for (size_t i = 0; i < prompt.audio_token_positions.size(); ++i) {
            const int32_t pos = prompt.audio_token_positions[i];
            if (pos < 0 || pos >= steps) {
                throw std::runtime_error("Audio Flamingo audio placeholder out of range");
            }
            std::copy_n(audio.values.data() + i * c.hidden_size, c.hidden_size, input.data() + pos * c.hidden_size);
        }
        auto result = qwen2_->prefill_embeddings_into_cache(input, steps, steps + options.max_new_tokens, kPrefillSteps);
        debug::timing_log_scalar("audio_flamingo.decoder.prefill_ms", debug::elapsed_ms(started));
        const auto decode_start = std::chrono::steady_clock::now();
        sampling::HfSamplingOptions sampling_options;
        sampling_options.do_sample = options.do_sample;
        sampling_options.repetition_penalty = options.repetition_penalty;
        sampling_options.temperature = options.temperature;
        sampling_options.top_k = options.top_k;
        sampling_options.top_p = options.top_p;
        sampling::HfSampler sampler;
        sampling::HfSamplerScratch scratch;
        std::mt19937 rng(static_cast<uint32_t>(options.seed));
        auto history = prompt.input_ids;
        AudioFlamingoGeneratedTokens out;
        for (int64_t i = 0; i < options.max_new_tokens; ++i) {
            const int32_t token = sampler.sample(result.logits, history, sampling_options, scratch, rng, nullptr, "Audio Flamingo");
            if (std::find(c.eos_token_ids.begin(), c.eos_token_ids.end(), token) != c.eos_token_ids.end()) {
                break;
            }
            out.token_ids.push_back(token);
            history.push_back(token);
            if (i + 1 < options.max_new_tokens) {
                result = qwen2_->decode_token(token);
            }
        }
        debug::timing_log_scalar("audio_flamingo.decoder.decode_ms", debug::elapsed_ms(decode_start));
        debug::trace_log_scalar("audio_flamingo.generated_tokens", out.token_ids.size());
        return out;
    }

    std::shared_ptr<const AudioFlamingoAssets> assets_;
    core::ExecutionContext & execution_;
    core::BackendWeightStore store_;
    std::unique_ptr<modules::CausalDecoderRuntime> qwen2_;
    std::unique_ptr<ggml_context, ContextDeleter> lookup_context_;
    std::unique_ptr<ggml_gallocr, AllocatorDeleter> lookup_allocator_;
    core::TensorValue ids_, embeddings_;
    ggml_cgraph * lookup_graph_ = nullptr;
    core::HostGraphPlan lookup_plan_;
};

AudioFlamingoQwen2DecoderRuntime::AudioFlamingoQwen2DecoderRuntime(std::shared_ptr<const AudioFlamingoAssets> assets,
    core::ExecutionContext & execution, assets::TensorStorageType storage)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, storage)) {}

AudioFlamingoQwen2DecoderRuntime::~AudioFlamingoQwen2DecoderRuntime() = default;

AudioFlamingoGeneratedTokens AudioFlamingoQwen2DecoderRuntime::generate(const AudioFlamingoPrompt & prompt,
    const AudioFlamingoAudioProjectorOutput & audio, const AudioFlamingoGenerationOptions & options) {
    return impl_->generate(prompt, audio, options);
}

}  // namespace engine::models::audio_flamingo
