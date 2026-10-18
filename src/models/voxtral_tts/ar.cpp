#include "engine/models/voxtral_tts/ar.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <array>

namespace engine::models::voxtral_tts {

modules::DecoderLayerWeights load_transformer_layer(
    core::BackendWeightStore & store, const assets::TensorSource & source,
    const std::string & prefix, const TransformerConfig & config, assets::TensorStorageType storage) {
    namespace binding = modules::binding;
    const auto hidden = config.hidden_size;
    const auto query = config.attention_heads * config.head_dim;
    const auto kv = config.kv_heads * config.head_dim;
    modules::DecoderLayerWeights weights;
    weights.input_norm = binding::norm_weight_from_source(store, source, prefix + ".attention_norm", hidden);
    weights.post_norm = binding::norm_weight_from_source(store, source, prefix + ".ffn_norm", hidden);
    weights.self_attention.q_weight = store.load_tensor(source, prefix + ".attention.wq.weight", storage, {query, hidden});
    weights.self_attention.k_weight = store.load_tensor(source, prefix + ".attention.wk.weight", storage, {kv, hidden});
    weights.self_attention.v_weight = store.load_tensor(source, prefix + ".attention.wv.weight", storage, {kv, hidden});
    weights.self_attention.out_weight = store.load_tensor(source, prefix + ".attention.wo.weight", storage, {hidden, query});
    weights.mlp.gate_proj = binding::linear_from_source(store, source, prefix + ".feed_forward.w1",
                                                       storage, config.intermediate_size, hidden, false);
    weights.mlp.up_proj = binding::linear_from_source(store, source, prefix + ".feed_forward.w3",
                                                     storage, config.intermediate_size, hidden, false);
    weights.mlp.down_proj = binding::linear_from_source(store, source, prefix + ".feed_forward.w2",
                                                       storage, hidden, config.intermediate_size, false);
    return weights;
}

struct VoxtralMinistralARRuntime::Impl {
    core::ExecutionContext & execution;
    const VoxtralTtsConfig & config;
    core::BackendWeightStore store;
    modules::CausalDecoderRuntimeWeights weights;
    core::TensorValue audio_embeddings;
    std::unique_ptr<modules::CausalDecoderRuntime> decoder;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::array<ggml_gallocr_t, 2> allocators{};
    std::array<ggml_cgraph *, 2> graphs{};
    std::array<core::HostGraphPlan, 2> plans;
    core::TensorValue text_ids, code_ids, text_output, code_output;

    Impl(const VoxtralTtsAssets & assets, core::ExecutionContext & execution_,
         assets::TensorStorageType storage, ggml_type cache_type)
        : execution(execution_), config(assets.config),
          store(execution.backend(), execution.backend_type(), "voxtral_tts.ar.weights", kContextBytes) {
        const auto & source = *assets.weights;
        const auto hidden = config.ar.hidden_size;
        weights.token_embedding = store.load_tensor(source, "mm_audio_embeddings.tok_embeddings.weight",
                                                    storage, {config.vocab_size, hidden});
        const auto unpadded_audio_vocab = config.semantic_vocab_size + 2 +
                                         config.acoustic_dim * (config.acoustic_levels + 2);
        const auto audio_vocab = ((unpadded_audio_vocab + 127) / 128) * 128;
        audio_embeddings = store.load_tensor(source, "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight",
                                             storage, {audio_vocab, hidden});
        for (int64_t layer = 0; layer < config.ar.layers; ++layer) {
            weights.stack.layers.push_back(load_transformer_layer(
                store, source, "layers." + std::to_string(layer), config.ar, storage));
        }
        weights.final_norm = modules::binding::norm_weight_from_source(store, source, "norm", hidden);
        const auto semantic_head_size = ((config.semantic_vocab_size + 2 + 127) / 128) * 128;
        weights.lm_head = modules::binding::linear_from_source(store, source,
            "acoustic_transformer.semantic_codebook_output", storage, semantic_head_size, hidden, false);
        store.upload();
        modules::CausalDecoderRuntimeConfig runtime_config;
        runtime_config.trace_name = "voxtral_tts.ar";
        runtime_config.prefill_graph_arena_bytes = kContextBytes;
        runtime_config.decode_graph_arena_bytes = kContextBytes;
        runtime_config.return_hidden = true;
        runtime_config.evict_cuda_graph_cache_on_release = true;
        auto & causal = runtime_config.decoder;
        causal.static_cache_type = cache_type;
        causal.logits_size = semantic_head_size;
        auto & stack = causal.stack;
        stack.hidden_size = hidden;
        stack.intermediate_size = config.ar.intermediate_size;
        stack.num_attention_heads = config.ar.attention_heads;
        stack.num_key_value_heads = config.ar.kv_heads;
        stack.head_dim = config.ar.head_dim;
        stack.layers = config.ar.layers;
        stack.rms_norm_eps = config.ar.norm_eps;
        stack.rope_theta = config.ar.rope_theta;
        stack.rope_type = GGML_ROPE_TYPE_NORMAL;
        stack.use_qk_norm = false;
        stack.runtime.attention.prefill_mode = modules::DecoderAttentionMode::FlashGroupedViewKV;
        stack.runtime.attention.static_mode = modules::DecoderAttentionMode::FlashGroupedViewKV;
        stack.runtime.static_cache.update_mode = modules::DecoderStaticCacheUpdateMode::DirectSetRows;
        stack.runtime.static_cache.set_rows_mode = modules::DecoderStaticCacheSetRowsMode::BackendViewOptimized;
        decoder = std::make_unique<modules::CausalDecoderRuntime>(execution, runtime_config, weights);

        context.reset(ggml_init({kContextBytes, nullptr, true}));
        core::ModuleBuildContext ctx{context.get(), "voxtral_tts.embeddings", execution.backend_type()};
        text_ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({128}));
        code_ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({config.acoustic_dim + 1}));
        ggml_set_input(text_ids.tensor);
        ggml_set_input(code_ids.tensor);
        text_output = modules::EmbeddingModule({config.vocab_size, hidden}).build(ctx, text_ids, weights.token_embedding);
        auto codes = modules::EmbeddingModule({audio_vocab, hidden}).build(ctx, code_ids, audio_embeddings);
        code_output = modules::ReduceSumModule({0}).build(ctx, codes);
        const std::array<ggml_tensor *, 2> outputs{text_output.tensor, code_output.tensor};
        for (size_t i = 0; i < graphs.size(); ++i) {
            ggml_set_output(outputs[i]);
            graphs[i] = ggml_new_graph_custom(context.get(), 256, false);
            ggml_build_forward_expand(graphs[i], outputs[i]);
            core::validate_backend_graph_supported(execution.backend(), graphs[i], "Voxtral TTS embeddings");
            allocators[i] = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
            if (!ggml_gallocr_alloc_graph(allocators[i], graphs[i])) {
                throw std::runtime_error("Voxtral TTS embedding graph allocation failed");
            }
            core::prepare_host_graph_plan(execution, graphs[i], plans[i]);
            debug::timing_log_scalar("voxtral_tts.embeddings.graph." + std::to_string(i) + ".buffer_mb",
                static_cast<double>(ggml_gallocr_get_buffer_size(allocators[i], 0)) / (1024 * 1024));
        }
        debug::timing_log_context_reservation("voxtral_tts.embeddings.graph", context.get());
    }

    ~Impl() {
        for (size_t i = 0; i < graphs.size(); ++i) {
            core::release_backend_graph_resources(execution.backend(), graphs[i], true);
            plans[i].reset();
            ggml_gallocr_free(allocators[i]);
        }
    }

    modules::CausalDecoderStepResult prefill(const std::vector<int32_t> & prompt,
                                            const std::vector<float> & voice, int64_t max_frames) {
        const auto begin = std::chrono::steady_clock::now();
        std::vector<float> embeddings;
        embeddings.reserve(prompt.size() * config.ar.hidden_size);
        for (size_t offset = 0; offset < prompt.size(); offset += 128) {
            std::array<int32_t, 128> ids{};
            const auto count = std::min<size_t>(ids.size(), prompt.size() - offset);
            std::copy_n(prompt.begin() + offset, count, ids.begin());
            ggml_backend_tensor_set(text_ids.tensor, ids.data(), 0, sizeof(ids));
            if (core::compute_graph(execution, graphs[0], plans[0], "Voxtral TTS text embeddings") != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Voxtral TTS text embedding compute failed");
            }
            auto block = core::read_tensor_f32(text_output.tensor);
            embeddings.insert(embeddings.end(), block.begin(), block.begin() + count * config.ar.hidden_size);
        }
        std::copy(voice.begin(), voice.end(), embeddings.begin() + 2 * config.ar.hidden_size);
        debug::timing_log_scalar("voxtral_tts.ar.embed_ms", debug::elapsed_ms(begin));
        const auto prefill_begin = std::chrono::steady_clock::now();
        auto prefill = decoder->prefill_embeddings_into_decode_cache(
            embeddings, static_cast<int64_t>(prompt.size()), static_cast<int64_t>(prompt.size()) + max_frames);
        debug::timing_log_scalar("voxtral_tts.ar.prefill_into_cache_ms", debug::elapsed_ms(prefill_begin));
        debug::timing_log_scalar("voxtral_tts.ar.prefill_ms", debug::elapsed_ms(begin));
        return {std::move(prefill.logits), std::move(prefill.hidden)};
    }

    modules::CausalDecoderStepResult decode(const std::vector<int32_t> & audio_codes) {
        if (audio_codes.size() != static_cast<size_t>(config.acoustic_dim + 1)) {
            throw std::runtime_error("Voxtral TTS audio frame has an invalid codebook count");
        }
        auto ids = audio_codes;
        int64_t offset = config.semantic_vocab_size + kAudioTokenOffset;
        const int64_t acoustic_vocab = config.acoustic_levels + kAudioTokenOffset;
        for (size_t i = 1; i < ids.size(); ++i, offset += acoustic_vocab) {
            ids[i] += static_cast<int32_t>(offset);
        }
        ggml_backend_tensor_set(code_ids.tensor, ids.data(), 0, ids.size() * sizeof(int32_t));
        if (core::compute_graph(execution, graphs[1], plans[1], "Voxtral TTS audio embeddings") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Voxtral TTS audio embedding compute failed");
        }
        return decoder->decode_embedding(core::read_tensor_f32(code_output.tensor));
    }
};

VoxtralMinistralARRuntime::VoxtralMinistralARRuntime(
    const VoxtralTtsAssets & assets, core::ExecutionContext & execution,
    assets::TensorStorageType storage, ggml_type cache_type)
    : impl_(std::make_unique<Impl>(assets, execution, storage, cache_type)) {}
VoxtralMinistralARRuntime::~VoxtralMinistralARRuntime() = default;
modules::CausalDecoderStepResult VoxtralMinistralARRuntime::prefill(
    const std::vector<int32_t> & prompt, const std::vector<float> & voice, int64_t max_frames) {
    return impl_->prefill(prompt, voice, max_frames);
}
modules::CausalDecoderStepResult VoxtralMinistralARRuntime::decode(const std::vector<int32_t> & codes) {
    return impl_->decode(codes);
}

}  // namespace engine::models::voxtral_tts
