#include "engine/community_models/mira_tts/generator.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/transformers/qwen_causal_decode_runtime.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/hf_sampler.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::community_models::mira_tts {
namespace {

namespace binding = engine::modules::binding;

struct MiraQwenWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    // This context owns only the sparse head's tensor metadata. The tied
    // embedding storage outlives it, and both outlive the decoder runtime.
    std::shared_ptr<ggml_context> head_context;
    core::TensorValue token_embedding;
    core::TensorValue lm_head;
    int64_t lm_head_row_offset = 0;
    modules::QwenDecoderStackWeights stack;
    modules::NormWeights final_norm;
};

modules::QwenDecoderLayerWeights load_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const MiraTTSConfig & config,
    assets::TensorStorageType storage_type,
    int64_t layer) {
    const std::string prefix = "model.layers." + std::to_string(layer);
    modules::QwenDecoderLayerWeights out;
    out.input_norm = binding::norm_weight_from_source(
        store, source, prefix + ".input_layernorm", config.hidden_size);

    const int64_t q_out = config.attention_heads * config.head_dim;
    const int64_t kv_out = config.kv_heads * config.head_dim;
    std::vector<float> qkv = source.require_f32(
        prefix + ".self_attn.q_proj.weight", {q_out, config.hidden_size});
    const auto k = source.require_f32(
        prefix + ".self_attn.k_proj.weight", {kv_out, config.hidden_size});
    const auto v = source.require_f32(
        prefix + ".self_attn.v_proj.weight", {kv_out, config.hidden_size});
    qkv.insert(qkv.end(), k.begin(), k.end());
    qkv.insert(qkv.end(), v.begin(), v.end());
    out.self_attention.qkv_weight = store.make_from_f32(
        core::TensorShape::from_dims({q_out + 2 * kv_out, config.hidden_size}),
        storage_type,
        std::move(qkv));
    std::vector<float> qkv_bias = source.require_f32(
        prefix + ".self_attn.q_proj.bias", {q_out});
    const auto k_bias = source.require_f32(
        prefix + ".self_attn.k_proj.bias", {kv_out});
    const auto v_bias = source.require_f32(
        prefix + ".self_attn.v_proj.bias", {kv_out});
    qkv_bias.insert(qkv_bias.end(), k_bias.begin(), k_bias.end());
    qkv_bias.insert(qkv_bias.end(), v_bias.begin(), v_bias.end());
    out.self_attention.qkv_bias = store.make_f32(
        core::TensorShape::from_dims({q_out + 2 * kv_out}),
        qkv_bias);
    out.self_attention.out_weight = store.load_tensor(
        source,
        prefix + ".self_attn.o_proj.weight",
        storage_type,
        {config.hidden_size, q_out});
    out.post_norm = binding::norm_weight_from_source(
        store, source, prefix + ".post_attention_layernorm", config.hidden_size);

    std::vector<float> gate_up = source.require_f32(
        prefix + ".mlp.gate_proj.weight",
        {config.intermediate_size, config.hidden_size});
    const auto up = source.require_f32(
        prefix + ".mlp.up_proj.weight",
        {config.intermediate_size, config.hidden_size});
    gate_up.insert(gate_up.end(), up.begin(), up.end());
    out.mlp.gate_up_proj = modules::LinearWeights{
        store.make_from_f32(
            core::TensorShape::from_dims(
                {2 * config.intermediate_size, config.hidden_size}),
            storage_type,
            std::move(gate_up)),
        std::nullopt};
    out.mlp.down_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.down_proj",
        storage_type,
        config.hidden_size,
        config.intermediate_size,
        false);
    return out;
}

modules::QwenCausalDecoderConfig decoder_config(
    const MiraTTSConfig & config,
    core::BackendType backend_type) {
    modules::QwenCausalDecoderConfig out;
    out.stack.hidden_size = config.hidden_size;
    out.stack.num_attention_heads = config.attention_heads;
    out.stack.num_key_value_heads = config.kv_heads;
    out.stack.head_dim = config.head_dim;
    out.stack.intermediate_size = config.intermediate_size;
    out.stack.layers = config.layers;
    out.stack.rms_norm_eps = config.rms_norm_eps;
    out.stack.rope_theta = config.rope_theta;
    out.stack.rope_type = GGML_ROPE_TYPE_NEOX;
    out.stack.use_qk_norm = false;
    out.stack.qkv_layout = modules::QwenDecoderQKVLayout::PackedQKV;
    out.stack.runtime.mlp.mode = modules::QwenDecoderMLPMode::PackedGateUp;
    out.stack.runtime.attention.prefill_mode =
        modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.attention.static_mode =
        modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.static_cache.update_mode =
        modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    if (backend_type == core::BackendType::Vulkan) {
        // Mira's projections are sensitive to Vulkan's default reduced
        // precision. Materialize grouped K/V heads for attention as well:
        // the strided-view path diverges during prompt evaluation.
        out.stack.projection_precision = GGML_PREC_F32;
        out.stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::FlashGrouped;
        out.stack.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGrouped;
    }
    out.logits_size = config.vocab_size;
    out.logits_mode = modules::QwenCausalDecoderLogitsMode::LastStep;
    out.use_lm_head_bias = false;
    if (backend_type == core::BackendType::Metal) {
        out.lm_head_input_type = GGML_TYPE_F16;
    } else if (backend_type != core::BackendType::Cpu &&
               backend_type != core::BackendType::Vulkan) {
        out.lm_head_input_type = GGML_TYPE_BF16;
    }
    return out;
}

std::vector<int32_t> generation_token_ids(const MiraTTSConfig & config) {
    std::vector<int32_t> out;
    out.reserve(static_cast<size_t>(
        config.speech_token_end - config.speech_token_start + 2));
    for (int32_t token = config.speech_token_start;
         token <= config.speech_token_end;
         ++token) {
        out.push_back(token);
    }
    out.push_back(config.eos_token_id);
    return out;
}

std::shared_ptr<MiraQwenWeights> load_weights(
    const MiraTTSAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t context_bytes,
    assets::TensorStorageType storage_type) {
    auto out = std::make_shared<MiraQwenWeights>();
    out->store = std::make_shared<core::BackendWeightStore>(
        backend, backend_type, "mira_tts.lm.weights", context_bytes);
    const auto & config = assets.config;
    const auto & source = *assets.language_model_weights;
    out->token_embedding = out->store->load_tensor(
        source,
        "model.embed_tokens.weight",
        storage_type,
        {config.vocab_size, config.hidden_size});
    out->stack.layers.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        out->stack.layers.push_back(load_layer(
            *out->store, source, config, storage_type, layer));
    }
    out->final_norm = binding::norm_weight_from_source(
        *out->store, source, "model.norm", config.hidden_size);
    out->store->upload();
    out->lm_head = out->token_embedding;
    const char * sparse_head = std::getenv("AUDIOCPP_MIRA_TTS_SPARSE_HEAD");
    if (backend_type == core::BackendType::Cpu &&
        !(sparse_head != nullptr && sparse_head[0] == '0')) {
        // Only MiraTTS knows its speech/EOS alphabet. Keep its weight window
        // here, presenting an ordinary, correctly sized head to shared Qwen.
        const int64_t offset = config.eos_token_id;
        if (offset < 0 || offset >= config.vocab_size ||
            config.speech_token_start < offset ||
            config.speech_token_end < config.speech_token_start ||
            config.speech_token_end >= config.vocab_size) {
            throw std::runtime_error("MiraTTS sparse head does not cover its generation alphabet");
        }
        const int64_t rows = config.vocab_size - offset;
        out->head_context = std::shared_ptr<ggml_context>(
            ggml_init({ggml_tensor_overhead(), nullptr, true}), ggml_free);
        if (!out->head_context) {
            throw std::runtime_error("failed to initialize MiraTTS sparse head context");
        }
        auto * base = out->token_embedding.tensor;
        auto * view = ggml_view_2d(
            out->head_context.get(), base, base->ne[0], rows, base->nb[1],
            static_cast<size_t>(offset) * base->nb[1]);
        if (ggml_backend_view_init(view) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("failed to initialize MiraTTS sparse head view");
        }
        out->lm_head = core::wrap_tensor(
            view, core::TensorShape::from_dims({rows, config.hidden_size}),
            out->token_embedding.type);
        out->lm_head_row_offset = offset;
    }
    return out;
}

modules::QwenCausalDecodeRuntimeConfig runtime_config(
    const MiraTTSConfig & config,
    const MiraQwenWeights & weights,
    core::BackendType backend_type,
    size_t prefill_bytes,
    size_t decode_bytes) {
    modules::QwenCausalDecodeRuntimeConfig out;
    out.trace_name = "mira_tts.lm";
    out.decoder = decoder_config(config, backend_type);
    out.decoder.logits_size = weights.lm_head.shape.dims[0];
    out.decoder.logits_mode = modules::QwenCausalDecoderLogitsMode::LastStep;
    out.decoder.use_lm_head_bias = false;
    out.logits_readback_token_ids = generation_token_ids(config);
    // Readback indices address our local head. Prompt/decode token IDs still
    // address the full embedding vocabulary and are never rebased.
    for (auto & token : out.logits_readback_token_ids) {
        token -= static_cast<int32_t>(weights.lm_head_row_offset);
    }
    out.prefill_graph_arena_bytes = prefill_bytes;
    out.decode_graph_arena_bytes = decode_bytes;
    return out;
}

modules::QwenCausalDecodeRuntimeWeights runtime_weights(
    const MiraQwenWeights & weights) {
    modules::QwenCausalDecodeRuntimeWeights out;
    out.token_embedding = weights.token_embedding;
    out.stack = weights.stack;
    out.final_norm = weights.final_norm;
    out.lm_head = modules::LinearWeights{weights.lm_head, std::nullopt};
    return out;
}

}  // namespace

struct MiraGenerator::Impl {
    Impl(
        const MiraTTSAssets & assets,
        core::ExecutionContext & execution,
        size_t prefill_bytes,
        size_t decode_bytes,
        size_t weight_bytes,
        assets::TensorStorageType storage_type)
        : config(assets.config),
          weights(load_weights(
              assets,
              execution.backend(),
              execution.backend_type(),
              weight_bytes,
              storage_type)),
          runtime(std::make_unique<modules::QwenCausalDecodeRuntime>(
              execution,
              runtime_config(config, *weights, execution.backend_type(), prefill_bytes, decode_bytes),
              runtime_weights(*weights))) {}

    std::vector<int32_t> generate(
        const std::vector<int32_t> & prompt,
        const MiraGenerationOptions & options) {
        if (prompt.empty()) {
            throw std::runtime_error("MiraTTS LM prompt is empty");
        }
        const int64_t room = config.max_position_embeddings -
            static_cast<int64_t>(prompt.size());
        const int64_t max_tokens = std::min(options.max_new_tokens, room);
        if (max_tokens <= 0) {
            throw std::runtime_error("MiraTTS LM prompt exceeds its context window");
        }
        auto prefill = runtime->prefill_tokens(prompt);
        runtime->start_decode_tokens(
            prefill.state, static_cast<int64_t>(prompt.size()) + max_tokens);

        sampling::HfSamplingOptions sampling_options;
        sampling_options.do_sample = true;
        sampling_options.temperature = options.temperature;
        sampling_options.top_k = options.top_k;
        sampling_options.top_p = options.top_p;
        sampling_options.min_p = options.min_p;
        sampling_options.repetition_penalty = options.repetition_penalty;
        sampling_options.min_tokens_to_keep = 1;
        sampling::HfSampler sampler;
        sampling::HfSamplerScratch scratch;
        scratch.reserve_vocab(static_cast<size_t>(config.vocab_size));
        std::mt19937 rng(static_cast<uint32_t>(options.seed));
        // Logits are compacted to [speech codes..., EOS]. Prompt tokens do not
        // overlap this alphabet, so only generated compact ids participate in
        // repetition penalty bookkeeping.
        std::vector<int32_t> history;
        std::vector<int32_t> codes;
        auto logits = std::move(prefill.logits);
        for (int64_t step = 0; step < max_tokens; ++step) {
            const int32_t compact_token = sampler.sample(
                logits,
                history,
                sampling_options,
                scratch,
                rng,
                nullptr,
                "MiraTTS LM");
            const int32_t token = compact_token ==
                    static_cast<int32_t>(config.speech_token_end -
                                         config.speech_token_start + 1)
                ? config.eos_token_id
                : config.speech_token_start + compact_token;
            if (token == config.eos_token_id) {
                break;
            }
            history.push_back(compact_token);
            if (token >= config.speech_token_start && token <= config.speech_token_end) {
                codes.push_back(token - config.speech_token_start);
            }
            logits = runtime->decode_token(token).logits;
        }
        if (codes.empty()) {
            throw std::runtime_error("MiraTTS LM produced no speech tokens");
        }
        return codes;
    }

    MiraTTSConfig config;
    std::shared_ptr<MiraQwenWeights> weights;
    std::unique_ptr<modules::QwenCausalDecodeRuntime> runtime;
};

MiraGenerator::MiraGenerator(
    const MiraTTSAssets & assets,
    core::ExecutionContext & execution,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(
          assets,
          execution,
          prefill_graph_arena_bytes,
          decode_graph_arena_bytes,
          weight_context_bytes,
          weight_storage_type)) {}

MiraGenerator::~MiraGenerator() = default;

std::vector<int32_t> MiraGenerator::generate(
    const std::vector<int32_t> & prompt_ids,
    const MiraGenerationOptions & options) {
    return impl_->generate(prompt_ids, options);
}

void MiraGenerator::release_runtime_graphs() {
    impl_->runtime->release_runtime_graphs();
}

}  // namespace engine::community_models::mira_tts
