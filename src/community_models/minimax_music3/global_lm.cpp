#include "engine/community_models/minimax_music3/global_lm.h"

#include "engine/framework/modules/weight_binding.h"

#include <stdexcept>
#include <string>

namespace engine::models::minimax_music3 {
namespace {

namespace binding = engine::modules::binding;

void validate_storage_type(assets::TensorStorageType storage_type) {
    switch (storage_type) {
        case assets::TensorStorageType::Native:
        case assets::TensorStorageType::F32:
        case assets::TensorStorageType::F16:
        case assets::TensorStorageType::BF16:
        case assets::TensorStorageType::Q8_0:
        case assets::TensorStorageType::Q4_0:
        case assets::TensorStorageType::Q4_K:
            return;
        default:
            throw std::runtime_error("MiniMax Music 3 weight_type supports native, bf16, f16, q8_0, q4_0, and q4_k");
    }
}

modules::QwenDecoderActivationCastPolicy activation_cast_policy(core::BackendType backend_type) {
    modules::QwenDecoderActivationCastPolicy policy;
    (void)backend_type;
    return policy;
}

modules::QwenDecoderLayerWeights load_qwen_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const MiniMaxMusic3QwenConfig & config,
    assets::TensorStorageType storage_type,
    int64_t layer) {
    const std::string prefix = "model.layers." + std::to_string(layer);
    modules::QwenDecoderLayerWeights out;
    out.input_norm = binding::norm_weight_from_source(store, source, prefix + ".input_layernorm", config.hidden_size);
    out.self_attention.q_weight = store.load_tensor(
        source,
        prefix + ".self_attn.q_proj.weight",
        storage_type,
        {config.attention_heads * config.head_dim, config.hidden_size});
    out.self_attention.k_weight = store.load_tensor(
        source,
        prefix + ".self_attn.k_proj.weight",
        storage_type,
        {config.kv_heads * config.head_dim, config.hidden_size});
    out.self_attention.v_weight = store.load_tensor(
        source,
        prefix + ".self_attn.v_proj.weight",
        storage_type,
        {config.kv_heads * config.head_dim, config.hidden_size});
    out.self_attention.out_weight = store.load_tensor(
        source,
        prefix + ".self_attn.o_proj.weight",
        storage_type,
        {config.hidden_size, config.attention_heads * config.head_dim});
    out.q_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.q_norm", config.head_dim);
    out.k_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.k_norm", config.head_dim);
    out.post_norm = binding::norm_weight_from_source(
        store,
        source,
        prefix + ".post_attention_layernorm",
        config.hidden_size);
    out.mlp.gate_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.gate_proj",
        storage_type,
        config.intermediate_size,
        config.hidden_size,
        false);
    out.mlp.up_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.up_proj",
        storage_type,
        config.intermediate_size,
        config.hidden_size,
        false);
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

}  // namespace

modules::QwenCausalDecodeRuntimeConfig make_minimax_music3_global_lm_runtime_config(
    const MiniMaxMusic3Config & config,
    core::BackendType backend_type,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes) {
    modules::QwenCausalDecodeRuntimeConfig out;
    out.trace_name = "minimax_music3.ar";
    out.prefill_graph_arena_bytes = prefill_graph_arena_bytes;
    out.decode_graph_arena_bytes = decode_graph_arena_bytes;
    out.decoder.stack.hidden_size = config.qwen.hidden_size;
    out.decoder.stack.num_attention_heads = config.qwen.attention_heads;
    out.decoder.stack.num_key_value_heads = config.qwen.kv_heads;
    out.decoder.stack.head_dim = config.qwen.head_dim;
    out.decoder.stack.intermediate_size = config.qwen.intermediate_size;
    out.decoder.stack.layers = config.qwen.layers;
    out.decoder.stack.rms_norm_eps = config.qwen.rms_norm_eps;
    out.decoder.stack.rope_theta = config.qwen.rope_theta;
    out.decoder.stack.rope_type = GGML_ROPE_TYPE_NEOX;
    out.decoder.stack.attention_precision = GGML_PREC_F32;
    out.decoder.stack.projection_precision = GGML_PREC_DEFAULT;
    out.decoder.stack.use_qk_norm = true;
    out.decoder.stack.activation_cast = activation_cast_policy(backend_type);
    out.decoder.stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.decoder.stack.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.decoder.stack.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    out.decoder.stack.runtime.static_cache.set_rows_mode =
        modules::QwenDecoderStaticCacheSetRowsMode::BackendViewOptimized;
    if (backend_type == core::BackendType::Cuda || backend_type == core::BackendType::Hip ||
        backend_type == core::BackendType::Vulkan) {
        out.decoder.static_cache_type = GGML_TYPE_F16;
    }
    out.decoder.logits_size = config.qwen.vocab_size;
    out.decoder.logits_mode = modules::QwenCausalDecoderLogitsMode::LastStep;
    out.decoder.lm_head_precision = GGML_PREC_DEFAULT;
    out.readback_round_type = GGML_TYPE_BF16;
    if (backend_type == core::BackendType::Metal) {
        out.decoder.lm_head_input_type = GGML_TYPE_F32;
    } else if (backend_type == core::BackendType::Vulkan) {
        out.decoder.lm_head_input_type = GGML_TYPE_F16;
    } else if (backend_type != core::BackendType::Cpu) {
        out.decoder.lm_head_input_type = GGML_TYPE_BF16;
    }
    return out;
}

MiniMaxMusic3GlobalLMWeights load_minimax_music3_global_lm_weights(
    const MiniMaxMusic3Assets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    validate_storage_type(storage_type);
    const auto & config = assets.config.qwen;
    const auto & source = *assets.language_model_weights;
    MiniMaxMusic3GlobalLMWeights out;
    out.store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "minimax_music3.global_lm.weights",
        weight_context_bytes);
    out.token_embedding = out.store->load_tensor(
        source,
        "model.embed_tokens.weight",
        storage_type,
        {config.vocab_size, config.hidden_size});
    out.qwen.token_embedding = out.token_embedding;
    out.qwen.stack.layers.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        out.qwen.stack.layers.push_back(load_qwen_layer(*out.store, source, config, storage_type, layer));
    }
    out.qwen.final_norm = binding::norm_weight_from_source(*out.store, source, "model.norm", config.hidden_size);
    out.qwen.lm_head = binding::linear_from_source(
        *out.store,
        source,
        "lm_head",
        storage_type,
        config.vocab_size,
        config.hidden_size,
        false);
    out.store->upload();
    assets.language_model_weights->release_storage();
    return out;
}

}  // namespace engine::models::minimax_music3
