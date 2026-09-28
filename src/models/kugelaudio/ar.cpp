#include "engine/models/kugelaudio/ar.h"

#include "engine/framework/modules/weight_binding.h"

#include <cstring>

namespace engine::models::kugelaudio {

modules::CausalDecoderRuntimeConfig ar_runtime_config(const ArConfig & config) {
    modules::CausalDecoderRuntimeConfig result;
    result.trace_name = "kugelaudio.ar";
    result.prefill_graph_arena_bytes = 4 * 1024 * 1024;
    result.decode_graph_arena_bytes = 4 * 1024 * 1024;
    result.return_hidden = true;
    result.evict_cuda_graph_cache_on_release = true;
    auto & decoder = result.decoder;
    decoder.logits_size = kSpeechTokens.size();
    decoder.logits_mode = modules::CausalDecoderLogitsMode::LastStep;
    decoder.static_cache_type = GGML_TYPE_F16;
    auto & stack = decoder.stack;
    stack.hidden_size = config.hidden_size;
    stack.intermediate_size = config.intermediate_size;
    stack.num_attention_heads = config.attention_heads;
    stack.num_key_value_heads = config.kv_heads;
    stack.head_dim = config.head_dim;
    stack.layers = config.layers;
    stack.rms_norm_eps = config.norm_eps;
    stack.rope_theta = config.rope_theta;
    stack.use_qk_norm = false;
    stack.runtime.attention.prefill_mode = modules::DecoderAttentionMode::FlashGroupedViewKV;
    stack.runtime.attention.static_mode = modules::DecoderAttentionMode::FlashGroupedViewKV;
    stack.runtime.static_cache.update_mode = modules::DecoderStaticCacheUpdateMode::DirectSetRows;
    stack.runtime.static_cache.set_rows_mode = modules::DecoderStaticCacheSetRowsMode::BackendViewOptimized;
    return result;
}

modules::CausalDecoderRuntimeWeights load_ar_weights(
    core::BackendWeightStore & store, const assets::TensorSource & source,
    assets::TensorStorageType storage, const ArConfig & config) {
    namespace binding = modules::binding;
    modules::CausalDecoderRuntimeWeights result;
    const std::string prefix = "model.language_model.";
    const auto hidden = config.hidden_size;
    result.token_embedding = store.load_tensor(source, prefix + "embed_tokens.weight", storage,
                                                {config.vocab_size, hidden});
    for (int64_t index = 0; index < config.layers; ++index) {
        const auto name = prefix + "layers." + std::to_string(index);
        modules::DecoderLayerWeights layer;
        layer.input_norm = binding::norm_weight_from_source(store, source, name + ".input_layernorm", hidden);
        layer.post_norm = binding::norm_weight_from_source(store, source, name + ".post_attention_layernorm", hidden);
        const auto q_channels = config.attention_heads * config.head_dim;
        const auto kv_channels = config.kv_heads * config.head_dim;
        auto & attention = layer.self_attention;
        attention.q_weight = store.load_tensor(source, name + ".self_attn.q_proj.weight", storage, {q_channels, hidden});
        attention.k_weight = store.load_tensor(source, name + ".self_attn.k_proj.weight", storage, {kv_channels, hidden});
        attention.v_weight = store.load_tensor(source, name + ".self_attn.v_proj.weight", storage, {kv_channels, hidden});
        attention.q_bias = store.load_f32_tensor(source, name + ".self_attn.q_proj.bias", {q_channels});
        attention.k_bias = store.load_f32_tensor(source, name + ".self_attn.k_proj.bias", {kv_channels});
        attention.v_bias = store.load_f32_tensor(source, name + ".self_attn.v_proj.bias", {kv_channels});
        attention.out_weight = store.load_tensor(source, name + ".self_attn.o_proj.weight", storage, {hidden, q_channels});
        layer.mlp.gate_proj = binding::linear_from_source(store, source, name + ".mlp.gate_proj", storage,
                                                         config.intermediate_size, hidden, false);
        layer.mlp.up_proj = binding::linear_from_source(store, source, name + ".mlp.up_proj", storage,
                                                       config.intermediate_size, hidden, false);
        layer.mlp.down_proj = binding::linear_from_source(store, source, name + ".mlp.down_proj", storage,
                                                         hidden, config.intermediate_size, false);
        result.stack.layers.push_back(std::move(layer));
    }
    result.final_norm = binding::norm_weight_from_source(store, source, prefix + "norm", hidden);
    // The official generator masks every other vocabulary entry. Upload only
    // these four output rows, retaining their original storage representation.
    const auto head = source.require_tensor("lm_head.weight", storage, {config.vocab_size, hidden});
    const auto row_bytes = ggml_row_size(head.type, hidden);
    std::vector<std::byte> rows(kSpeechTokens.size() * row_bytes);
    for (size_t i = 0; i < kSpeechTokens.size(); ++i) {
        std::memcpy(rows.data() + i * row_bytes, head.bytes.data() + kSpeechTokens[i] * row_bytes, row_bytes);
    }
    result.lm_head = modules::LinearWeights{
        store.make_tensor(core::TensorShape::from_dims({static_cast<int64_t>(kSpeechTokens.size()), hidden}),
                          head.type, rows.data(), rows.size()), std::nullopt};
    return result;
}

}  // namespace engine::models::kugelaudio
