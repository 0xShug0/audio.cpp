#include "engine/community_models/granite_speech5/decoder.h"

#include <stdexcept>
#include <utility>

namespace engine::community_models::granite_speech5 {
namespace {
modules::DecoderStackConfig make_granite_stack_config(core::BackendType backend) {
    modules::DecoderStackConfig stack;
    stack.hidden_size = 2048;
    stack.num_attention_heads = 16;
    stack.num_key_value_heads = 4;
    stack.head_dim = 128;
    stack.intermediate_size = 4096;
    stack.layers = kGraniteSpeech5DecoderLayers;
    stack.rms_norm_eps = 1e-5f;
    stack.rope_theta = 10000.0f;
    stack.rope_type = GGML_ROPE_TYPE_NEOX;
    stack.use_qk_norm = false;
    stack.attention_scale = kGraniteSpeech5AttentionScale;
    stack.residual_scale = kGraniteSpeech5ResidualScale;
    modules::RoPEConfig rope{stack.head_dim, stack.rope_type, stack.rope_theta};
    rope.freq_scale = kGraniteSpeech5YarnFreqScale;
    rope.ext_factor = kGraniteSpeech5YarnExtFactor;
    rope.attn_factor = kGraniteSpeech5YarnAttnFactor;
    rope.beta_fast = kGraniteSpeech5YarnBetaFast;
    rope.beta_slow = kGraniteSpeech5YarnBetaSlow;
    rope.orig_context = kGraniteSpeech5YarnOrigContext;
    stack.rope = rope;
    // Retain manual Vulkan prefill after flash prefill triggered device loss.
    // Single-token decoding still uses the framework flash path.
    stack.runtime.attention.prefill_mode = backend == core::BackendType::Vulkan
        ? modules::DecoderAttentionMode::ManualRepeat
        : modules::DecoderAttentionMode::FlashGrouped;
    stack.runtime.static_cache.update_mode = modules::DecoderStaticCacheUpdateMode::DirectSetRows;
    return stack;
}

}  // namespace

modules::CausalDecoderConfig make_granite_speech5_causal_decoder_config(core::BackendType backend) {
    modules::CausalDecoderConfig config;
    config.stack = make_granite_stack_config(backend);
    config.logits_size = 100353;
    config.logits_mode = modules::CausalDecoderLogitsMode::LastStep;
    config.use_lm_head_bias = false;
    config.embedding_scale = kGraniteSpeech5EmbedScale;
    config.logit_divisor = kGraniteSpeech5LogitDivisor;
    config.static_cache_type = GGML_TYPE_F32;
    return config;
}

runtime::GreedyCausalDecoderSpec make_granite_speech5_greedy_decoder_spec(core::BackendType backend) {
    runtime::GreedyCausalDecoderSpec spec;
    spec.decoder = make_granite_speech5_causal_decoder_config(backend);
    spec.vocab_size = 100353;
    spec.max_position_embeddings = kGraniteSpeech5MaxPositionEmbeddings;
    spec.tie_word_embeddings = false;
    spec.attention_bias = false;
    spec.packed_qkv = false;
    spec.token_embedding_tensor = "language_model.model.embed_tokens.weight";
    spec.lm_head_tensor = "language_model.lm_head.weight";
    spec.final_norm_tensor = "language_model.model.norm.weight";
    spec.layer_prefix = "language_model.model.layers";
    spec.eos_token_ids = {kGraniteSpeech5EosTokenId};
    return spec;
}

void granite_speech5_require_generation_budget(
    int64_t prompt_len,
    int64_t max_new_tokens,
    int64_t max_position_embeddings) {
    if (prompt_len <= 0) {
        throw std::runtime_error("granite_speech5 decoder prompt is empty");
    }
    if (max_new_tokens <= 0) {
        throw std::runtime_error("granite_speech5 decoder max_new_tokens must be positive");
    }
    if (prompt_len + max_new_tokens > max_position_embeddings) {
        throw std::runtime_error(
            "granite_speech5 decoder request exceeds max_position_embeddings (" +
            std::to_string(prompt_len + max_new_tokens) + " > " + std::to_string(max_position_embeddings) + ")");
    }
}

}  // namespace engine::community_models::granite_speech5
