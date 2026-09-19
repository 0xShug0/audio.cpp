#include "engine/community_models/confucius4_r2t2/thinker.h"

#include "engine/framework/runtime/greedy_qwen_decoder.h"

#include <stdexcept>
#include <utility>

namespace engine::community_models::confucius4_r2t2 {
namespace {

namespace modules = engine::modules;

runtime::GreedyQwenDecoderSpec make_decoder_spec(const R2T2ASRConfig & config) {
    const auto & text = config.text_decoder;
    runtime::GreedyQwenDecoderSpec spec;
    // Qwen3-style stack: Q/K norms, no attention biases, separate QKV.
    spec.decoder.stack.hidden_size = text.hidden_size;
    spec.decoder.stack.num_attention_heads = text.num_attention_heads;
    spec.decoder.stack.num_key_value_heads = text.num_key_value_heads;
    spec.decoder.stack.head_dim = text.head_dim;
    spec.decoder.stack.intermediate_size = text.intermediate_size;
    spec.decoder.stack.layers = text.num_hidden_layers;
    spec.decoder.stack.rms_norm_eps = text.rms_norm_eps;
    spec.decoder.stack.rope_theta = text.rope_theta;
    spec.decoder.stack.use_qk_norm = true;
    spec.decoder.stack.runtime.static_cache.update_mode =
        modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    spec.decoder.logits_size = text.output_size;
    spec.decoder.logits_mode = modules::QwenCausalDecoderLogitsMode::LastStep;
    spec.vocab_size = text.vocab_size;
    spec.max_position_embeddings = text.max_position_embeddings;
    spec.tie_word_embeddings = config.tie_word_embeddings;
    spec.attention_bias = text.attention_bias;
    spec.packed_qkv = false;

    // R2T2 ships the legacy `thinker.*` namespace and the tied LM head of the
    // base Qwen3-ASR checkpoint; the HF layout is accepted for converted GGUFs.
    const std::string model_prefix = config.hf_transformers_layout
        ? "model.language_model"
        : "thinker.model";
    spec.token_embedding_tensor = model_prefix + ".embed_tokens.weight";
    spec.final_norm_tensor = model_prefix + ".norm.weight";
    spec.layer_prefix = model_prefix + ".layers";
    spec.lm_head_tensor = config.hf_transformers_layout ? "lm_head.weight" : "thinker.lm_head.weight";
    spec.eos_token_ids = text.eos_token_ids;
    return spec;
}

}  // namespace

struct R2T2ASRThinkerRuntime::Impl {
    Impl(
        std::shared_ptr<const R2T2ASRAssets> assets,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type)
        : config(assets == nullptr ? throw std::runtime_error("R2T2 ASR thinker requires assets") : assets->config),
          runtime(
              assets->model_weights,
              make_decoder_spec(assets->config),
              execution,
              prefill_graph_arena_bytes,
              decode_graph_arena_bytes,
              weight_context_bytes,
              weight_storage_type) {}

    R2T2ASRConfig config;
    runtime::GreedyQwenDecoderRuntime runtime;
};

R2T2ASRThinkerRuntime::R2T2ASRThinkerRuntime(
    std::shared_ptr<const R2T2ASRAssets> assets,
    core::ExecutionContext & execution,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          prefill_graph_arena_bytes,
          decode_graph_arena_bytes,
          weight_context_bytes,
          weight_storage_type)) {}

R2T2ASRThinkerRuntime::~R2T2ASRThinkerRuntime() = default;

R2T2ASRGeneratedTokens R2T2ASRThinkerRuntime::generate(
    const R2T2ASRPrompt & prompt,
    const R2T2ASRAudioEmbeddings & audio_embeddings,
    const R2T2ASRGenerationOptions & options) {
    if (prompt.input_ids.empty()) {
        throw std::runtime_error("R2T2 ASR thinker prompt is empty");
    }
    const auto & text = impl_->config.text_decoder;
    if (audio_embeddings.hidden_size != text.hidden_size ||
        audio_embeddings.tokens != static_cast<int64_t>(prompt.audio_token_positions.size()) ||
        static_cast<int64_t>(audio_embeddings.values.size()) != audio_embeddings.tokens * text.hidden_size) {
        throw std::runtime_error("R2T2 ASR audio embeddings do not match the prompt placeholders");
    }
    for (const int32_t position : prompt.audio_token_positions) {
        if (position < 0 || position >= static_cast<int32_t>(prompt.input_ids.size())) {
            throw std::runtime_error("R2T2 ASR audio placeholder position out of range");
        }
    }
    runtime::GreedyQwenDecoderRuntime::Prompt decoder_prompt;
    decoder_prompt.input_ids = prompt.input_ids;
    decoder_prompt.injection.values = audio_embeddings.values;
    decoder_prompt.injection.tokens = audio_embeddings.tokens;
    decoder_prompt.injection.positions = prompt.audio_token_positions;

    R2T2ASRGeneratedTokens out;
    out.token_ids = impl_->runtime.generate(decoder_prompt, options.max_new_tokens, options.reuse_graphs);
    return out;
}

}  // namespace engine::community_models::confucius4_r2t2
