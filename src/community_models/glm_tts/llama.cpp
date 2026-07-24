#include "engine/community_models/glm_tts/llama.h"

#include "engine/community_models/outetts/llama.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::glm_tts {
namespace {

std::shared_ptr<const outetts::OuteTTSAssets> make_llama_adapter(
    const std::shared_ptr<const GlmTTSAssets> & assets) {
    if (assets == nullptr || assets->llama_weights == nullptr) {
        throw std::runtime_error("GLM-TTS Llama runtime requires weights");
    }
    const auto & source = assets->config.llama;
    auto adapter = std::make_shared<outetts::OuteTTSAssets>();
    auto & target = adapter->config;
    target.bos_token_id = source.bos_token_id;
    target.eos_token_id = source.eos_token_id;
    target.pad_token_id = assets->config.pad_token;
    target.hidden_size = source.hidden_size;
    target.intermediate_size = source.intermediate_size;
    target.max_position_embeddings = source.max_position_embeddings;
    target.num_attention_heads = source.num_attention_heads;
    target.num_hidden_layers = source.num_hidden_layers;
    target.num_key_value_heads = source.num_key_value_heads;
    target.head_dim = source.head_dim;
    target.vocab_size = source.vocab_size;
    target.rms_norm_eps = source.rms_norm_eps;
    target.rope_theta = source.rope_theta;
    target.rope_scaling.factor = 1.0F;
    adapter->generation.max_length = source.max_position_embeddings;
    adapter->model_weights = assets->llama_weights;
    return adapter;
}

}  // namespace

struct GlmTTSLlamaRuntime::Impl {
    Impl(
        std::shared_ptr<const GlmTTSAssets> assets_in,
        core::BackendType backend_type,
        int device,
        int threads,
        assets::TensorStorageType weight_storage_type,
        size_t weight_context_bytes,
        size_t constant_context_bytes)
        : assets(std::move(assets_in)),
          adapter(make_llama_adapter(assets)),
          runtime(
              adapter,
              backend_type,
              device,
              threads,
              weight_context_bytes,
              constant_context_bytes,
              weight_storage_type) {}

    std::shared_ptr<const GlmTTSAssets> assets;
    std::shared_ptr<const outetts::OuteTTSAssets> adapter;
    outetts::OuteTTSLlamaRuntime runtime;
};

GlmTTSLlamaRuntime::GlmTTSLlamaRuntime(
    std::shared_ptr<const GlmTTSAssets> assets,
    core::BackendType backend_type,
    int device,
    int threads,
    assets::TensorStorageType weight_storage_type,
    size_t weight_context_bytes,
    size_t constant_context_bytes)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          backend_type,
          device,
          threads,
          weight_storage_type,
          weight_context_bytes,
          constant_context_bytes)) {}

GlmTTSLlamaRuntime::~GlmTTSLlamaRuntime() = default;

GlmTTSGenerateResult GlmTTSLlamaRuntime::generate(
    const GlmTTSPrompt & prompt,
    const GlmTTSGenerateOptions & options) const {
    if (prompt.input_ids.empty() ||
        prompt.minimum_audio_tokens < 0 ||
        prompt.maximum_audio_tokens <= 0) {
        throw std::runtime_error("GLM-TTS generation prompt is invalid");
    }
    const auto & config = impl_->assets->config;
    outetts::OuteTTSGenerateOptions generation;
    generation.max_new_tokens =
        options.max_new_tokens > 0
            ? std::min(options.max_new_tokens, prompt.maximum_audio_tokens)
            : prompt.maximum_audio_tokens;
    generation.minimum_new_tokens = std::min(
        prompt.minimum_audio_tokens,
        generation.max_new_tokens);
    generation.temperature = options.temperature;
    generation.repetition_penalty = 1.0F;
    generation.repetition_window = 0;
    generation.top_k = options.top_k;
    generation.top_p = options.top_p;
    generation.min_p = 0.0F;
    generation.seed = options.seed;
    generation.repetition_aware_sampling = true;
    generation.repetition_aware_window = 10;
    generation.repetition_aware_threshold = 1;
    generation.compact_sorted_multinomial = true;
    if (options.restrict_output_head) {
        generation.allowed_token_min =
            static_cast<int32_t>(config.audio_token_start);
        generation.allowed_token_max =
            static_cast<int32_t>(config.audio_token_end);
        generation.allowed_special_token =
            static_cast<int32_t>(config.end_audio_token);
    }

    const auto generated = impl_->runtime.generate(
        prompt.input_ids,
        generation,
        static_cast<int32_t>(config.llama.eos_token_id),
        static_cast<int32_t>(config.end_audio_token));
    GlmTTSGenerateResult result;
    result.speech_tokens.reserve(generated.tokens.size());
    for (const int32_t token : generated.tokens) {
        if (static_cast<int64_t>(token) == config.end_audio_token) {
            result.stopped_on_end_of_audio = true;
            break;
        }
        if (static_cast<int64_t>(token) < config.audio_token_start ||
            static_cast<int64_t>(token) > config.audio_token_end) {
            throw std::runtime_error(
                "GLM-TTS Llama generated a non-audio token");
        }
        result.speech_tokens.push_back(static_cast<int32_t>(
            static_cast<int64_t>(token) - config.audio_token_start));
    }
    if (result.speech_tokens.empty()) {
        throw std::runtime_error(
            "GLM-TTS Llama generated no speech tokens");
    }
    debug::trace_log_scalar(
        "glm_tts.llama.speech_tokens",
        static_cast<int64_t>(result.speech_tokens.size()));
    debug::trace_log_scalar(
        "glm_tts.llama.stopped_on_end_of_audio",
        result.stopped_on_end_of_audio);
    return result;
}

void GlmTTSLlamaRuntime::release_runtime_graph() {
    (void)impl_->runtime.release_cached_step_graph();
}

}  // namespace engine::models::glm_tts
