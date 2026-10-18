#include "engine/models/vibevoice_asr/text_decoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/transformers/causal_decoder.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/kv_cache.h"

#include "engine/framework/core/constant_tensor_cache.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::vibevoice_asr {
namespace {

namespace binding = modules::binding;

constexpr int64_t kCacheInitialGraphSteps = 1024;
constexpr int64_t kCacheGraphGrowthStep = 512;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

modules::DecoderLayerWeights to_qwen2_layer_weights(
    const VibeVoiceQwen2LayerWeights & weights,
    core::ConstantTensorCache & constants) {
    modules::DecoderLayerWeights out;
    out.input_norm = binding::norm_data(constants, weights.input_norm);
    out.self_attention = weights.self_attention;
    out.post_norm = binding::norm_data(constants, weights.post_norm);
    out.mlp.gate_proj = weights.mlp.gate_proj;
    out.mlp.up_proj = weights.mlp.up_proj;
    out.mlp.down_proj = weights.mlp.down_proj;
    return out;
}

modules::CausalDecoderConfig make_qwen2_decoder_config(const VibeVoiceQwen2Config & config) {
    modules::CausalDecoderConfig out;
    out.stack.hidden_size = config.hidden_size;
    out.stack.num_attention_heads = config.num_attention_heads;
    out.stack.num_key_value_heads = config.num_key_value_heads;
    out.stack.head_dim = config.head_dim;
    out.stack.intermediate_size = config.intermediate_size;
    out.stack.layers = config.num_hidden_layers;
    out.stack.rms_norm_eps = config.rms_norm_eps;
    out.stack.rope_theta = config.rope_theta;
    out.stack.use_qk_norm = false;
    out.stack.runtime.attention.prefill_mode = modules::DecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.attention.static_mode = modules::DecoderAttentionMode::FlashGroupedViewKV;
    // The suffix path appends onto the prefix KV cache; without this it
    // takes the eager branch and materializes per-head F32 intermediates
    // that scale with the cache size.
    out.stack.runtime.attention.prefix_mode = modules::DecoderPrefixAttentionMode::FlashWithPrefix;
    out.stack.runtime.static_cache.update_mode = modules::DecoderStaticCacheUpdateMode::DirectSetRows;
    out.stack.runtime.static_cache.set_rows_mode = modules::DecoderStaticCacheSetRowsMode::BackendViewOptimized;
    out.logits_size = config.vocab_size;
    out.logits_mode = modules::CausalDecoderLogitsMode::LastStep;
    out.static_cache_type = GGML_TYPE_F16;
    return out;
}

modules::CausalDecoderWeights make_qwen2_decoder_weights(
    const VibeVoiceQwen2Weights & weights,
    core::ConstantTensorCache & constants) {
    modules::CausalDecoderWeights out;
    out.stack.layers.reserve(weights.layers.size());
    for (const auto & layer : weights.layers) {
        out.stack.layers.push_back(to_qwen2_layer_weights(layer, constants));
    }
    out.final_norm = binding::norm_data(constants, weights.norm);
    out.lm_head = binding::linear_data(constants, weights.lm_head);
    return out;
}

int64_t require_head_dim(const VibeVoiceQwen2Config & config) {
    if (config.hidden_size <= 0 || config.intermediate_size <= 0) {
        throw std::runtime_error("VibeVoice decoder hidden sizes must be positive");
    }
    if (config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 || config.head_dim <= 0) {
        throw std::runtime_error("VibeVoice decoder attention dimensions must be positive");
    }
    if (config.num_attention_heads % config.num_key_value_heads != 0) {
        throw std::runtime_error("VibeVoice decoder attention heads must be divisible by key/value heads");
    }
    if (config.num_attention_heads * config.head_dim != config.hidden_size) {
        throw std::runtime_error("VibeVoice decoder hidden_size must equal attention heads times head_dim");
    }
    return config.head_dim;
}

VibeVoiceQwen2LayerWeights load_layer_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const VibeVoiceQwen2Config & config,
    int64_t layer,
    assets::TensorStorageType weight_storage_type) {
    const int64_t dim = require_head_dim(config);
    const std::string prefix = "model.language_model.layers." + std::to_string(layer);
    VibeVoiceQwen2LayerWeights weights;
    weights.input_norm = source.require_f32_tensor(prefix + ".input_layernorm.weight", {config.hidden_size});
    weights.self_attention.q_weight = store.load_tensor(
        source,
        prefix + ".self_attn.q_proj.weight",
        weight_storage_type,
        {config.num_attention_heads * dim, config.hidden_size});
    weights.self_attention.q_bias = store.load_tensor(
        source,
        prefix + ".self_attn.q_proj.bias",
        assets::TensorStorageType::F32,
        {config.num_attention_heads * dim});
    weights.self_attention.k_weight = store.load_tensor(
        source,
        prefix + ".self_attn.k_proj.weight",
        weight_storage_type,
        {config.num_key_value_heads * dim, config.hidden_size});
    weights.self_attention.k_bias = store.load_tensor(
        source,
        prefix + ".self_attn.k_proj.bias",
        assets::TensorStorageType::F32,
        {config.num_key_value_heads * dim});
    weights.self_attention.v_weight = store.load_tensor(
        source,
        prefix + ".self_attn.v_proj.weight",
        weight_storage_type,
        {config.num_key_value_heads * dim, config.hidden_size});
    weights.self_attention.v_bias = store.load_tensor(
        source,
        prefix + ".self_attn.v_proj.bias",
        assets::TensorStorageType::F32,
        {config.num_key_value_heads * dim});
    weights.self_attention.out_weight = store.load_tensor(
        source,
        prefix + ".self_attn.o_proj.weight",
        weight_storage_type,
        {config.hidden_size, config.num_attention_heads * dim});
    weights.post_norm = source.require_f32_tensor(prefix + ".post_attention_layernorm.weight", {config.hidden_size});
    weights.mlp.gate_proj.weight = store.load_tensor(
        source,
        prefix + ".mlp.gate_proj.weight",
        weight_storage_type,
        {config.intermediate_size, config.hidden_size});
    weights.mlp.up_proj.weight = store.load_tensor(
        source,
        prefix + ".mlp.up_proj.weight",
        weight_storage_type,
        {config.intermediate_size, config.hidden_size});
    weights.mlp.down_proj.weight = store.load_tensor(
        source,
        prefix + ".mlp.down_proj.weight",
        weight_storage_type,
        {config.hidden_size, config.intermediate_size});
    return weights;
}

int64_t cache_graph_capacity(int64_t required_capacity, int64_t model_capacity) {
    if (required_capacity <= 0) {
        throw std::runtime_error("VibeVoice decoder cache capacity must be positive");
    }
    if (model_capacity > 0 && required_capacity > model_capacity) {
        throw std::runtime_error("VibeVoice decoder cache requirement exceeds model position capacity");
    }
    const int64_t requested = std::max<int64_t>(required_capacity, kCacheInitialGraphSteps);
    if (requested > std::numeric_limits<int64_t>::max() - (kCacheGraphGrowthStep - 1)) {
        throw std::runtime_error("VibeVoice decoder cache capacity overflow");
    }
    int64_t capacity = ((requested + kCacheGraphGrowthStep - 1) / kCacheGraphGrowthStep) *
        kCacheGraphGrowthStep;
    return model_capacity > 0 ? std::min(capacity, model_capacity) : capacity;
}

runtime::TransformerKVState empty_decoder_state(size_t layers) {
    runtime::TransformerKVState state;
    state.layers.resize(layers);
    return state;
}

}  // namespace

VibeVoiceQwen2Weights load_vibevoice_decoder_weights(
    const VibeVoiceASRAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type) {
    if (assets.model_weights == nullptr) {
        throw std::runtime_error("VibeVoice decoder requires model weights");
    }
    const auto & config = assets.config.decoder;
    require_head_dim(config);
    VibeVoiceQwen2Weights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "vibevoice.decoder.weights",
        weight_context_bytes);
    const auto queue_started = std::chrono::steady_clock::now();
    weights.token_embedding = weights.store->load_tensor(
        *assets.model_weights,
        "model.language_model.embed_tokens.weight",
        weight_storage_type,
        {config.vocab_size, config.hidden_size});
    if (config.tie_word_embeddings) {
        weights.lm_head = weights.token_embedding;
    } else {
        weights.lm_head = weights.store->load_tensor(
            *assets.model_weights,
            assets.model_weights->require_tensor_name({"lm_head.weight", "model.lm_head.weight"}),
            weight_storage_type,
            {config.vocab_size, config.hidden_size});
    }
    weights.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        weights.layers.push_back(load_layer_weights(
            *weights.store,
            *assets.model_weights,
            config,
            layer,
            weight_storage_type));
    }
    weights.norm = assets.model_weights->require_f32_tensor("model.language_model.norm.weight", {config.hidden_size});
    engine::debug::timing_log_scalar(
        "vibevoice.runtime.decoder_weight_queue_ms",
        engine::debug::elapsed_ms(queue_started));
    const auto upload_started = std::chrono::steady_clock::now();
    weights.store->upload();
    engine::debug::timing_log_scalar(
        "vibevoice.runtime.decoder_weight_upload_ms",
        engine::debug::elapsed_ms(upload_started));
    return weights;
}

class VibeVoiceQwen2EmbeddingGraph {
public:
    VibeVoiceQwen2EmbeddingGraph(
        const VibeVoiceQwen2WeightsRuntime & runtime,
        int64_t steps,
        size_t graph_arena_bytes)
        : runtime_(&runtime),
          steps_(steps) {
        if (steps_ <= 0) {
            throw std::runtime_error("VibeVoice decoder embedding graph requires positive steps");
        }
        const auto & config = runtime_->assets().config.decoder;
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VibeVoice decoder embedding graph context");
        }

        core::ModuleBuildContext ctx{ctx_.get(), "vibevoice.decoder.embed_tokens"};
        input_ids_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_I32, steps_, 1);
        auto ids = core::wrap_tensor(
            input_ids_,
            core::TensorShape::from_dims({1, steps_}),
            GGML_TYPE_I32);
        auto output = modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                          .build(ctx, ids, runtime_->weights().token_embedding);
        output = core::ensure_backend_addressable_layout(ctx, output);
        output_ = output.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 4096, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate VibeVoice decoder embedding graph");
        }
    }

    ~VibeVoiceQwen2EmbeddingGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_, true);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        engine::core::trim_backend_pools(runtime_->backend());
    }

    bool matches(const VibeVoiceQwen2WeightsRuntime & runtime, int64_t steps) const {
        return runtime_ == &runtime && steps_ == steps;
    }

    VibeVoiceTokenEmbeddings run(const std::vector<int32_t> & input_ids) {
        const auto & config = runtime_->assets().config.decoder;
        if (static_cast<int64_t>(input_ids.size()) != steps_) {
            throw std::runtime_error("VibeVoice decoder embedding input size mismatch");
        }
        ggml_backend_tensor_set(input_ids_, input_ids.data(), 0, input_ids.size() * sizeof(int32_t));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VibeVoice decoder embedding graph compute failed");
        }

        VibeVoiceTokenEmbeddings out;
        out.steps = steps_;
        out.hidden_size = config.hidden_size;
        out.values.resize(static_cast<size_t>(steps_ * config.hidden_size));
        ggml_backend_tensor_get(output_, out.values.data(), 0, out.values.size() * sizeof(float));
        return out;
    }

private:
    const VibeVoiceQwen2WeightsRuntime * runtime_ = nullptr;
    int64_t steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ids_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

class VibeVoiceQwen2PrefillGraph {
public:
    VibeVoiceQwen2PrefillGraph(
        const VibeVoiceQwen2WeightsRuntime & runtime,
        int64_t batch_size,
        int64_t prompt_steps,
        int64_t speech_tokens,
        size_t graph_arena_bytes)
        : runtime_(&runtime),
          batch_size_(batch_size),
          prompt_steps_(prompt_steps),
          speech_tokens_(speech_tokens) {
        if (batch_size_ <= 0) {
            throw std::runtime_error("VibeVoice decoder prefill graph requires positive batch size");
        }
        if (prompt_steps_ <= 0) {
            throw std::runtime_error("VibeVoice decoder prefill graph requires positive prompt steps");
        }
        if (speech_tokens_ < 0) {
            throw std::runtime_error("VibeVoice decoder prefill graph received invalid speech token count");
        }
        if (speech_tokens_ > 0 && batch_size_ != 1) {
            throw std::runtime_error("VibeVoice decoder speech prompt prefill currently expects batch size 1");
        }
        const auto & config = runtime_->assets().config.decoder;
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VibeVoice decoder prefill graph context");
        }

        core::ModuleBuildContext ctx{ctx_.get(), "vibevoice.decoder.prefill"};
        core::TensorValue x;
        if (speech_tokens_ > 0) {
            input_ids_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
            auto ids = core::wrap_tensor(input_ids_, core::TensorShape::from_dims({prompt_steps_}), GGML_TYPE_I32);
            x = modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                    .build(ctx, ids, runtime_->weights().token_embedding);
            speech_input_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_F32, config.hidden_size, speech_tokens_);
            auto speech = core::wrap_tensor(
                speech_input_,
                core::TensorShape::from_dims({speech_tokens_, config.hidden_size}),
                GGML_TYPE_F32);
            speech_positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I64, speech_tokens_);
            auto positions = core::wrap_tensor(
                speech_positions_,
                core::TensorShape::from_dims({speech_tokens_}),
                GGML_TYPE_I64);
            x = core::wrap_tensor(ggml_set_rows(ctx.ggml, x.tensor, speech.tensor, positions.tensor), x.shape, GGML_TYPE_F32);
            x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, prompt_steps_, config.hidden_size}));
        } else {
            x = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({batch_size_, prompt_steps_, config.hidden_size}));
            input_ = x.tensor;
        }
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
        auto positions_value = core::wrap_tensor(
            positions_,
            core::TensorShape::from_dims({prompt_steps_}),
            GGML_TYPE_I32);
        attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, prompt_steps_, prompt_steps_, 1, batch_size_);
        auto attention_mask = core::wrap_tensor(
            attention_mask_,
            core::TensorShape::from_dims({batch_size_, 1, prompt_steps_, prompt_steps_}),
            GGML_TYPE_F16);
        auto & constants = runtime_->constants();
        constants.begin_graph();
        auto decoder_out = modules::CausalDecoderModule(make_qwen2_decoder_config(config))
                               .build(
                                   ctx,
                                   x,
                                   positions_value,
                                   make_qwen2_decoder_weights(runtime_->weights(), constants),
                                   std::nullopt,
                                   attention_mask);
        for (const auto & layer : decoder_out.state.layers) {
            if (!layer.key.has_value() || !layer.value.has_value()) {
                throw std::runtime_error("VibeVoice decoder prefill did not return K/V state");
            }
            auto * key_readback = ggml_cpy(ctx_.get(), layer.key->tensor, ggml_dup_tensor(ctx_.get(), layer.key->tensor));
            auto * value_readback = ggml_cpy(ctx_.get(), layer.value->tensor, ggml_dup_tensor(ctx_.get(), layer.value->tensor));
            ggml_set_output(key_readback);
            ggml_set_output(value_readback);
            keys_.push_back(key_readback);
            values_.push_back(value_readback);
        }
        hidden_output_ = ggml_cpy(ctx_.get(), decoder_out.hidden.tensor, ggml_dup_tensor(ctx_.get(), decoder_out.hidden.tensor));
        ggml_set_output(hidden_output_);
        logits_output_ = decoder_out.logits.tensor;
        ggml_set_output(logits_output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        for (auto * key : keys_) {
            ggml_build_forward_expand(graph_, key);
        }
        for (auto * value : values_) {
            ggml_build_forward_expand(graph_, value);
        }
        ggml_build_forward_expand(graph_, hidden_output_);
        ggml_build_forward_expand(graph_, logits_output_);
        constants.finish_graph();
        constants.ensure_uploaded();
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate VibeVoice decoder prefill graph");
        }

        position_values_ = modules::decoder_position_ids(prompt_steps_);
        attention_mask_values_ = modules::causal_prefill_mask_values(batch_size_, prompt_steps_);
    }

    ~VibeVoiceQwen2PrefillGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_, true);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        engine::core::trim_backend_pools(runtime_->backend());
    }

    bool matches(
        const VibeVoiceQwen2WeightsRuntime & runtime,
        int64_t batch_size,
        int64_t prompt_steps,
        int64_t speech_tokens) const {
        return runtime_ == &runtime && batch_size_ == batch_size && prompt_steps_ == prompt_steps &&
            speech_tokens_ == speech_tokens;
    }

    VibeVoiceDecoderPrefillOutput run_prompt(
        const std::vector<int32_t> & input_ids,
        const std::vector<float> & speech_features,
        const std::vector<int32_t> & speech_positions) {
        const auto & config = runtime_->assets().config.decoder;
        if (batch_size_ != 1 || speech_tokens_ <= 0) {
            throw std::runtime_error("VibeVoice decoder prompt prefill graph is not configured for speech tokens");
        }
        if (static_cast<int64_t>(input_ids.size()) != prompt_steps_) {
            throw std::runtime_error("VibeVoice decoder prompt token count mismatch");
        }
        if (static_cast<int64_t>(speech_positions.size()) != speech_tokens_) {
            throw std::runtime_error("VibeVoice decoder prompt speech position count mismatch");
        }
        if (static_cast<int64_t>(speech_features.size()) != speech_tokens_ * config.hidden_size) {
            throw std::runtime_error("VibeVoice decoder prompt speech feature size mismatch");
        }
        std::vector<int64_t> positions64;
        positions64.reserve(speech_positions.size());
        for (const auto position : speech_positions) {
            positions64.push_back(position);
        }
        ggml_backend_tensor_set(input_ids_, input_ids.data(), 0, input_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(speech_input_, speech_features.data(), 0, speech_features.size() * sizeof(float));
        ggml_backend_tensor_set(speech_positions_, positions64.data(), 0, positions64.size() * sizeof(int64_t));
        auto results = run_batch_after_inputs_set();
        if (results.size() != 1) {
            throw std::runtime_error("VibeVoice decoder prompt prefill returned unexpected batch size");
        }
        return std::move(results.front());
    }

    VibeVoiceDecoderPrefillOutput run(const std::vector<float> & embeddings) {
        auto results = run_batch({embeddings});
        if (results.size() != 1) {
            throw std::runtime_error("VibeVoice decoder prefill graph returned unexpected batch size");
        }
        return std::move(results.front());
    }

    std::vector<VibeVoiceDecoderPrefillOutput> run_batch(const std::vector<std::vector<float>> & embeddings) {
        const auto & config = runtime_->assets().config.decoder;
        if (static_cast<int64_t>(embeddings.size()) != batch_size_) {
            throw std::runtime_error("VibeVoice decoder prefill embedding batch size mismatch");
        }
        const size_t per_sample = static_cast<size_t>(prompt_steps_ * config.hidden_size);
        for (const auto & sample : embeddings) {
            if (sample.size() != per_sample) {
                throw std::runtime_error("VibeVoice decoder prefill embedding size mismatch");
            }
        }
        std::vector<float> input(static_cast<size_t>(batch_size_) * per_sample, 0.0F);
        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            const auto & sample = embeddings[static_cast<size_t>(batch)];
            std::copy(
                sample.begin(),
                sample.end(),
                input.begin() + static_cast<std::ptrdiff_t>(batch * static_cast<int64_t>(per_sample)));
        }
        ggml_backend_tensor_set(input_, input.data(), 0, input.size() * sizeof(float));
        return run_batch_after_inputs_set();
    }

    std::vector<VibeVoiceDecoderPrefillOutput> run_batch_after_inputs_set() {
        const auto & config = runtime_->assets().config.decoder;
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        ggml_backend_tensor_set(positions_, position_values_.data(), 0, position_values_.size() * sizeof(int32_t));
        ggml_backend_tensor_set(
            attention_mask_,
            attention_mask_values_.data(),
            0,
            attention_mask_values_.size() * sizeof(ggml_fp16_t));
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VibeVoice decoder prefill graph compute failed");
        }

        const size_t logits_per_sample = static_cast<size_t>(config.vocab_size);
        const size_t hidden_per_sample = static_cast<size_t>(config.hidden_size);
        std::vector<float> logits(static_cast<size_t>(batch_size_) * logits_per_sample, 0.0F);
        std::vector<float> hidden(static_cast<size_t>(batch_size_) * hidden_per_sample, 0.0F);
        ggml_backend_tensor_get(logits_output_, logits.data(), 0, logits.size() * sizeof(float));
        ggml_backend_tensor_get(hidden_output_, hidden.data(), 0, hidden.size() * sizeof(float));

        std::vector<VibeVoiceDecoderPrefillOutput> out(static_cast<size_t>(batch_size_));
        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            auto & sample = out[static_cast<size_t>(batch)];
            sample.result.logits.vocab_size = config.vocab_size;
            sample.result.logits.values.assign(
                logits.begin() + static_cast<std::ptrdiff_t>(batch * static_cast<int64_t>(logits_per_sample)),
                logits.begin() + static_cast<std::ptrdiff_t>((batch + 1) * static_cast<int64_t>(logits_per_sample)));
            sample.result.last_hidden.dims = config.hidden_size;
            sample.result.last_hidden.values.assign(
                hidden.begin() + static_cast<std::ptrdiff_t>(batch * static_cast<int64_t>(hidden_per_sample)),
                hidden.begin() + static_cast<std::ptrdiff_t>((batch + 1) * static_cast<int64_t>(hidden_per_sample)));
            sample.state.current_end = prompt_steps_;
            sample.state.layers.resize(keys_.size());
        }

        const size_t layer_values = static_cast<size_t>(
            prompt_steps_ * config.num_key_value_heads * require_head_dim(config));
        for (size_t layer = 0; layer < keys_.size(); ++layer) {
            std::vector<float> key_values(static_cast<size_t>(batch_size_) * layer_values, 0.0F);
            std::vector<float> value_values(static_cast<size_t>(batch_size_) * layer_values, 0.0F);
            ggml_backend_tensor_get(keys_[layer], key_values.data(), 0, key_values.size() * sizeof(float));
            ggml_backend_tensor_get(values_[layer], value_values.data(), 0, value_values.size() * sizeof(float));
            for (int64_t batch = 0; batch < batch_size_; ++batch) {
                auto & state_layer = out[static_cast<size_t>(batch)].state.layers[layer];
                state_layer.valid_steps = prompt_steps_;
                const auto begin = static_cast<std::ptrdiff_t>(batch * static_cast<int64_t>(layer_values));
                const auto end = static_cast<std::ptrdiff_t>((batch + 1) * static_cast<int64_t>(layer_values));
                state_layer.key.assign(key_values.begin() + begin, key_values.begin() + end);
                state_layer.value.assign(value_values.begin() + begin, value_values.begin() + end);
            }
        }
        return out;
    }

private:
    const VibeVoiceQwen2WeightsRuntime * runtime_ = nullptr;
    int64_t batch_size_ = 0;
    int64_t prompt_steps_ = 0;
    int64_t speech_tokens_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ids_ = nullptr;
    ggml_tensor * speech_input_ = nullptr;
    ggml_tensor * speech_positions_ = nullptr;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * hidden_output_ = nullptr;
    ggml_tensor * logits_output_ = nullptr;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    std::vector<int32_t> position_values_;
    std::vector<ggml_fp16_t> attention_mask_values_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

class VibeVoiceQwen2KVCache {
public:
    VibeVoiceQwen2KVCache(
        const VibeVoiceQwen2WeightsRuntime & runtime,
        int64_t cache_steps)
        : runtime_(&runtime),
          cache_steps_(cache_steps) {
        if (cache_steps_ <= 0) {
            throw std::runtime_error("VibeVoice decoder KV cache requires positive capacity");
        }
        const auto & config = runtime_->assets().config.decoder;
        const int64_t step_elems = config.num_key_value_heads * require_head_dim(config);
        ggml_init_params params{4ull * 1024ull * 1024ull, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VibeVoice decoder KV cache context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "vibevoice.decoder.kv_cache"};
        std::vector<core::TensorValue> keys;
        std::vector<core::TensorValue> values;
        keys.reserve(runtime_->weights().layers.size());
        values.reserve(runtime_->weights().layers.size());
        for (size_t layer = 0; layer < runtime_->weights().layers.size(); ++layer) {
            keys.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F16,
                core::TensorShape::from_dims({1, cache_steps_, config.num_key_value_heads, config.head_dim})));
            values.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F16,
                core::TensorShape::from_dims({1, cache_steps_, config.num_key_value_heads, config.head_dim})));
        }
        runtime::TransformerKVCacheOptions cache_options;
        cache_options.allow_f16_storage = true;
        if (runtime_->max_history_steps() > 0) {
            cache_options.ring_mode = true;
            cache_options.ring_pinned_steps = runtime_->pinned_prefix_steps();
        }
        cache_ = runtime::TransformerKVCache(
            cache_steps_,
            step_elems,
            std::move(keys),
            std::move(values),
            cache_options);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate VibeVoice decoder KV cache");
        }
    }

    ~VibeVoiceQwen2KVCache() {
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool can_run(const VibeVoiceQwen2WeightsRuntime & runtime, int64_t required_steps) const {
        return runtime_ == &runtime && cache_steps_ >= required_steps;
    }

    int64_t cache_steps() const noexcept {
        return cache_.cache_steps();
    }

    int64_t valid_steps() const noexcept {
        return cache_.valid_steps();
    }

    int64_t current_end() const noexcept {
        return cache_.current_end();
    }

    int64_t slot_for_position(int64_t position) const {
        return cache_.slot_for_position(position);
    }

    void import_state(const runtime::TransformerKVState & state) {
        cache_.import_state(state);
    }

    runtime::TransformerKVState export_state() const {
        return cache_.export_state();
    }

    void advance_after_direct_append(int64_t steps) {
        cache_.advance_after_direct_append(steps);
    }

    const core::TensorValue & key_tensor(size_t layer) const {
        return cache_.key_tensor(layer);
    }

    const core::TensorValue & value_tensor(size_t layer) const {
        return cache_.value_tensor(layer);
    }

private:
    const VibeVoiceQwen2WeightsRuntime * runtime_ = nullptr;
    int64_t cache_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    runtime::TransformerKVCache cache_;
    ggml_backend_buffer_t buffer_ = nullptr;
};

class VibeVoiceQwen2CachedStepGraph {
public:
    VibeVoiceQwen2CachedStepGraph(
        const VibeVoiceQwen2WeightsRuntime & runtime,
        VibeVoiceQwen2KVCache & cache,
        size_t graph_arena_bytes)
        : runtime_(&runtime),
          cache_(&cache),
          cache_steps_(cache.cache_steps()) {
        if (cache_steps_ <= 0) {
            throw std::runtime_error("VibeVoice decoder cached step graph requires positive cache steps");
        }
        if (!cache_->can_run(runtime, cache_steps_)) {
            throw std::runtime_error("VibeVoice decoder cached step graph requires matching KV cache");
        }
        const auto & config = runtime_->assets().config.decoder;
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VibeVoice decoder cached step graph context");
        }

        core::ModuleBuildContext ctx{ctx_.get(), "vibevoice.decoder.cached_step"};
        auto x = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, config.hidden_size}));
        input_ = x.tensor;
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto positions_value = core::wrap_tensor(positions_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        cache_slot_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto cache_slot_value = core::wrap_tensor(cache_slot_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, cache_steps_, 1, 1, 1);
        auto attention_mask_value = core::wrap_tensor(
            attention_mask_,
            core::TensorShape::from_dims({1, 1, 1, cache_steps_}),
            GGML_TYPE_F16);

        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        auto & constants = runtime_->constants();
        constants.begin_graph();
        auto decoder_config = make_qwen2_decoder_config(config);
        auto layer_input = x;
        const modules::DecoderLayerModule layer_module(
            modules::decoder_layer_config_from_stack(decoder_config.stack));
        for (size_t layer_index = 0; layer_index < runtime_->weights().layers.size(); ++layer_index) {
            auto layer_out = layer_module.build_with_static_cache_tail(
                ctx,
                graph_,
                layer_input,
                positions_value,
                to_qwen2_layer_weights(runtime_->weights().layers[layer_index], constants),
                cache_->key_tensor(layer_index),
                cache_->value_tensor(layer_index),
                cache_slot_value,
                attention_mask_value);
            layer_input = layer_out.output;
        }
        auto hidden = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                          .build(ctx, layer_input, binding::norm_data(constants, runtime_->weights().norm));
        auto logits = modules::LinearModule({
                          config.hidden_size,
                          config.vocab_size,
                          false,
                      })
                          .build(ctx, hidden, {runtime_->weights().lm_head, std::nullopt});
        hidden_output_ = ggml_cpy(ctx_.get(), hidden.tensor, ggml_dup_tensor(ctx_.get(), hidden.tensor));
        ggml_set_output(hidden_output_);
        logits_output_ = logits.tensor;
        ggml_set_output(logits_output_);
        ggml_build_forward_expand(graph_, hidden_output_);
        ggml_build_forward_expand(graph_, logits_output_);
        constants.finish_graph();
        constants.ensure_uploaded();
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate VibeVoice decoder cached step graph");
        }
        attention_mask_buffer_.assign(static_cast<size_t>(cache_steps_), ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
    }

    ~VibeVoiceQwen2CachedStepGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_, true);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
        engine::core::trim_backend_pools(runtime_->backend());
    }

    bool can_decode(const VibeVoiceQwen2WeightsRuntime & runtime, int64_t required_capacity) const {
        return runtime_ == &runtime && cache_ != nullptr && cache_->can_run(runtime, required_capacity);
    }

    int64_t current_end() const noexcept {
        return cache_->current_end();
    }

    int64_t cache_steps() const noexcept {
        return cache_->cache_steps();
    }

    VibeVoiceDecoderResult run_step(const std::vector<float> & embedding) {
        run_step_no_readback(embedding);
        const auto & config = runtime_->assets().config.decoder;
        VibeVoiceDecoderResult out;
        out.logits.vocab_size = config.vocab_size;
        out.logits.values.resize(static_cast<size_t>(config.vocab_size));
        out.last_hidden.dims = config.hidden_size;
        out.last_hidden.values.resize(static_cast<size_t>(config.hidden_size));
        ggml_backend_tensor_get(logits_output_, out.logits.values.data(), 0, out.logits.values.size() * sizeof(float));
        ggml_backend_tensor_get(hidden_output_, out.last_hidden.values.data(), 0, out.last_hidden.values.size() * sizeof(float));
        ggml_backend_synchronize(runtime_->backend());
        return out;
    }

    void run_step_no_readback(const std::vector<float> & embedding) {
        const auto & config = runtime_->assets().config.decoder;
        if (static_cast<int64_t>(embedding.size()) != config.hidden_size) {
            throw std::runtime_error("VibeVoice decoder cached step embedding size mismatch");
        }
        if (cache_->valid_steps() >= cache_steps_ && runtime_->max_history_steps() <= 0) {
            throw std::runtime_error("VibeVoice decoder cached step exceeds cache capacity");
        }
        ggml_backend_tensor_set(input_, embedding.data(), 0, embedding.size() * sizeof(float));
        const int64_t position = cache_->current_end();
        const int32_t position_i32 = static_cast<int32_t>(position);
        ggml_backend_tensor_set(positions_, &position_i32, 0, sizeof(position_i32));
        const int32_t cache_slot = static_cast<int32_t>(cache_->slot_for_position(position));
        ggml_backend_tensor_set(cache_slot_, &cache_slot, 0, sizeof(cache_slot));
        {
            const auto masked = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
            const auto visible = ggml_fp32_to_fp16(0.0F);
            std::fill(attention_mask_buffer_.begin(), attention_mask_buffer_.end(), masked);
            const int64_t valid = cache_->valid_steps();
            const int64_t pinned = runtime_->pinned_prefix_steps();
            for (int64_t row = 0; row < valid; ++row) {
                const int64_t stored =
                    runtime::ring_stored_position(row, valid, cache_->current_end(), pinned);
                attention_mask_buffer_[static_cast<size_t>(cache_->slot_for_position(stored))] = visible;
            }
            attention_mask_buffer_[static_cast<size_t>(cache_slot)] = visible;
            ggml_backend_tensor_set(
                attention_mask_,
                attention_mask_buffer_.data(),
                0,
                attention_mask_buffer_.size() * sizeof(ggml_fp16_t));
        }

        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VibeVoice decoder cached step graph compute failed");
        }
        cache_->advance_after_direct_append(1);
    }

private:
    const VibeVoiceQwen2WeightsRuntime * runtime_ = nullptr;
    VibeVoiceQwen2KVCache * cache_ = nullptr;
    int64_t cache_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * cache_slot_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * hidden_output_ = nullptr;
    ggml_tensor * logits_output_ = nullptr;
    std::vector<ggml_fp16_t> attention_mask_buffer_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

class VibeVoiceQwen2CachedSuffixGraph {
public:
    VibeVoiceQwen2CachedSuffixGraph(
        const VibeVoiceQwen2WeightsRuntime & runtime,
        int64_t suffix_steps,
        VibeVoiceQwen2KVCache & cache,
        size_t graph_arena_bytes)
        : runtime_(&runtime),
          cache_(&cache),
          suffix_steps_(suffix_steps),
          cache_steps_(cache.cache_steps()) {
        if (suffix_steps_ <= 0) {
            throw std::runtime_error("VibeVoice decoder cached suffix graph requires positive suffix steps");
        }
        if (cache_steps_ <= 0) {
            throw std::runtime_error("VibeVoice decoder cached suffix graph requires positive cache steps");
        }
        if (!cache_->can_run(runtime, cache_steps_)) {
            throw std::runtime_error("VibeVoice decoder cached suffix graph requires matching KV cache");
        }
        const auto & config = runtime_->assets().config.decoder;
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VibeVoice decoder cached suffix graph context");
        }

        core::ModuleBuildContext ctx{ctx_.get(), "vibevoice.decoder.cached_suffix"};
        auto x = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, suffix_steps_, config.hidden_size}));
        input_ = x.tensor;
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, suffix_steps_);
        auto positions_value = core::wrap_tensor(
            positions_,
            core::TensorShape::from_dims({suffix_steps_}),
            GGML_TYPE_I32);
        cache_slot_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, suffix_steps_);
        auto cache_slot_value = core::wrap_tensor(
            cache_slot_,
            core::TensorShape::from_dims({suffix_steps_}),
            GGML_TYPE_I32);
        attention_key_steps_ = cache_steps_ + suffix_steps_;
        attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, attention_key_steps_, suffix_steps_, 1, 1);
        auto attention_mask_value = core::wrap_tensor(
            attention_mask_,
            core::TensorShape::from_dims({1, 1, suffix_steps_, attention_key_steps_}),
            GGML_TYPE_F16);

        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        auto & constants = runtime_->constants();
        constants.begin_graph();
        auto decoder_config = make_qwen2_decoder_config(config);
        const int64_t step_elems = config.num_key_value_heads * require_head_dim(config);
        auto layer_input = x;
        const modules::DecoderLayerModule layer_module(
            modules::decoder_layer_config_from_stack(decoder_config.stack));
        for (size_t layer_index = 0; layer_index < runtime_->weights().layers.size(); ++layer_index) {
            const auto & key_cache = cache_->key_tensor(layer_index);
            const auto & value_cache = cache_->value_tensor(layer_index);
            auto layer_out = layer_module.build(
                ctx,
                layer_input,
                positions_value,
                to_qwen2_layer_weights(runtime_->weights().layers[layer_index], constants),
                key_cache,
                value_cache,
                attention_mask_value);
            layer_input = layer_out.output;

            auto key_rows = core::ensure_backend_addressable_layout(ctx, layer_out.key);
            auto value_rows = core::ensure_backend_addressable_layout(ctx, layer_out.value);
            auto flat_key_cache = core::reshape_tensor(
                ctx,
                key_cache,
                core::TensorShape::from_dims({cache_steps_, step_elems}));
            auto flat_value_cache = core::reshape_tensor(
                ctx,
                value_cache,
                core::TensorShape::from_dims({cache_steps_, step_elems}));
            auto flat_key_rows = core::reshape_tensor(
                ctx,
                key_rows,
                core::TensorShape::from_dims({suffix_steps_, step_elems}));
            auto flat_value_rows = core::reshape_tensor(
                ctx,
                value_rows,
                core::TensorShape::from_dims({suffix_steps_, step_elems}));
            ggml_build_forward_expand(
                graph_,
                ggml_set_rows(ctx.ggml, flat_key_cache.tensor, flat_key_rows.tensor, cache_slot_value.tensor));
            ggml_build_forward_expand(
                graph_,
                ggml_set_rows(ctx.ggml, flat_value_cache.tensor, flat_value_rows.tensor, cache_slot_value.tensor));
        }

        auto last_layer_input = modules::SliceModule({1, suffix_steps_ - 1, 1}).build(ctx, layer_input);
        auto hidden = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                          .build(ctx, last_layer_input, binding::norm_data(constants, runtime_->weights().norm));
        auto logits = modules::LinearModule({
                          config.hidden_size,
                          config.vocab_size,
                          false,
                      })
                          .build(ctx, hidden, {runtime_->weights().lm_head, std::nullopt});
        hidden_output_ = ggml_cpy(ctx_.get(), hidden.tensor, ggml_dup_tensor(ctx_.get(), hidden.tensor));
        ggml_set_output(hidden_output_);
        logits_output_ = logits.tensor;
        ggml_set_output(logits_output_);
        ggml_build_forward_expand(graph_, hidden_output_);
        ggml_build_forward_expand(graph_, logits_output_);
        constants.finish_graph();
        constants.ensure_uploaded();
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate VibeVoice decoder cached suffix graph");
        }
        position_values_.resize(static_cast<size_t>(suffix_steps_));
        cache_slot_values_.resize(static_cast<size_t>(suffix_steps_));
        attention_mask_buffer_.resize(static_cast<size_t>(suffix_steps_ * attention_key_steps_));
    }

    ~VibeVoiceQwen2CachedSuffixGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_, true);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
        engine::core::trim_backend_pools(runtime_->backend());
    }

    bool can_decode(const VibeVoiceQwen2WeightsRuntime & runtime, int64_t suffix_steps, int64_t required_capacity) const {
        return runtime_ == &runtime && this->suffix_steps_ == suffix_steps && cache_ != nullptr &&
            cache_->can_run(runtime, required_capacity);
    }

    int64_t current_end() const noexcept {
        return cache_->current_end();
    }

    int64_t cache_steps() const noexcept {
        return cache_->cache_steps();
    }

    VibeVoiceDecoderResult run_suffix(const std::vector<float> & embeddings) {
        const auto & config = runtime_->assets().config.decoder;
        if (static_cast<int64_t>(embeddings.size()) != suffix_steps_ * config.hidden_size) {
            throw std::runtime_error("VibeVoice decoder cached suffix embedding size mismatch");
        }
        if (cache_->valid_steps() + suffix_steps_ > cache_steps_ && runtime_->max_history_steps() <= 0) {
            throw std::runtime_error("VibeVoice decoder cached suffix exceeds cache capacity");
        }
        ggml_backend_tensor_set(input_, embeddings.data(), 0, embeddings.size() * sizeof(float));
        const int64_t append_end = cache_->current_end();
        const int64_t append_valid = cache_->valid_steps();
        const int64_t pinned = runtime_->pinned_prefix_steps();
        for (int64_t step = 0; step < suffix_steps_; ++step) {
            position_values_[static_cast<size_t>(step)] = static_cast<int32_t>(append_end + step);
            cache_slot_values_[static_cast<size_t>(step)] =
                static_cast<int32_t>(cache_->slot_for_position(append_end + step));
        }
        ggml_backend_tensor_set(positions_, position_values_.data(), 0, position_values_.size() * sizeof(int32_t));
        ggml_backend_tensor_set(cache_slot_, cache_slot_values_.data(), 0, cache_slot_values_.size() * sizeof(int32_t));

        const auto masked = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        const auto visible = ggml_fp32_to_fp16(0.0F);
        std::fill(attention_mask_buffer_.begin(), attention_mask_buffer_.end(), masked);
        for (int64_t row = 0; row < append_valid; ++row) {
            const int64_t stored = runtime::ring_stored_position(row, append_valid, append_end, pinned);
            const size_t slot = static_cast<size_t>(cache_->slot_for_position(stored));
            for (int64_t query = 0; query < suffix_steps_; ++query) {
                attention_mask_buffer_[static_cast<size_t>(query * attention_key_steps_) + slot] = visible;
            }
        }
        for (int64_t row = 0; row < suffix_steps_; ++row) {
            for (int64_t step = 0; step <= row; ++step) {
                const size_t slot =
                    static_cast<size_t>(cache_->slot_for_position(append_end + step));
                attention_mask_buffer_[static_cast<size_t>(row * attention_key_steps_) + slot] = visible;
            }
        }
        ggml_backend_tensor_set(
            attention_mask_,
            attention_mask_buffer_.data(),
            0,
            attention_mask_buffer_.size() * sizeof(ggml_fp16_t));

        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VibeVoice decoder cached suffix graph compute failed");
        }
        cache_->advance_after_direct_append(suffix_steps_);

        std::vector<float> logits(static_cast<size_t>(config.vocab_size));
        std::vector<float> hidden(static_cast<size_t>(config.hidden_size));
        ggml_backend_tensor_get(logits_output_, logits.data(), 0, logits.size() * sizeof(float));
        ggml_backend_tensor_get(hidden_output_, hidden.data(), 0, hidden.size() * sizeof(float));
        ggml_backend_synchronize(runtime_->backend());

        VibeVoiceDecoderResult out;
        out.logits.vocab_size = config.vocab_size;
        out.logits.values = std::move(logits);
        out.last_hidden.dims = config.hidden_size;
        out.last_hidden.values = std::move(hidden);
        return out;
    }

private:
    const VibeVoiceQwen2WeightsRuntime * runtime_ = nullptr;
    VibeVoiceQwen2KVCache * cache_ = nullptr;
    int64_t suffix_steps_ = 0;
    int64_t cache_steps_ = 0;
    int64_t attention_key_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * cache_slot_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * hidden_output_ = nullptr;
    ggml_tensor * logits_output_ = nullptr;
    std::vector<int32_t> position_values_;
    std::vector<int32_t> cache_slot_values_;
    std::vector<ggml_fp16_t> attention_mask_buffer_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

VibeVoiceQwen2CachedState::VibeVoiceQwen2CachedState() = default;
VibeVoiceQwen2CachedState::~VibeVoiceQwen2CachedState() = default;
VibeVoiceQwen2CachedState::VibeVoiceQwen2CachedState(VibeVoiceQwen2CachedState &&) noexcept = default;
VibeVoiceQwen2CachedState & VibeVoiceQwen2CachedState::operator=(VibeVoiceQwen2CachedState &&) noexcept = default;

VibeVoiceQwen2WeightsRuntime::VibeVoiceQwen2WeightsRuntime(
    std::shared_ptr<const VibeVoiceASRAssets> assets,
    core::BackendType backend_type,
    int device,
    int threads,
    size_t weight_context_bytes,
    size_t constant_context_bytes,
    assets::TensorStorageType weight_storage_type,
    int64_t max_history_steps)
    : assets_(std::move(assets)),
      threads_(threads),
      max_history_steps_(max_history_steps) {
    if (assets_ == nullptr) {
        throw std::runtime_error("VibeVoice decoder weights runtime requires assets");
    }
    if (max_history_steps_ < 0) {
        throw std::runtime_error("VibeVoice decoder max_history_steps must be >= 0 (0 disables the history window)");
    }
    if (threads_ <= 0) {
        throw std::runtime_error("VibeVoice decoder weights runtime requires positive thread count");
    }
    const auto backend_started = std::chrono::steady_clock::now();
    backend_ = core::init_backend({backend_type, device, threads_});
    engine::debug::timing_log_scalar(
        "vibevoice.runtime.decoder_backend_init_ms",
        engine::debug::elapsed_ms(backend_started));
    const auto weights_started = std::chrono::steady_clock::now();
    weights_ = std::make_shared<VibeVoiceQwen2Weights>(
        load_vibevoice_decoder_weights(
            *assets_,
            backend_,
            backend_type,
            weight_context_bytes,
            weight_storage_type));
    engine::debug::timing_log_scalar(
        "vibevoice.runtime.decoder_weights_load_ms",
        engine::debug::elapsed_ms(weights_started));
    constants_ = std::make_unique<core::ConstantTensorCache>(
        backend_,
        threads_,
        "vibevoice.decoder.constants",
        constant_context_bytes);
}

VibeVoiceQwen2WeightsRuntime::~VibeVoiceQwen2WeightsRuntime() {
    prefill_graph_.reset();
    embedding_graph_.reset();
    constants_.reset();
    weights_.reset();
    if (backend_ != nullptr) {
        ggml_backend_free(backend_);
    }
}

const VibeVoiceASRAssets & VibeVoiceQwen2WeightsRuntime::assets() const noexcept {
    return *assets_;
}

const VibeVoiceQwen2Weights & VibeVoiceQwen2WeightsRuntime::weights() const noexcept {
    return *weights_;
}

ggml_backend_t VibeVoiceQwen2WeightsRuntime::backend() const noexcept {
    return backend_;
}

core::ConstantTensorCache & VibeVoiceQwen2WeightsRuntime::constants() const noexcept {
    return *constants_;
}

int VibeVoiceQwen2WeightsRuntime::threads() const noexcept {
    return threads_;
}

void VibeVoiceQwen2WeightsRuntime::set_pinned_prefix_steps(int64_t steps) {
    if (steps < 0) {
        throw std::runtime_error("VibeVoice decoder pinned prefix steps must be >= 0");
    }
    pinned_prefix_steps_ = steps;
}

int64_t VibeVoiceQwen2WeightsRuntime::pinned_prefix_steps() const noexcept {
    return pinned_prefix_steps_;
}

int64_t VibeVoiceQwen2WeightsRuntime::max_history_steps() const noexcept {
    return max_history_steps_;
}

int64_t VibeVoiceQwen2WeightsRuntime::cached_state_end_plus(
    const VibeVoiceQwen2CachedState & state,
    int64_t incoming_steps) {
    const int64_t current_end = state.cache_has_state_ && state.cache_ != nullptr
        ? state.cache_->current_end()
        : state.pending_state_.current_end;
    return current_end + incoming_steps;
}

int64_t VibeVoiceQwen2WeightsRuntime::apply_history_window(int64_t unbounded_required) const {
    // Clamp before tiering: tiering throws past model capacity, which a long
    // stream's absolute history can exceed while the window stays small.
    if (max_history_steps_ <= 0) {
        return cache_graph_capacity(unbounded_required, assets_->config.decoder.max_position_embeddings);
    }
    return cache_graph_capacity(max_history_steps_, assets_->config.decoder.max_position_embeddings);
}

VibeVoiceTokenEmbeddings VibeVoiceQwen2WeightsRuntime::embed_tokens(
    const std::vector<int32_t> & input_ids) const {
    if (input_ids.empty()) {
        throw std::runtime_error("VibeVoice decoder embedding requires at least one token");
    }
    const int64_t steps = static_cast<int64_t>(input_ids.size());
    if (embedding_graph_ == nullptr || !embedding_graph_->matches(*this, steps)) {
        embedding_graph_.reset();
        embedding_graph_ = std::make_unique<VibeVoiceQwen2EmbeddingGraph>(
            *this,
            steps,
            64ull * 1024ull * 1024ull);
    }
    return embedding_graph_->run(input_ids);
}

VibeVoiceDecoderPrefillOutput VibeVoiceQwen2WeightsRuntime::prefill_embeddings(
    const std::vector<float> & embeddings,
    int64_t steps) const {
    const auto & config = assets_->config.decoder;
    if (steps <= 0) {
        throw std::runtime_error("VibeVoice decoder prefill requires positive steps");
    }
    if (static_cast<int64_t>(embeddings.size()) != steps * config.hidden_size) {
        throw std::runtime_error("VibeVoice decoder prefill embedding payload size mismatch");
    }
    if (prefill_graph_ == nullptr || !prefill_graph_->matches(*this, 1, steps, 0)) {
        prefill_graph_.reset();
        prefill_graph_ = std::make_unique<VibeVoiceQwen2PrefillGraph>(
            *this,
            1,
            steps,
            0,
            1024ull * 1024ull * 1024ull);
    }
    return prefill_graph_->run(embeddings);
}

VibeVoiceDecoderPrefillOutput VibeVoiceQwen2WeightsRuntime::prefill_prompt(
    const std::vector<int32_t> & input_ids,
    const std::vector<float> & speech_features,
    const std::vector<int32_t> & speech_positions) const {
    const auto & config = assets_->config.decoder;
    if (input_ids.empty()) {
        throw std::runtime_error("VibeVoice decoder prompt prefill requires input tokens");
    }
    if (speech_positions.empty()) {
        throw std::runtime_error("VibeVoice decoder prompt prefill requires speech positions");
    }
    const int64_t steps = static_cast<int64_t>(input_ids.size());
    const int64_t speech_tokens = static_cast<int64_t>(speech_positions.size());
    if (static_cast<int64_t>(speech_features.size()) != speech_tokens * config.hidden_size) {
        throw std::runtime_error("VibeVoice decoder prompt prefill speech feature payload size mismatch");
    }
    if (prefill_graph_ == nullptr || !prefill_graph_->matches(*this, 1, steps, speech_tokens)) {
        prefill_graph_.reset();
        prefill_graph_ = std::make_unique<VibeVoiceQwen2PrefillGraph>(
            *this,
            1,
            steps,
            speech_tokens,
            1024ull * 1024ull * 1024ull);
    }
    return prefill_graph_->run_prompt(input_ids, speech_features, speech_positions);
}

void VibeVoiceQwen2WeightsRuntime::reset_cached_state(
    VibeVoiceQwen2CachedState & state,
    runtime::TransformerKVState prefill_state) const {
    state.graph_.reset();
    state.suffix_graph_.reset();
    state.cache_.reset();
    state.pending_state_ = std::move(prefill_state);
    state.cache_has_state_ = false;
}

void VibeVoiceQwen2WeightsRuntime::prepare_cached_state(
    VibeVoiceQwen2CachedState & state,
    int64_t cache_capacity) const {
    if (cache_capacity <= 0) {
        throw std::runtime_error("VibeVoice decoder cached state prepare requires positive cache capacity");
    }
    const int64_t required_capacity =
        apply_history_window(cache_capacity);
    if (state.cache_ != nullptr && state.cache_has_state_ && !state.cache_->can_run(*this, required_capacity)) {
        state.pending_state_ = state.cache_->export_state();
        state.cache_has_state_ = false;
        state.graph_.reset();
        state.suffix_graph_.reset();
        state.cache_.reset();
    }
    if (state.cache_ == nullptr || !state.cache_->can_run(*this, required_capacity)) {
        state.graph_.reset();
        state.suffix_graph_.reset();
        state.cache_ = std::make_unique<VibeVoiceQwen2KVCache>(*this, required_capacity);
    }
    if (!state.cache_has_state_) {
        if (state.pending_state_.layers.empty() && state.pending_state_.current_end == 0) {
            state.pending_state_ = empty_decoder_state(weights_->layers.size());
        }
        state.cache_->import_state(state.pending_state_);
        state.pending_state_ = {};
        state.cache_has_state_ = true;
    }
}

runtime::TransformerKVState VibeVoiceQwen2WeightsRuntime::export_cached_state(
    VibeVoiceQwen2CachedState & state) const {
    if (state.cache_has_state_ && state.cache_ != nullptr) {
        state.pending_state_ = state.cache_->export_state();
        state.cache_has_state_ = false;
    }
    return state.pending_state_;
}

void VibeVoiceQwen2WeightsRuntime::clone_cached_state(
    const VibeVoiceQwen2CachedState & source,
    VibeVoiceQwen2CachedState & target,
    int64_t cache_capacity) const {
    if (cache_capacity <= 0) {
        throw std::runtime_error("VibeVoice decoder cached state clone requires positive cache capacity");
    }
    if (!source.cache_has_state_ || source.cache_ == nullptr) {
        throw std::runtime_error("VibeVoice decoder cached state clone requires live source graph state");
    }
    const auto source_state = source.cache_->export_state();
    const int64_t required_capacity = apply_history_window(
        std::max<int64_t>(cache_capacity, source.cache_->current_end() + 1));
    if (target.cache_ == nullptr || !target.cache_->can_run(*this, required_capacity)) {
        target.graph_.reset();
        target.suffix_graph_.reset();
        target.cache_ = std::make_unique<VibeVoiceQwen2KVCache>(*this, required_capacity);
    }
    target.cache_->import_state(source_state);
    target.pending_state_ = {};
    target.cache_has_state_ = true;
}

VibeVoiceDecoderResult VibeVoiceQwen2WeightsRuntime::cached_step(
    const std::vector<float> & embedding,
    VibeVoiceQwen2CachedState & state,
    int64_t cache_capacity) const {
    const auto & config = assets_->config.decoder;
    if (static_cast<int64_t>(embedding.size()) != config.hidden_size) {
        throw std::runtime_error("VibeVoice decoder cached step embedding payload size mismatch");
    }
    if (cache_capacity <= 0) {
        throw std::runtime_error("VibeVoice decoder cached step requires positive cache capacity");
    }
    const int64_t required_capacity =
        apply_history_window(std::max<int64_t>(cache_capacity, cached_state_end_plus(state, 1)));
    if (state.cache_ != nullptr && state.cache_has_state_ && !state.cache_->can_run(*this, required_capacity)) {
        state.pending_state_ = state.cache_->export_state();
        state.cache_has_state_ = false;
        state.graph_.reset();
        state.suffix_graph_.reset();
        state.cache_.reset();
    }
    if (state.cache_ == nullptr || !state.cache_->can_run(*this, required_capacity)) {
        state.graph_.reset();
        state.suffix_graph_.reset();
        state.cache_ = std::make_unique<VibeVoiceQwen2KVCache>(*this, required_capacity);
    }
    if (!state.cache_has_state_) {
        if (state.pending_state_.layers.empty() && state.pending_state_.current_end == 0) {
            state.pending_state_ = empty_decoder_state(weights_->layers.size());
        }
        state.cache_->import_state(state.pending_state_);
        state.pending_state_ = {};
        state.cache_has_state_ = true;
    }
    state.suffix_graph_.reset();
    if (state.graph_ == nullptr || !state.graph_->can_decode(*this, required_capacity)) {
        state.graph_.reset();
        const size_t graph_arena_bytes = 1536ull * 1024ull * 1024ull;
        state.graph_ = std::make_unique<VibeVoiceQwen2CachedStepGraph>(
            *this,
            *state.cache_,
            graph_arena_bytes);
    }
    return state.graph_->run_step(embedding);
}

void VibeVoiceQwen2WeightsRuntime::append_cached_step(
    const std::vector<float> & embedding,
    VibeVoiceQwen2CachedState & state,
    int64_t cache_capacity) const {
    const auto & config = assets_->config.decoder;
    if (static_cast<int64_t>(embedding.size()) != config.hidden_size) {
        throw std::runtime_error("VibeVoice decoder cached append embedding payload size mismatch");
    }
    if (cache_capacity <= 0) {
        throw std::runtime_error("VibeVoice decoder cached append requires positive cache capacity");
    }
    const int64_t required_capacity =
        apply_history_window(std::max<int64_t>(cache_capacity, cached_state_end_plus(state, 1)));
    if (state.cache_ != nullptr && state.cache_has_state_ && !state.cache_->can_run(*this, required_capacity)) {
        state.pending_state_ = state.cache_->export_state();
        state.cache_has_state_ = false;
        state.graph_.reset();
        state.suffix_graph_.reset();
        state.cache_.reset();
    }
    if (state.cache_ == nullptr || !state.cache_->can_run(*this, required_capacity)) {
        state.graph_.reset();
        state.suffix_graph_.reset();
        state.cache_ = std::make_unique<VibeVoiceQwen2KVCache>(*this, required_capacity);
    }
    if (!state.cache_has_state_) {
        if (state.pending_state_.layers.empty() && state.pending_state_.current_end == 0) {
            state.pending_state_ = empty_decoder_state(weights_->layers.size());
        }
        state.cache_->import_state(state.pending_state_);
        state.pending_state_ = {};
        state.cache_has_state_ = true;
    }
    state.suffix_graph_.reset();
    if (state.graph_ == nullptr || !state.graph_->can_decode(*this, required_capacity)) {
        state.graph_.reset();
        const size_t graph_arena_bytes = 1536ull * 1024ull * 1024ull;
        state.graph_ = std::make_unique<VibeVoiceQwen2CachedStepGraph>(
            *this,
            *state.cache_,
            graph_arena_bytes);
    }
    state.graph_->run_step_no_readback(embedding);
}

VibeVoiceDecoderResult VibeVoiceQwen2WeightsRuntime::cached_suffix(
    const std::vector<float> & embeddings,
    int64_t steps,
    VibeVoiceQwen2CachedState & state,
    int64_t cache_capacity) const {
    const auto & config = assets_->config.decoder;
    if (steps <= 0) {
        throw std::runtime_error("VibeVoice decoder cached suffix requires positive steps");
    }
    if (static_cast<int64_t>(embeddings.size()) != steps * config.hidden_size) {
        throw std::runtime_error("VibeVoice decoder cached suffix embedding payload size mismatch");
    }
    if (cache_capacity <= 0) {
        throw std::runtime_error("VibeVoice decoder cached suffix requires positive cache capacity");
    }
    const int64_t required_capacity =
        apply_history_window(std::max<int64_t>(cache_capacity, cached_state_end_plus(state, steps)));
    if (state.cache_ != nullptr && state.cache_has_state_ && !state.cache_->can_run(*this, required_capacity)) {
        state.pending_state_ = state.cache_->export_state();
        state.cache_has_state_ = false;
        state.graph_.reset();
        state.suffix_graph_.reset();
        state.cache_.reset();
    }
    if (state.cache_ == nullptr || !state.cache_->can_run(*this, required_capacity)) {
        state.graph_.reset();
        state.suffix_graph_.reset();
        state.cache_ = std::make_unique<VibeVoiceQwen2KVCache>(*this, required_capacity);
    }
    if (!state.cache_has_state_) {
        if (state.pending_state_.layers.empty() && state.pending_state_.current_end == 0) {
            state.pending_state_ = empty_decoder_state(weights_->layers.size());
        }
        state.cache_->import_state(state.pending_state_);
        state.pending_state_ = {};
        state.cache_has_state_ = true;
    }
    state.graph_.reset();
    if (state.suffix_graph_ == nullptr || !state.suffix_graph_->can_decode(*this, steps, required_capacity)) {
        state.suffix_graph_.reset();
        const size_t graph_arena_bytes = 1536ull * 1024ull * 1024ull;
        state.suffix_graph_ = std::make_unique<VibeVoiceQwen2CachedSuffixGraph>(
            *this,
            steps,
            *state.cache_,
            graph_arena_bytes);
    }
    return state.suffix_graph_->run_suffix(embeddings);
}

}  // namespace engine::models::vibevoice_asr
