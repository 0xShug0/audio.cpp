#include "engine/models/stable_audio/t5gemma.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::stable_audio {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kT5WeightContextBytes = 8ull * 1024ull * 1024ull;

modules::LinearWeights load_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_features,
    int64_t in_features,
    bool use_bias) {
    modules::LinearWeights weights;
    weights.weight = store.load_tensor(source, prefix + ".weight", storage_type, {out_features, in_features});
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    }
    return weights;
}

modules::T5GemmaEncoderLayerWeights load_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const StableAudioConfig & config,
    assets::TensorStorageType storage_type) {
    modules::T5GemmaEncoderLayerWeights weights;
    weights.pre_self_attn_norm = store.load_f32_tensor(source, prefix + ".pre_self_attn_layernorm.weight", {config.t5_hidden_size});
    weights.post_self_attn_norm = store.load_f32_tensor(source, prefix + ".post_self_attn_layernorm.weight", {config.t5_hidden_size});
    weights.pre_ff_norm = store.load_f32_tensor(source, prefix + ".pre_feedforward_layernorm.weight", {config.t5_hidden_size});
    weights.post_ff_norm = store.load_f32_tensor(source, prefix + ".post_feedforward_layernorm.weight", {config.t5_hidden_size});
    weights.q_proj = load_linear(
        store,
        source,
        prefix + ".self_attn.q_proj",
        storage_type,
        config.t5_attention_heads * config.t5_head_dim,
        config.t5_hidden_size,
        false);
    weights.k_proj = load_linear(
        store,
        source,
        prefix + ".self_attn.k_proj",
        storage_type,
        config.t5_kv_heads * config.t5_head_dim,
        config.t5_hidden_size,
        false);
    weights.v_proj = load_linear(
        store,
        source,
        prefix + ".self_attn.v_proj",
        storage_type,
        config.t5_kv_heads * config.t5_head_dim,
        config.t5_hidden_size,
        false);
    weights.o_proj = load_linear(
        store,
        source,
        prefix + ".self_attn.o_proj",
        storage_type,
        config.t5_hidden_size,
        config.t5_attention_heads * config.t5_head_dim,
        false);
    weights.gate_proj = load_linear(
        store,
        source,
        prefix + ".mlp.gate_proj",
        storage_type,
        config.t5_intermediate_size,
        config.t5_hidden_size,
        false);
    weights.up_proj = load_linear(
        store,
        source,
        prefix + ".mlp.up_proj",
        storage_type,
        config.t5_intermediate_size,
        config.t5_hidden_size,
        false);
    weights.down_proj = load_linear(
        store,
        source,
        prefix + ".mlp.down_proj",
        storage_type,
        config.t5_hidden_size,
        config.t5_intermediate_size,
        false);
    return weights;
}

modules::T5GemmaEncoderConfig encoder_config(const StableAudioConfig & config) {
    modules::T5GemmaEncoderConfig out;
    out.hidden_size = config.t5_hidden_size;
    out.layers = config.t5_layers;
    out.attention_heads = config.t5_attention_heads;
    out.kv_heads = config.t5_kv_heads;
    out.head_dim = config.t5_head_dim;
    out.intermediate_size = config.t5_intermediate_size;
    out.vocab_size = config.t5_vocab_size;
    out.rope_theta = config.t5_rope_theta;
    out.rms_norm_eps = config.t5_rms_norm_eps;
    out.attn_logit_softcap = config.t5_attn_logit_softcap;
    out.query_pre_attn_scalar = config.t5_query_pre_attn_scalar;
    out.scale_embeddings = true;
    return out;
}

}  // namespace

StableAudioT5Weights load_stable_audio_t5_weights(
    const StableAudioAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type) {
    const auto & config = assets.config;
    const auto & source = *assets.t5_weights;
    StableAudioT5Weights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "stable_audio.t5gemma.weights",
        weight_context_bytes == 0 ? kT5WeightContextBytes : weight_context_bytes);
    weights.encoder.embed_tokens = weights.store->load_tensor(
        source,
        "model.encoder.embed_tokens.weight",
        weight_storage_type,
        {config.t5_vocab_size, config.t5_hidden_size});
    weights.encoder.layers.reserve(static_cast<size_t>(config.t5_layers));
    for (int64_t layer = 0; layer < config.t5_layers; ++layer) {
        weights.encoder.layers.push_back(load_layer(
            *weights.store,
            source,
            "model.encoder.layers." + std::to_string(layer),
            config,
            weight_storage_type));
    }
    weights.encoder.norm = weights.store->load_f32_tensor(source, "model.encoder.norm.weight", {config.t5_hidden_size});
    weights.store->upload();
    return weights;
}

StableAudioT5EncoderRuntime::StableAudioT5EncoderRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const StableAudioAssets> assets,
    assets::TensorStorageType weight_storage_type)
    : execution_(&execution),
      assets_(std::move(assets)),
      weight_storage_type_(weight_storage_type) {
    if (execution_ == nullptr) {
        throw std::runtime_error("Stable Audio T5 runtime requires execution context");
    }
    if (assets_ == nullptr) {
        throw std::runtime_error("Stable Audio T5 runtime requires assets");
    }
}

StableAudioT5EncoderRuntime::~StableAudioT5EncoderRuntime() = default;

void StableAudioT5EncoderRuntime::prepare() const {
    if (!encoder_) {
        modules::T5EncoderRuntimeConfig config;
        config.label = "stable_audio.t5";
        config.pad_token_id = static_cast<int32_t>(assets_->config.t5_pad_token_id);
        encoder_ = std::make_unique<modules::T5EncoderRuntime>(*execution_, encoder_config(assets_->config),
            [assets = assets_, execution = execution_, storage = weight_storage_type_]() {
                auto weights = load_stable_audio_t5_weights(*assets, execution->backend(),
                    execution->backend_type(), 0, storage);
                assets->t5_weights->release_storage();
                return modules::T5EncoderRuntimeWeights{weights.store, std::move(weights.encoder), std::nullopt};
            }, config);
    }
}

StableAudioT5Encoding StableAudioT5EncoderRuntime::encode(const StableAudioTokenBatch & tokens) const {
    prepare();
    const auto start = Clock::now();
    if (tokens.tokens != assets_->config.prompt_max_length) {
        throw std::runtime_error("Stable Audio T5 token length must match configured prompt_max_length");
    }
    std::vector<std::vector<int32_t>> rows;
    std::vector<int64_t> lengths;
    rows.reserve(static_cast<size_t>(tokens.batch));
    lengths.reserve(static_cast<size_t>(tokens.batch));
    for (int64_t b = 0; b < tokens.batch; ++b) {
        const auto begin = tokens.input_ids.begin() + b * tokens.tokens;
        rows.emplace_back(begin, begin + tokens.tokens);
        const auto mask = tokens.attention_mask.begin() + b * tokens.tokens;
        lengths.push_back(std::count(mask, mask + tokens.tokens, uint8_t{1}));
    }
    const auto encoded = encoder_->encode_batch(rows, lengths);
    StableAudioT5Encoding output;
    output.batch = tokens.batch;
    output.tokens = tokens.tokens;
    output.hidden_size = assets_->config.t5_hidden_size;
    output.values.resize(static_cast<size_t>(tokens.batch * tokens.tokens * output.hidden_size));
    for (int64_t b = 0; b < tokens.batch; ++b) {
        std::copy(encoded[static_cast<size_t>(b)].begin(), encoded[static_cast<size_t>(b)].end(),
            output.values.begin() + b * tokens.tokens * output.hidden_size);
    }
    engine::debug::timing_log_scalar("stable_audio.t5.total_ms", engine::debug::elapsed_ms(start));
    return output;
}

void StableAudioT5EncoderRuntime::release_runtime_graphs() const {
    if (encoder_) encoder_->release_weights();
}

}  // namespace engine::models::stable_audio
