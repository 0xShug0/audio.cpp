#include "engine/models/moonshine_stt/weights.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/packed_linear_weights.h"
#include "engine/framework/modules/weight_binding.h"

#include <chrono>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::moonshine_stt {
namespace {

using Clock = std::chrono::steady_clock;
namespace binding = engine::modules::binding;

engine::modules::NormWeights load_encoder_layer_norm(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden_size) {
    auto gamma = source.require_f32(prefix + ".gamma", {hidden_size});
    for (float & value : gamma) {
        value += 1.0F;
    }
    engine::modules::NormWeights weights;
    weights.weight = store.make_f32(engine::core::TensorShape::from_dims({hidden_size}), std::move(gamma));
    return weights;
}

engine::modules::AttentionWeights load_attention(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type,
    int64_t hidden_size,
    int64_t heads,
    int64_t kv_heads,
    int64_t head_dim,
    bool use_bias) {
    engine::modules::AttentionWeights weights;
    auto q_proj =
        binding::linear_from_source(store, source, prefix + ".q_proj", storage_type, heads * head_dim, hidden_size, use_bias);
    auto k_proj =
        binding::linear_from_source(store, source, prefix + ".k_proj", storage_type, kv_heads * head_dim, hidden_size, use_bias);
    auto v_proj =
        binding::linear_from_source(store, source, prefix + ".v_proj", storage_type, kv_heads * head_dim, hidden_size, use_bias);
    auto o_proj =
        binding::linear_from_source(store, source, prefix + ".o_proj", storage_type, hidden_size, heads * head_dim, false);
    weights.q_weight = q_proj.weight;
    weights.q_bias = q_proj.bias;
    weights.k_weight = k_proj.weight;
    weights.k_bias = k_proj.bias;
    weights.v_weight = v_proj.weight;
    weights.v_bias = v_proj.bias;
    weights.out_weight = o_proj.weight;
    weights.out_bias = o_proj.bias;
    return weights;
}

engine::modules::AttentionWeights load_packed_self_attention(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type,
    int64_t hidden_size,
    int64_t heads,
    int64_t kv_heads,
    int64_t head_dim,
    bool use_bias) {
    engine::modules::AttentionWeights weights;
    const int64_t q_features = heads * head_dim;
    const int64_t kv_features = kv_heads * head_dim;
    auto qkv_proj = engine::modules::PackedLinearWeightsBuilder({
        hidden_size,
        {
            {prefix + ".q_proj.weight", use_bias ? std::make_optional(prefix + ".q_proj.bias") : std::nullopt, q_features},
            {prefix + ".k_proj.weight", use_bias ? std::make_optional(prefix + ".k_proj.bias") : std::nullopt, kv_features},
            {prefix + ".v_proj.weight", use_bias ? std::make_optional(prefix + ".v_proj.bias") : std::nullopt, kv_features},
        },
        use_bias,
    }).build(store, source, storage_type);
    auto o_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".o_proj",
        storage_type,
        hidden_size,
        q_features,
        false);
    weights.qkv_weight = qkv_proj.weight;
    weights.qkv_bias = qkv_proj.bias;
    weights.out_weight = o_proj.weight;
    weights.out_bias = o_proj.bias;
    return weights;
}

MoonshineEncoderLayerWeights load_encoder_layer(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const MoonshineEncoderConfig & config,
    int64_t layer,
    engine::assets::TensorStorageType storage_type) {
    const std::string prefix = "model.encoder.layers." + std::to_string(layer);
    MoonshineEncoderLayerWeights weights;
    weights.input_norm = load_encoder_layer_norm(store, source, prefix + ".input_layernorm", config.hidden_size);
    weights.self_attn = load_packed_self_attention(
        store,
        source,
        prefix + ".self_attn",
        storage_type,
        config.hidden_size,
        config.heads,
        config.kv_heads,
        config.head_dim,
        config.attention_bias);
    weights.post_attention_norm =
        load_encoder_layer_norm(store, source, prefix + ".post_attention_layernorm", config.hidden_size);
    auto fc1 = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.fc1",
        storage_type,
        config.intermediate_size,
        config.hidden_size,
        true);
    auto fc2 = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.fc2",
        storage_type,
        config.hidden_size,
        config.intermediate_size,
        true);
    weights.feed_forward = {
        fc1.weight,
        fc1.bias,
        fc2.weight,
        fc2.bias,
    };
    return weights;
}

MoonshineDecoderLayerWeights load_decoder_layer(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const MoonshineDecoderConfig & config,
    int64_t layer,
    engine::assets::TensorStorageType storage_type) {
    const std::string prefix = "model.decoder.layers." + std::to_string(layer);
    MoonshineDecoderLayerWeights weights;
    weights.input_norm = binding::norm_weight_from_source(store, source, prefix + ".input_layernorm", config.hidden_size);
    weights.self_attn = load_attention(
        store,
        source,
        prefix + ".self_attn",
        storage_type,
        config.hidden_size,
        config.heads,
        config.kv_heads,
        config.head_dim,
        config.attention_bias);
    weights.post_attention_norm =
        binding::norm_weight_from_source(store, source, prefix + ".post_attention_layernorm", config.hidden_size);
    weights.cross_attn = load_attention(
        store,
        source,
        prefix + ".encoder_attn",
        storage_type,
        config.hidden_size,
        config.heads,
        config.kv_heads,
        config.head_dim,
        config.attention_bias);
    weights.final_norm = binding::norm_weight_from_source(store, source, prefix + ".final_layernorm", config.hidden_size);
    weights.mlp_fc1 = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.fc1",
        storage_type,
        config.intermediate_size * 2,
        config.hidden_size,
        true);
    weights.mlp_fc2 = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.fc2",
        storage_type,
        config.hidden_size,
        config.intermediate_size,
        true);
    return weights;
}

}  // namespace

std::shared_ptr<const MoonshineWeights> load_moonshine_stt_weights(
    const MoonshineAssets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType encoder_storage_type,
    engine::assets::TensorStorageType decoder_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes) {
    const auto start = Clock::now();
    auto weights = std::make_shared<MoonshineWeights>();
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        backend,
        backend_type,
        "Moonshine STT",
        weight_context_bytes);
    const auto & source = *assets.source;
    const auto & enc = assets.config.encoder;
    const auto & dec = assets.config.decoder;

    const auto log_k = source.require_f32("model.encoder.embedder.comp.log_k");
    if (log_k.size() != 1) {
        throw std::runtime_error("Moonshine STT log_k must contain one scalar");
    }
    weights->encoder.frontend.log_k = log_k.front();
    weights->encoder.frontend.linear = binding::linear_from_source(
        *weights->store,
        source,
        "model.encoder.embedder.linear",
        encoder_storage_type,
        enc.hidden_size,
        enc.frame_length,
        false);
    weights->encoder.frontend.conv1 = binding::conv1d_from_source(
        *weights->store,
        source,
        "model.encoder.embedder.conv1",
        conv_storage_type,
        enc.hidden_size * 2,
        enc.hidden_size,
        5,
        true);
    weights->encoder.frontend.conv2 = binding::conv1d_from_source(
        *weights->store,
        source,
        "model.encoder.embedder.conv2",
        conv_storage_type,
        enc.hidden_size,
        enc.hidden_size * 2,
        5,
        true);

    weights->encoder.layers.reserve(static_cast<size_t>(enc.layers));
    for (int64_t i = 0; i < enc.layers; ++i) {
        weights->encoder.layers.push_back(load_encoder_layer(*weights->store, source, enc, i, encoder_storage_type));
    }
    weights->encoder.final_norm = load_encoder_layer_norm(*weights->store, source, "model.encoder.final_norm", enc.hidden_size);

    weights->decoder.token_embedding =
        weights->store->load_tensor(source, "model.decoder.embed_tokens.weight", decoder_storage_type, {dec.vocab_size, dec.hidden_size});
    weights->decoder.position_embedding =
        weights->store->load_f32_tensor(source, "model.decoder.pos_emb.weight", {dec.max_position_embeddings, enc.hidden_size});
    if (source.has_tensor("model.decoder.proj.weight")) {
        weights->decoder.adapter_proj =
            binding::linear_from_source(
                *weights->store,
                source,
                "model.decoder.proj",
                decoder_storage_type,
                dec.hidden_size,
                enc.hidden_size,
                false);
    }
    weights->decoder.layers.reserve(static_cast<size_t>(dec.layers));
    for (int64_t i = 0; i < dec.layers; ++i) {
        weights->decoder.layers.push_back(load_decoder_layer(*weights->store, source, dec, i, decoder_storage_type));
    }
    weights->decoder.norm =
        binding::norm_weight_from_source(*weights->store, source, "model.decoder.norm", dec.hidden_size);
    const std::string head = source.has_tensor("proj_out.weight") ? "proj_out" : "model.proj_out";
    weights->decoder.output_projection =
        binding::linear_from_source(
            *weights->store,
            source,
            head,
            decoder_storage_type,
            dec.vocab_size,
            dec.hidden_size,
            false);
    weights->store->upload();
    debug::timing_log_scalar("moonshine_stt.weights_load_ms", engine::debug::elapsed_ms(start, Clock::now()));
    return weights;
}

}  // namespace engine::models::moonshine_stt
