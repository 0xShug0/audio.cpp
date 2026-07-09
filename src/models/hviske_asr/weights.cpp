#include "engine/models/hviske_asr/weights.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/weight_binding.h"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::hviske_asr {
namespace {

using Clock = std::chrono::steady_clock;

namespace binding = engine::modules::binding;

engine::modules::LinearWeights load_linear_as_shape(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type,
    const std::vector<int64_t> & source_shape,
    int64_t out_features,
    int64_t in_features,
    bool use_bias) {
    engine::modules::LinearWeights weights;
    weights.weight = store.load_tensor_as_shape(
        source,
        prefix + ".weight",
        storage_type,
        source_shape,
        engine::core::TensorShape::from_dims({out_features, in_features}));
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    }
    return weights;
}

HviskeBatchNormEvalWeights load_fused_batch_norm(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    engine::assets::TensorStorageType storage_type) {
    const auto weight = source.require_f32(prefix + ".weight", {channels});
    const auto bias = source.require_f32(prefix + ".bias", {channels});
    const auto running_mean = source.require_f32(prefix + ".running_mean", {channels});
    const auto running_var = source.require_f32(prefix + ".running_var", {channels});
    std::vector<float> scale(static_cast<size_t>(channels), 0.0f);
    std::vector<float> fused_bias(static_cast<size_t>(channels), 0.0f);
    constexpr float eps = 1.0e-5f;
    for (int64_t i = 0; i < channels; ++i) {
        const auto index = static_cast<size_t>(i);
        const float channel_scale = weight[index] / std::sqrt(running_var[index] + eps);
        scale[index] = channel_scale;
        fused_bias[index] = bias[index] - running_mean[index] * channel_scale;
    }
    return {
        store.make_from_f32(engine::core::TensorShape::from_dims({channels}), storage_type, std::move(scale)),
        store.make_from_f32(engine::core::TensorShape::from_dims({channels}), storage_type, std::move(fused_bias)),
    };
}

engine::modules::RelativeAttentionWeights load_relative_attention(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type,
    int64_t hidden,
    int64_t heads,
    int64_t head_dim) {
    engine::modules::RelativeAttentionWeights weights;
    weights.attention.q_weight = store.load_tensor(source, prefix + ".linear_q.weight", storage_type, {hidden, hidden});
    weights.attention.q_bias = store.load_f32_tensor(source, prefix + ".linear_q.bias", {hidden});
    weights.attention.k_weight = store.load_tensor(source, prefix + ".linear_k.weight", storage_type, {hidden, hidden});
    weights.attention.k_bias = store.load_f32_tensor(source, prefix + ".linear_k.bias", {hidden});
    weights.attention.v_weight = store.load_tensor(source, prefix + ".linear_v.weight", storage_type, {hidden, hidden});
    weights.attention.v_bias = store.load_f32_tensor(source, prefix + ".linear_v.bias", {hidden});
    weights.attention.out_weight = store.load_tensor(source, prefix + ".linear_out.weight", storage_type, {hidden, hidden});
    weights.attention.out_bias = store.load_f32_tensor(source, prefix + ".linear_out.bias", {hidden});
    weights.pos_weight = store.load_tensor(source, prefix + ".linear_pos.weight", storage_type, {hidden, hidden});
    weights.pos_bias_u = store.load_f32_tensor(source, prefix + ".pos_bias_u", {heads, head_dim});
    weights.pos_bias_v = store.load_f32_tensor(source, prefix + ".pos_bias_v", {heads, head_dim});
    return weights;
}

HviskeEncoderLayerWeights load_encoder_layer(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const HviskeConfig & config,
    int64_t layer_index,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    const std::string prefix = "encoder.layers." + std::to_string(layer_index);
    const int64_t hidden = config.encoder.hidden_size;
    const int64_t intermediate = config.encoder.intermediate_size;
    const int64_t heads = config.encoder.heads;
    const int64_t head_dim = hidden / heads;
    const int64_t kernel = config.encoder.conv_kernel;

    HviskeEncoderLayerWeights layer;
    layer.norm_feed_forward1 = binding::norm_from_source(store, source, prefix + ".norm_feed_forward1", hidden);
    layer.norm_self_att = binding::norm_from_source(store, source, prefix + ".norm_self_att", hidden);
    layer.norm_conv = binding::norm_from_source(store, source, prefix + ".norm_conv", hidden);
    layer.norm_feed_forward2 = binding::norm_from_source(store, source, prefix + ".norm_feed_forward2", hidden);
    layer.norm_out = binding::norm_from_source(store, source, prefix + ".norm_out", hidden);
    layer.ff1_linear1 = binding::linear_from_source(store, source, prefix + ".feed_forward1.linear1", matmul_storage_type, intermediate, hidden, true);
    layer.ff1_linear2 = binding::linear_from_source(store, source, prefix + ".feed_forward1.linear2", matmul_storage_type, hidden, intermediate, true);
    layer.ff2_linear1 = binding::linear_from_source(store, source, prefix + ".feed_forward2.linear1", matmul_storage_type, intermediate, hidden, true);
    layer.ff2_linear2 = binding::linear_from_source(store, source, prefix + ".feed_forward2.linear2", matmul_storage_type, hidden, intermediate, true);
    layer.self_attn = load_relative_attention(store, source, prefix + ".self_attn", matmul_storage_type, hidden, heads, head_dim);
    layer.conv_pointwise1 = load_linear_as_shape(
        store,
        source,
        prefix + ".conv.pointwise_conv1",
        conv_storage_type,
        {2 * hidden, hidden, 1},
        2 * hidden,
        hidden,
        true);
    layer.conv_depthwise = {
        store.load_tensor(source, prefix + ".conv.depthwise_conv.weight", conv_storage_type, {hidden, 1, kernel}),
        store.load_f32_tensor(source, prefix + ".conv.depthwise_conv.bias", {hidden}),
    };
    layer.conv_norm = load_fused_batch_norm(store, source, prefix + ".conv.batch_norm", hidden, conv_storage_type);
    layer.conv_pointwise2 = load_linear_as_shape(
        store,
        source,
        prefix + ".conv.pointwise_conv2",
        conv_storage_type,
        {hidden, hidden, 1},
        hidden,
        hidden,
        true);
    return layer;
}

HviskeEncoderWeights load_encoder_weights(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const HviskeConfig & config,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    HviskeEncoderWeights encoder;
    const int64_t channels = config.encoder.subsampling_conv_channels;
    encoder.subsampling.conv0 = binding::conv2d_from_source(
        store,
        source,
        "encoder.pre_encode.conv.0",
        conv_storage_type,
        channels,
        1,
        3,
        3,
        true);
    encoder.subsampling.depthwise1_weight =
        store.load_tensor(source, "encoder.pre_encode.conv.2.weight", conv_storage_type, {channels, 1, 3, 3});
    encoder.subsampling.depthwise1_bias =
        store.load_f32_tensor(source, "encoder.pre_encode.conv.2.bias", {channels});
    encoder.subsampling.pointwise1 = binding::conv2d_from_source(
        store,
        source,
        "encoder.pre_encode.conv.3",
        conv_storage_type,
        channels,
        channels,
        1,
        1,
        true);
    encoder.subsampling.depthwise2_weight =
        store.load_tensor(source, "encoder.pre_encode.conv.5.weight", conv_storage_type, {channels, 1, 3, 3});
    encoder.subsampling.depthwise2_bias =
        store.load_f32_tensor(source, "encoder.pre_encode.conv.5.bias", {channels});
    encoder.subsampling.pointwise2 = binding::conv2d_from_source(
        store,
        source,
        "encoder.pre_encode.conv.6",
        conv_storage_type,
        channels,
        channels,
        1,
        1,
        true);
    encoder.subsampling.linear = binding::linear_from_source(
        store,
        source,
        "encoder.pre_encode.out",
        matmul_storage_type,
        config.encoder.hidden_size,
        channels * (config.encoder.feat_in / config.encoder.subsampling_factor),
        true);
    encoder.layers.reserve(static_cast<size_t>(config.encoder.layers));
    for (int64_t i = 0; i < config.encoder.layers; ++i) {
        encoder.layers.push_back(load_encoder_layer(store, source, config, i, matmul_storage_type, conv_storage_type));
    }
    encoder.encoder_projector = binding::linear_from_source(
        store,
        source,
        "encoder_decoder_proj",
        matmul_storage_type,
        config.decoder.hidden_size,
        config.encoder.hidden_size,
        true);
    return encoder;
}

engine::modules::AttentionWeights load_decoder_attention(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type,
    int64_t hidden) {
    engine::modules::AttentionWeights weights;
    weights.q_weight = store.load_tensor(source, prefix + ".query_net.weight", storage_type, {hidden, hidden});
    weights.q_bias = store.load_f32_tensor(source, prefix + ".query_net.bias", {hidden});
    weights.k_weight = store.load_tensor(source, prefix + ".key_net.weight", storage_type, {hidden, hidden});
    weights.k_bias = store.load_f32_tensor(source, prefix + ".key_net.bias", {hidden});
    weights.v_weight = store.load_tensor(source, prefix + ".value_net.weight", storage_type, {hidden, hidden});
    weights.v_bias = store.load_f32_tensor(source, prefix + ".value_net.bias", {hidden});
    weights.out_weight = store.load_tensor(source, prefix + ".out_projection.weight", storage_type, {hidden, hidden});
    weights.out_bias = store.load_f32_tensor(source, prefix + ".out_projection.bias", {hidden});
    return weights;
}

HviskeDecoderLayerWeights load_decoder_layer(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const HviskeConfig & config,
    int64_t layer_index,
    engine::assets::TensorStorageType storage_type) {
    const std::string prefix = "transf_decoder._decoder.layers." + std::to_string(layer_index);
    const int64_t hidden = config.decoder.hidden_size;
    HviskeDecoderLayerWeights layer;
    layer.self_norm = binding::norm_from_source(store, source, prefix + ".layer_norm_1", hidden);
    layer.self_attn = load_decoder_attention(store, source, prefix + ".first_sub_layer", storage_type, hidden);
    layer.cross_norm = binding::norm_from_source(store, source, prefix + ".layer_norm_2", hidden);
    layer.cross_attn = load_decoder_attention(store, source, prefix + ".second_sub_layer", storage_type, hidden);
    layer.ff_norm = binding::norm_from_source(store, source, prefix + ".layer_norm_3", hidden);
    layer.ff_in = binding::linear_from_source(store, source, prefix + ".third_sub_layer.dense_in", storage_type, config.decoder.intermediate_size, hidden, true);
    layer.ff_out = binding::linear_from_source(store, source, prefix + ".third_sub_layer.dense_out", storage_type, hidden, config.decoder.intermediate_size, true);
    return layer;
}

HviskeDecoderWeights load_decoder_weights(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const HviskeConfig & config,
    engine::assets::TensorStorageType storage_type) {
    HviskeDecoderWeights decoder;
    decoder.token_embedding = store.load_tensor(source, "transf_decoder._embedding.token_embedding.weight", storage_type, {config.decoder.vocab_size, config.decoder.hidden_size});
    decoder.position_embedding = store.load_tensor(source, "transf_decoder._embedding.position_embedding.pos_enc", storage_type, {config.decoder.max_sequence_length, config.decoder.hidden_size});
    decoder.embedding_norm = binding::norm_from_source(store, source, "transf_decoder._embedding.layer_norm", config.decoder.hidden_size);
    decoder.layers.reserve(static_cast<size_t>(config.decoder.layers));
    for (int64_t i = 0; i < config.decoder.layers; ++i) {
        decoder.layers.push_back(load_decoder_layer(store, source, config, i, storage_type));
    }
    decoder.final_norm = binding::norm_from_source(store, source, "transf_decoder._decoder.final_layer_norm", config.decoder.hidden_size);
    decoder.classifier = binding::linear_from_source(
        store,
        source,
        "log_softmax.mlp.layer0",
        storage_type,
        config.decoder.vocab_size,
        config.decoder.hidden_size,
        true);
    return decoder;
}

}  // namespace

std::shared_ptr<const HviskeWeights> load_hviske_weights(
    const HviskeAssets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes) {
    if (assets.model_weights == nullptr) {
        throw std::runtime_error("Hviske ASR tensor source must not be null");
    }
    const auto load_start = Clock::now();
    auto weights = std::make_shared<HviskeWeights>();
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        backend,
        backend_type,
        "hviske_asr.weights",
        weight_context_bytes);
    weights->encoder = load_encoder_weights(
        *weights->store,
        *assets.model_weights,
        assets.config,
        matmul_storage_type,
        conv_storage_type);
    weights->decoder = load_decoder_weights(
        *weights->store,
        *assets.model_weights,
        assets.config,
        matmul_storage_type);
    const auto upload_start = Clock::now();
    weights->store->upload();
    engine::debug::timing_log_scalar("hviske_asr.weights.upload_ms", engine::debug::elapsed_ms(upload_start, Clock::now()));
    engine::debug::timing_log_scalar("hviske_asr.weights.load_total_ms", engine::debug::elapsed_ms(load_start, Clock::now()));
    assets.model_weights->release_storage();
    return weights;
}

}  // namespace engine::models::hviske_asr
