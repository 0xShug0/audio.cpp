#include "engine/models/parakeet_tdt/weights.h"

#include "engine/framework/modules/weight_binding.h"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>

namespace engine::models::parakeet_tdt {
namespace {

modules::LinearWeights load_linear_as_shape(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    const std::vector<int64_t> & source_weight_shape,
    int64_t out_features,
    int64_t in_features,
    bool use_bias = true) {
    modules::LinearWeights weights;
    weights.weight = store.load_tensor_as_shape(
        source,
        prefix + ".weight",
        storage_type,
        source_weight_shape,
        core::TensorShape::from_dims({out_features, in_features}));
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    }
    return weights;
}

BatchNorm1dEvalWeights load_fused_batch_norm(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    assets::TensorStorageType storage_type) {
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
        store.make_from_f32(core::TensorShape::from_dims({channels}), storage_type, std::move(scale)),
        store.make_from_f32(core::TensorShape::from_dims({channels}), storage_type, std::move(fused_bias)),
    };
}

modules::RelativeAttentionWeights load_relative_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t hidden,
    int64_t heads,
    int64_t head_dim) {
    modules::RelativeAttentionWeights weights;
    weights.attention.q_weight = store.load_tensor(source, prefix + ".q_proj.weight", storage_type, {hidden, hidden});
    weights.attention.k_weight = store.load_tensor(source, prefix + ".k_proj.weight", storage_type, {hidden, hidden});
    weights.attention.v_weight = store.load_tensor(source, prefix + ".v_proj.weight", storage_type, {hidden, hidden});
    weights.attention.out_weight = store.load_tensor(source, prefix + ".o_proj.weight", storage_type, {hidden, hidden});
    weights.pos_weight = store.load_tensor(source, prefix + ".relative_k_proj.weight", storage_type, {hidden, hidden});
    weights.pos_bias_u = store.load_f32_tensor(source, prefix + ".bias_u", {heads, head_dim});
    weights.pos_bias_v = store.load_f32_tensor(source, prefix + ".bias_v", {heads, head_dim});
    return weights;
}

ParakeetEncoderLayerWeights load_encoder_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    int64_t layer_index,
    const ParakeetModelConfig & config,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type) {
    const std::string prefix = "encoder.layers." + std::to_string(layer_index);
    const int64_t hidden = config.encoder.hidden_size;
    const int64_t intermediate = config.encoder.intermediate_size;
    const int64_t heads = config.encoder.num_attention_heads;
    const int64_t head_dim = hidden / heads;
    const int64_t kernel = config.encoder.conv_kernel_size;

    ParakeetEncoderLayerWeights layer;
    layer.norm_feed_forward1 = modules::binding::norm_from_source(store, source, prefix + ".norm_feed_forward1", hidden);
    layer.norm_self_att = modules::binding::norm_from_source(store, source, prefix + ".norm_self_att", hidden);
    layer.norm_conv = modules::binding::norm_from_source(store, source, prefix + ".norm_conv", hidden);
    layer.norm_feed_forward2 = modules::binding::norm_from_source(store, source, prefix + ".norm_feed_forward2", hidden);
    layer.norm_out = modules::binding::norm_from_source(store, source, prefix + ".norm_out", hidden);

    layer.ff1_linear1 = modules::binding::linear_from_source(store, source, prefix + ".feed_forward1.linear1", matmul_storage_type, intermediate, hidden, false);
    layer.ff1_linear2 = modules::binding::linear_from_source(store, source, prefix + ".feed_forward1.linear2", matmul_storage_type, hidden, intermediate, false);
    layer.ff2_linear1 = modules::binding::linear_from_source(store, source, prefix + ".feed_forward2.linear1", matmul_storage_type, intermediate, hidden, false);
    layer.ff2_linear2 = modules::binding::linear_from_source(store, source, prefix + ".feed_forward2.linear2", matmul_storage_type, hidden, intermediate, false);

    layer.self_attn = load_relative_attention(
        store,
        source,
        prefix + ".self_attn",
        matmul_storage_type,
        hidden,
        heads,
        head_dim);

    layer.conv_pointwise_conv1 = load_linear_as_shape(
        store,
        source,
        prefix + ".conv.pointwise_conv1",
        conv_storage_type,
        {2 * hidden, hidden, 1},
        2 * hidden,
        hidden,
        false);
    layer.conv_depthwise_conv = {
        store.load_tensor(source, prefix + ".conv.depthwise_conv.weight", conv_storage_type, {hidden, 1, kernel}),
        std::nullopt,
    };
    layer.conv_norm = load_fused_batch_norm(store, source, prefix + ".conv.norm", hidden, conv_storage_type);
    layer.conv_pointwise_conv2 = load_linear_as_shape(
        store,
        source,
        prefix + ".conv.pointwise_conv2",
        conv_storage_type,
        {hidden, hidden, 1},
        hidden,
        hidden,
        false);
    return layer;
}

ParakeetEncoderWeights load_encoder_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const ParakeetModelConfig & config,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type) {
    ParakeetEncoderWeights encoder;
    const int64_t channels = config.encoder.subsampling_conv_channels;
    const int64_t kernel = config.encoder.subsampling_conv_kernel_size;
    encoder.subsampling.conv0 = modules::binding::conv2d_from_source(
        store,
        source,
        "encoder.subsampling.layers.0",
        conv_storage_type,
        channels,
        1,
        kernel,
        kernel,
        true);
    encoder.subsampling.depthwise1_weight =
        store.load_tensor(source, "encoder.subsampling.layers.2.weight", conv_storage_type, {channels, 1, kernel, kernel});
    encoder.subsampling.depthwise1_bias =
        store.load_f32_tensor(source, "encoder.subsampling.layers.2.bias", {channels});
    encoder.subsampling.pointwise1 = modules::binding::conv2d_from_source(
        store,
        source,
        "encoder.subsampling.layers.3",
        conv_storage_type,
        channels,
        channels,
        1,
        1,
        true);
    encoder.subsampling.depthwise2_weight =
        store.load_tensor(source, "encoder.subsampling.layers.5.weight", conv_storage_type, {channels, 1, kernel, kernel});
    encoder.subsampling.depthwise2_bias =
        store.load_f32_tensor(source, "encoder.subsampling.layers.5.bias", {channels});
    encoder.subsampling.pointwise2 = modules::binding::conv2d_from_source(
        store,
        source,
        "encoder.subsampling.layers.6",
        conv_storage_type,
        channels,
        channels,
        1,
        1,
        true);
    const int64_t reduced_mels = config.encoder.num_mel_bins / config.encoder.subsampling_factor;
    encoder.subsampling.linear = modules::binding::linear_from_source(
        store,
        source,
        "encoder.subsampling.linear",
        matmul_storage_type,
        config.encoder.hidden_size,
        channels * reduced_mels,
        true);

    encoder.layers.reserve(static_cast<size_t>(config.encoder.num_hidden_layers));
    for (int64_t i = 0; i < config.encoder.num_hidden_layers; ++i) {
        encoder.layers.push_back(load_encoder_layer(
            store,
            source,
            i,
            config,
            matmul_storage_type,
            conv_storage_type));
    }
    encoder.encoder_projector = modules::binding::linear_from_source(
        store,
        source,
        "encoder_projector",
        matmul_storage_type,
        config.decoder_hidden_size,
        config.encoder.hidden_size,
        true);
    return encoder;
}

modules::LSTMCellWeights load_lstm_cell(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t hidden,
    int64_t layer) {
    const std::string suffix = "_l" + std::to_string(layer);
    return {
        store.load_tensor(source, prefix + ".weight_ih" + suffix, storage_type, {4 * hidden, hidden}),
        store.load_tensor(source, prefix + ".weight_hh" + suffix, storage_type, {4 * hidden, hidden}),
        store.load_f32_tensor(source, prefix + ".bias_ih" + suffix, {4 * hidden}),
        store.load_f32_tensor(source, prefix + ".bias_hh" + suffix, {4 * hidden}),
    };
}

ParakeetDecoderWeights load_decoder_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const ParakeetModelConfig & config,
    assets::TensorStorageType storage_type) {
    ParakeetDecoderWeights decoder;
    const int64_t hidden = config.decoder_hidden_size;
    decoder.embedding_weight = store.load_tensor(source, "decoder.embedding.weight", storage_type, {config.vocab_size, hidden});
    decoder.lstm0 = load_lstm_cell(store, source, "decoder.lstm", storage_type, hidden, 0);
    decoder.lstm1 = load_lstm_cell(store, source, "decoder.lstm", storage_type, hidden, 1);
    decoder.decoder_projector = modules::binding::linear_from_source(
        store,
        source,
        "decoder.decoder_projector",
        storage_type,
        hidden,
        hidden,
        true);
    return decoder;
}

ParakeetJointWeights load_joint_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const ParakeetModelConfig & config,
    assets::TensorStorageType storage_type) {
    ParakeetJointWeights joint;
    const int64_t logits = config.vocab_size + static_cast<int64_t>(config.durations.size());
    joint.head = modules::binding::linear_from_source(
        store,
        source,
        "joint.head",
        storage_type,
        logits,
        config.decoder_hidden_size,
        true);
    return joint;
}

}  // namespace

std::shared_ptr<const ParakeetTDTWeights> load_parakeet_tdt_weights(
    const ParakeetAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes) {
    if (assets.model_weights == nullptr) {
        throw std::runtime_error("Parakeet TDT tensor source must not be null");
    }
    auto weights = std::make_shared<ParakeetTDTWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "Parakeet TDT weights",
        weight_context_bytes);
    weights->encoder = load_encoder_weights(
        *weights->store,
        *assets.model_weights,
        assets.model_config,
        matmul_storage_type,
        conv_storage_type);
    weights->decoder = load_decoder_weights(
        *weights->store,
        *assets.model_weights,
        assets.model_config,
        matmul_storage_type);
    weights->joint = load_joint_weights(
        *weights->store,
        *assets.model_weights,
        assets.model_config,
        matmul_storage_type);
    weights->store->upload();
    return weights;
}

}  // namespace engine::models::parakeet_tdt
