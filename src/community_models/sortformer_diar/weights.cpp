#include "engine/community_models/sortformer_diar/weights.h"

#include "engine/framework/modules/weight_binding.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::sortformer_diar {

namespace {

namespace modules = engine::modules;

modules::LinearWeights load_linear_as_shape(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    const std::vector<int64_t> & source_weight_shape,
    int64_t out_features,
    int64_t in_features) {
    modules::LinearWeights weights;
    weights.weight = store.load_tensor_as_shape(
        source,
        prefix + ".weight",
        storage_type,
        source_weight_shape,
        core::TensorShape::from_dims({out_features, in_features}));
    weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    return weights;
}

SortformerV2BatchNormWeights load_batch_norm(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    assets::TensorStorageType storage_type) {
    const auto weight = source.require_f32(prefix + ".weight", {channels});
    const auto bias = source.require_f32(prefix + ".bias", {channels});
    const auto running_mean = source.require_f32(prefix + ".running_mean", {channels});
    const auto running_var = source.require_f32(prefix + ".running_var", {channels});
    std::vector<float> scale(static_cast<size_t>(channels));
    std::vector<float> fused_bias(static_cast<size_t>(channels));
    constexpr float eps = 1.0e-5f;
    for (int64_t i = 0; i < channels; ++i) {
        const auto index = static_cast<size_t>(i);
        scale[index] = weight[index] / std::sqrt(running_var[index] + eps);
        fused_bias[index] = bias[index] - running_mean[index] * scale[index];
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

SortformerV2ConformerLayerWeights load_conformer_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    int64_t index,
    const SortformerV2ModelConfig & config,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type) {
    SortformerV2ConformerLayerWeights layer;
    const auto & encoder = config.fc_encoder;
    const std::string prefix = "enc.blocks." + std::to_string(index);
    const int64_t hidden = encoder.hidden_size;
    const int64_t intermediate = encoder.intermediate_size;
    const int64_t heads = encoder.num_attention_heads;
    const int64_t head_dim = hidden / heads;
    const int64_t kernel = encoder.conv_kernel_size;

    layer.norm_feed_forward1 = modules::binding::norm_from_source(store, source, prefix + ".norm_ff1", hidden);
    layer.norm_self_att = modules::binding::norm_from_source(store, source, prefix + ".norm_attn", hidden);
    layer.norm_conv = modules::binding::norm_from_source(store, source, prefix + ".norm_conv", hidden);
    layer.norm_feed_forward2 = modules::binding::norm_from_source(store, source, prefix + ".norm_ff2", hidden);
    layer.norm_out = modules::binding::norm_from_source(store, source, prefix + ".norm_out", hidden);
    layer.ff1_linear1 = modules::binding::linear_from_source(store, source, prefix + ".ff1.linear1", matmul_storage_type, intermediate, hidden, true);
    layer.ff1_linear2 = modules::binding::linear_from_source(store, source, prefix + ".ff1.linear2", matmul_storage_type, hidden, intermediate, true);
    layer.ff2_linear1 = modules::binding::linear_from_source(store, source, prefix + ".ff2.linear1", matmul_storage_type, intermediate, hidden, true);
    layer.ff2_linear2 = modules::binding::linear_from_source(store, source, prefix + ".ff2.linear2", matmul_storage_type, hidden, intermediate, true);
    layer.self_attn = load_relative_attention(store, source, prefix + ".attn", matmul_storage_type, hidden, heads, head_dim);
    layer.conv_pointwise_conv1 = load_linear_as_shape(
        store, source, prefix + ".conv.pointwise1", conv_storage_type, {2 * hidden, hidden, 1}, 2 * hidden, hidden);
    layer.conv_depthwise_conv = modules::binding::depthwise_conv1d_from_source(
        store, source, prefix + ".conv.depthwise", conv_storage_type, hidden, kernel, true);
    layer.conv_norm = load_batch_norm(store, source, prefix + ".conv.bn", hidden, conv_storage_type);
    layer.conv_pointwise_conv2 = load_linear_as_shape(
        store, source, prefix + ".conv.pointwise2", conv_storage_type, {hidden, hidden, 1}, hidden, hidden);
    return layer;
}

SortformerV2TransformerLayerWeights load_transformer_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    int64_t index,
    const SortformerV2ModelConfig & config,
    assets::TensorStorageType storage_type) {
    SortformerV2TransformerLayerWeights layer;
    const auto & transformer = config.tf_encoder;
    const std::string prefix = "tf.blocks." + std::to_string(index);
    const int64_t hidden = transformer.hidden_size;
    const int64_t intermediate = transformer.intermediate_size;
    layer.self_attn_layer_norm = modules::binding::norm_from_source(store, source, prefix + ".norm_1", hidden);
    layer.self_attn_q_proj = modules::binding::linear_from_source(store, source, prefix + ".attn.q", storage_type, hidden, hidden, true);
    layer.self_attn_k_proj = modules::binding::linear_from_source(store, source, prefix + ".attn.k", storage_type, hidden, hidden, true);
    layer.self_attn_v_proj = modules::binding::linear_from_source(store, source, prefix + ".attn.v", storage_type, hidden, hidden, true);
    layer.self_attn_out_proj = modules::binding::linear_from_source(store, source, prefix + ".attn.out", storage_type, hidden, hidden, true);
    layer.final_layer_norm = modules::binding::norm_from_source(store, source, prefix + ".norm_2", hidden);
    layer.fc1 = modules::binding::linear_from_source(store, source, prefix + ".ff.in", storage_type, intermediate, hidden, true);
    layer.fc2 = modules::binding::linear_from_source(store, source, prefix + ".ff.out", storage_type, hidden, intermediate, true);
    return layer;
}

}  // namespace

std::shared_ptr<const SortformerV2Weights> load_sortformer_v2_weights(
    const SortformerV2Assets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes) {
    if (assets.model_weights == nullptr) {
        throw std::runtime_error("Sortformer v2 tensor source must not be null");
    }
    const auto & source = *assets.model_weights;
    const auto & config = assets.model_config;
    auto weights = std::make_shared<SortformerV2Weights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend, backend_type, "sortformer v2.1 weights", weight_context_bytes);

    const auto & encoder = config.fc_encoder;
    weights->subsampling.conv0 = modules::binding::conv2d_from_source(
        *weights->store, source, "enc.pre_encode.conv.0", conv_storage_type,
        encoder.subsampling_conv_channels, 1, 3, 3, true);
    weights->subsampling.depthwise1_weight = weights->store->load_tensor(
        source, "enc.pre_encode.conv.2.weight", conv_storage_type,
        {encoder.subsampling_conv_channels, 1, 3, 3});
    weights->subsampling.depthwise1_bias = weights->store->load_f32_tensor(
        source, "enc.pre_encode.conv.2.bias", {encoder.subsampling_conv_channels});
    weights->subsampling.pointwise1 = modules::binding::conv2d_from_source(
        *weights->store, source, "enc.pre_encode.conv.3", conv_storage_type,
        encoder.subsampling_conv_channels, encoder.subsampling_conv_channels, 1, 1, true);
    weights->subsampling.depthwise2_weight = weights->store->load_tensor(
        source, "enc.pre_encode.conv.5.weight", conv_storage_type,
        {encoder.subsampling_conv_channels, 1, 3, 3});
    weights->subsampling.depthwise2_bias = weights->store->load_f32_tensor(
        source, "enc.pre_encode.conv.5.bias", {encoder.subsampling_conv_channels});
    weights->subsampling.pointwise2 = modules::binding::conv2d_from_source(
        *weights->store, source, "enc.pre_encode.conv.6", conv_storage_type,
        encoder.subsampling_conv_channels, encoder.subsampling_conv_channels, 1, 1, true);
    const int64_t reduced_mels = encoder.num_mel_bins / encoder.subsampling_factor;
    weights->subsampling.linear = modules::binding::linear_from_source(
        *weights->store, source, "enc.pre_encode.out", matmul_storage_type,
        encoder.hidden_size, encoder.subsampling_conv_channels * reduced_mels, true);

    weights->conformer_layers.reserve(static_cast<size_t>(encoder.num_hidden_layers));
    for (int64_t i = 0; i < encoder.num_hidden_layers; ++i) {
        weights->conformer_layers.push_back(
            load_conformer_layer(*weights->store, source, i, config, matmul_storage_type, conv_storage_type));
    }

    const auto & transformer = config.tf_encoder;
    weights->transformer_layers.reserve(static_cast<size_t>(transformer.num_hidden_layers));
    for (int64_t i = 0; i < transformer.num_hidden_layers; ++i) {
        weights->transformer_layers.push_back(
            load_transformer_layer(*weights->store, source, i, config, matmul_storage_type));
    }

    weights->head.encoder_proj = modules::binding::linear_from_source(
        *weights->store, source, "diar.encoder_proj", matmul_storage_type,
        config.modules.tf_d_model, config.modules.fc_d_model, true);
    weights->head.first_hidden_to_hidden = modules::binding::linear_from_source(
        *weights->store, source, "diar.fc1", matmul_storage_type,
        config.modules.tf_d_model, config.modules.tf_d_model, true);
    weights->head.single_hidden_to_spks = modules::binding::linear_from_source(
        *weights->store, source, "diar.single_spk_head", matmul_storage_type,
        config.modules.num_speakers, config.modules.tf_d_model, true);
    weights->store->upload();
    return weights;
}

}  // namespace engine::community_models::sortformer_diar
