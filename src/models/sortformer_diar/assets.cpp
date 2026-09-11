#include "engine/models/sortformer_diar/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/io/json.h"
#include "engine/framework/modules/weight_binding.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::sortformer_diar {

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
    weights.attention.q_bias = store.load_f32_tensor(source, prefix + ".q_proj.bias", {hidden});
    weights.attention.k_weight = store.load_tensor(source, prefix + ".k_proj.weight", storage_type, {hidden, hidden});
    weights.attention.k_bias = store.load_f32_tensor(source, prefix + ".k_proj.bias", {hidden});
    weights.attention.v_weight = store.load_tensor(source, prefix + ".v_proj.weight", storage_type, {hidden, hidden});
    weights.attention.v_bias = store.load_f32_tensor(source, prefix + ".v_proj.bias", {hidden});
    weights.attention.out_weight = store.load_tensor(source, prefix + ".o_proj.weight", storage_type, {hidden, hidden});
    weights.attention.out_bias = store.load_f32_tensor(source, prefix + ".o_proj.bias", {hidden});
    weights.pos_weight = store.load_tensor(source, prefix + ".relative_k_proj.weight", storage_type, {hidden, hidden});
    weights.pos_bias_u = store.load_f32_tensor(source, prefix + ".bias_u", {heads, head_dim});
    weights.pos_bias_v = store.load_f32_tensor(source, prefix + ".bias_v", {heads, head_dim});
    return weights;
}

SortformerConformerLayerWeights load_conformer_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    int64_t layer_index,
    const SortformerModelConfig & config,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type) {
    SortformerConformerLayerWeights layer;
    const auto & encoder = config.fc_encoder;
    const std::string prefix = "fc_encoder.layers." + std::to_string(layer_index);
    const int64_t hidden = encoder.hidden_size;
    const int64_t intermediate = encoder.intermediate_size;
    const int64_t heads = encoder.num_attention_heads;
    const int64_t head_dim = hidden / heads;
    const int64_t kernel = encoder.conv_kernel_size;

    layer.norm_feed_forward1 = modules::binding::norm_from_source(store, source, prefix + ".norm_feed_forward1", hidden);
    layer.norm_self_att = modules::binding::norm_from_source(store, source, prefix + ".norm_self_att", hidden);
    layer.norm_conv = modules::binding::norm_from_source(store, source, prefix + ".norm_conv", hidden);
    layer.norm_feed_forward2 = modules::binding::norm_from_source(store, source, prefix + ".norm_feed_forward2", hidden);
    layer.norm_out = modules::binding::norm_from_source(store, source, prefix + ".norm_out", hidden);

    layer.ff1_linear1 = modules::binding::linear_from_source(store, source, prefix + ".feed_forward1.linear1", matmul_storage_type, intermediate, hidden, true);
    layer.ff1_linear2 = modules::binding::linear_from_source(store, source, prefix + ".feed_forward1.linear2", matmul_storage_type, hidden, intermediate, true);
    layer.ff2_linear1 = modules::binding::linear_from_source(store, source, prefix + ".feed_forward2.linear1", matmul_storage_type, intermediate, hidden, true);
    layer.ff2_linear2 = modules::binding::linear_from_source(store, source, prefix + ".feed_forward2.linear2", matmul_storage_type, hidden, intermediate, true);

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
        hidden);
    layer.conv_depthwise_conv = modules::binding::depthwise_conv1d_from_source(store, source, prefix + ".conv.depthwise_conv", conv_storage_type, hidden, kernel, true);
    layer.conv_norm = load_fused_batch_norm(store, source, prefix + ".conv.norm", hidden, conv_storage_type);
    layer.conv_pointwise_conv2 = load_linear_as_shape(
        store,
        source,
        prefix + ".conv.pointwise_conv2",
        conv_storage_type,
        {hidden, hidden, 1},
        hidden,
        hidden);
    return layer;
}

SortformerTransformerLayerWeights load_transformer_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    int64_t layer_index,
    const SortformerModelConfig & config,
    assets::TensorStorageType storage_type) {
    SortformerTransformerLayerWeights layer;
    const auto & encoder = config.tf_encoder;
    const int64_t hidden = encoder.hidden_size;
    const int64_t intermediate = encoder.intermediate_size;
    const std::string prefix = "tf_encoder.layers." + std::to_string(layer_index);

    layer.self_attn_layer_norm = modules::binding::norm_from_source(store, source, prefix + ".self_attn_layer_norm", hidden);
    layer.self_attn_q_proj = modules::binding::linear_from_source(store, source, prefix + ".self_attn.q_proj", storage_type, hidden, hidden, true);
    layer.self_attn_k_proj.weight =
        store.load_tensor(source, prefix + ".self_attn.k_proj.weight", storage_type, {hidden, hidden});
    if (source.has_tensor(prefix + ".self_attn.k_proj.bias")) {
        layer.self_attn_k_proj.bias = store.load_f32_tensor(source, prefix + ".self_attn.k_proj.bias", {hidden});
    }
    layer.self_attn_v_proj = modules::binding::linear_from_source(store, source, prefix + ".self_attn.v_proj", storage_type, hidden, hidden, true);
    layer.self_attn_out_proj = modules::binding::linear_from_source(store, source, prefix + ".self_attn.out_proj", storage_type, hidden, hidden, true);
    layer.final_layer_norm = modules::binding::norm_from_source(store, source, prefix + ".final_layer_norm", hidden);
    layer.fc1 = modules::binding::linear_from_source(store, source, prefix + ".fc1", storage_type, intermediate, hidden, true);
    layer.fc2 = modules::binding::linear_from_source(store, source, prefix + ".fc2", storage_type, hidden, intermediate, true);
    return layer;
}

}  // namespace

std::shared_ptr<const SortformerAssets> load_sortformer_assets(const std::filesystem::path & model_root) {
    auto assets = std::make_shared<SortformerAssets>();
    assets->resources = engine::model_spec::load_resource_bundle(
        model_root,
        engine::model_spec::default_spec_path("sortformer_diar"));
    assets->model_config = parse_sortformer_model_config(assets->resources);
    assets->feature_config = parse_sortformer_feature_config(assets->resources);
    assets->model_weights = assets->resources.open_tensor_source("weights");
    return assets;
}

SortformerModelConfig parse_sortformer_model_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    SortformerModelConfig config;
    config.model_type = root.require("model_type").as_string();
    config.variant = root.find("architectures") != nullptr && !root.require("architectures").as_array().empty()
        ? root.require("architectures").as_array().front().as_string()
        : "SortformerOffline";
    config.num_speakers = root.require("num_speakers").as_i64();
    config.pil_weight = root.require("pil_weight").as_f32();
    config.ats_weight = root.require("ats_weight").as_f32();

    const auto & fc = root.require("fc_encoder_config");
    config.fc_encoder.hidden_size = fc.require("hidden_size").as_i64();
    config.fc_encoder.intermediate_size = fc.require("intermediate_size").as_i64();
    config.fc_encoder.num_attention_heads = fc.require("num_attention_heads").as_i64();
    config.fc_encoder.num_hidden_layers = fc.require("num_hidden_layers").as_i64();
    config.fc_encoder.num_key_value_heads = fc.require("num_key_value_heads").as_i64();
    config.fc_encoder.num_mel_bins = fc.require("num_mel_bins").as_i64();
    config.fc_encoder.max_position_embeddings = fc.require("max_position_embeddings").as_i64();
    config.fc_encoder.conv_kernel_size = fc.require("conv_kernel_size").as_i64();
    config.fc_encoder.subsampling_factor = fc.require("subsampling_factor").as_i64();
    config.fc_encoder.subsampling_conv_channels = fc.require("subsampling_conv_channels").as_i64();
    config.fc_encoder.subsampling_conv_kernel_size = fc.require("subsampling_conv_kernel_size").as_i64();
    config.fc_encoder.subsampling_conv_stride = fc.require("subsampling_conv_stride").as_i64();
    config.fc_encoder.attention_bias = fc.require("attention_bias").as_bool();
    config.fc_encoder.scale_input = fc.require("scale_input").as_bool();
    config.fc_encoder.hidden_act = fc.require("hidden_act").as_string();

    const auto & tf = root.require("tf_encoder_config");
    config.tf_encoder.hidden_size = tf.require("d_model").as_i64();
    config.tf_encoder.intermediate_size = tf.require("encoder_ffn_dim").as_i64();
    config.tf_encoder.num_attention_heads = tf.require("encoder_attention_heads").as_i64();
    config.tf_encoder.num_hidden_layers = tf.require("encoder_layers").as_i64();
    config.tf_encoder.max_source_positions = tf.require("max_source_positions").as_i64();
    config.tf_encoder.layer_norm_eps = tf.require("layer_norm_eps").as_f32();
    config.tf_encoder.activation_function = tf.require("activation_function").as_string();

    const auto & modules = root.require("modules_config");
    config.modules.num_speakers = modules.require("num_speakers").as_i64();
    config.modules.fc_d_model = modules.require("fc_d_model").as_i64();
    config.modules.tf_d_model = modules.require("tf_d_model").as_i64();
    config.modules.subsampling_factor = modules.require("subsampling_factor").as_i64();
    config.modules.dropout_rate = modules.require("dropout_rate").as_f32();

    return config;
}

SortformerFeatureExtractorConfig parse_sortformer_feature_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("processor");
    const auto & feature = root.require("feature_extractor");
    SortformerFeatureExtractorConfig config;
    config.sample_rate = feature.require("sampling_rate").as_i64();
    config.n_fft = feature.require("n_fft").as_i64();
    config.win_length = feature.require("win_length").as_i64();
    config.hop_length = feature.require("hop_length").as_i64();
    config.num_mel_bins = feature.require("feature_size").as_i64();
    config.preemphasis = feature.require("preemphasis").as_f32();
    config.return_attention_mask = feature.require("return_attention_mask").as_bool();
    return config;
}

std::shared_ptr<SortformerDiarWeights> load_sortformer_diar_weights(
    const SortformerAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes) {
    if (assets.model_weights == nullptr) {
        throw std::runtime_error("Sortformer tensor source must not be null");
    }
    const auto & tensor_source = *assets.model_weights;
    const auto & config = assets.model_config;
    auto weights = std::make_shared<SortformerDiarWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "sortformer diar weights",
        weight_context_bytes);

    const auto & fc = config.fc_encoder;
    weights->subsampling.conv0 = modules::binding::conv2d_from_source(
        *weights->store,
        tensor_source,
        "fc_encoder.subsampling.layers.0",
        conv_storage_type,
        fc.subsampling_conv_channels,
        1,
        fc.subsampling_conv_kernel_size,
        fc.subsampling_conv_kernel_size,
        true);
    weights->subsampling.depthwise1_weight = weights->store->load_tensor(
        tensor_source,
        "fc_encoder.subsampling.layers.2.weight",
        conv_storage_type,
        {fc.subsampling_conv_channels, 1, fc.subsampling_conv_kernel_size, fc.subsampling_conv_kernel_size});
    weights->subsampling.depthwise1_bias = weights->store->load_f32_tensor(
        tensor_source,
        "fc_encoder.subsampling.layers.2.bias", {fc.subsampling_conv_channels});
    weights->subsampling.pointwise1 = modules::binding::conv2d_from_source(
        *weights->store,
        tensor_source,
        "fc_encoder.subsampling.layers.3",
        conv_storage_type,
        fc.subsampling_conv_channels,
        fc.subsampling_conv_channels,
        1,
        1,
        true);
    weights->subsampling.depthwise2_weight = weights->store->load_tensor(
        tensor_source,
        "fc_encoder.subsampling.layers.5.weight",
        conv_storage_type,
        {fc.subsampling_conv_channels, 1, fc.subsampling_conv_kernel_size, fc.subsampling_conv_kernel_size});
    weights->subsampling.depthwise2_bias = weights->store->load_f32_tensor(
        tensor_source,
        "fc_encoder.subsampling.layers.5.bias", {fc.subsampling_conv_channels});
    weights->subsampling.pointwise2 = modules::binding::conv2d_from_source(
        *weights->store,
        tensor_source,
        "fc_encoder.subsampling.layers.6",
        conv_storage_type,
        fc.subsampling_conv_channels,
        fc.subsampling_conv_channels,
        1,
        1,
        true);
    const int64_t reduced_mels = fc.num_mel_bins / fc.subsampling_factor;
    weights->subsampling.linear = modules::binding::linear_from_source(
        *weights->store,
        tensor_source,
        "fc_encoder.subsampling.linear",
        matmul_storage_type,
        fc.hidden_size,
        fc.subsampling_conv_channels * reduced_mels,
        true);

    weights->conformer_layers.resize(static_cast<size_t>(fc.num_hidden_layers));
    for (int64_t i = 0; i < fc.num_hidden_layers; ++i) {
        weights->conformer_layers[static_cast<size_t>(i)] =
            load_conformer_layer(*weights->store, tensor_source, i, config, matmul_storage_type, conv_storage_type);
    }

    const auto & tf = config.tf_encoder;
    weights->transformer_layers.resize(static_cast<size_t>(tf.num_hidden_layers));
    for (int64_t i = 0; i < tf.num_hidden_layers; ++i) {
        weights->transformer_layers[static_cast<size_t>(i)] =
            load_transformer_layer(*weights->store, tensor_source, i, config, matmul_storage_type);
    }

    weights->head.encoder_proj = modules::binding::linear_from_source(
        *weights->store,
        tensor_source,
        "sortformer_modules.encoder_proj",
        matmul_storage_type,
        config.modules.tf_d_model,
        config.modules.fc_d_model,
        true);
    weights->head.first_hidden_to_hidden = modules::binding::linear_from_source(
        *weights->store,
        tensor_source,
        "sortformer_modules.first_hidden_to_hidden",
        matmul_storage_type,
        config.modules.tf_d_model,
        config.modules.tf_d_model,
        true);
    weights->head.single_hidden_to_spks = modules::binding::linear_from_source(
        *weights->store,
        tensor_source,
        "sortformer_modules.single_hidden_to_spks",
        matmul_storage_type,
        config.modules.num_speakers,
        config.modules.tf_d_model,
        true);
    weights->store->upload();
    return weights;
}

}  // namespace engine::models::sortformer_diar
