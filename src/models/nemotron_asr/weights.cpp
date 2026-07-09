#include "engine/models/nemotron_asr/weights.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/weight_binding.h"

#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::nemotron_asr {
namespace {

using Clock = std::chrono::steady_clock;
namespace binding = engine::modules::binding;

engine::modules::LinearWeights load_linear(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type,
    int64_t out_features,
    int64_t in_features,
    bool use_bias) {
    engine::modules::LinearWeights weights;
    weights.weight = store.load_tensor(source, prefix + ".weight", storage_type, {out_features, in_features});
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    }
    return weights;
}

engine::modules::Conv2dWeights load_conv2d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_h,
    int64_t kernel_w,
    bool use_bias) {
    engine::modules::Conv2dWeights weights;
    weights.weight = store.load_tensor(
        source,
        prefix + ".weight",
        storage_type,
        {out_channels, in_channels, kernel_h, kernel_w});
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    }
    return weights;
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
    weights.attention.q_weight = store.load_tensor(source, prefix + ".q_proj.weight", storage_type, {hidden, hidden});
    weights.attention.k_weight = store.load_tensor(source, prefix + ".k_proj.weight", storage_type, {hidden, hidden});
    weights.attention.v_weight = store.load_tensor(source, prefix + ".v_proj.weight", storage_type, {hidden, hidden});
    weights.attention.out_weight = store.load_tensor(source, prefix + ".o_proj.weight", storage_type, {hidden, hidden});
    weights.pos_weight = store.load_tensor(source, prefix + ".relative_k_proj.weight", storage_type, {hidden, hidden});
    weights.pos_bias_u = store.load_f32_tensor(source, prefix + ".bias_u", {heads, head_dim});
    weights.pos_bias_v = store.load_f32_tensor(source, prefix + ".bias_v", {heads, head_dim});
    return weights;
}

NemotronSubsamplingWeights load_subsampling(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const NemotronConfig & config,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    NemotronSubsamplingWeights weights;
    const auto & enc = config.encoder;
    const int64_t channels = enc.subsampling_channels;
    weights.conv_in = load_conv2d(
        store,
        source,
        "encoder.subsampling.conv_in",
        conv_storage_type,
        channels,
        1,
        enc.subsampling_kernel,
        enc.subsampling_kernel,
        true);
    weights.layers.resize(2);
    for (size_t i = 0; i < weights.layers.size(); ++i) {
        const std::string prefix = "encoder.subsampling.layers." + std::to_string(i);
        weights.layers[i].depthwise_weight = store.load_tensor(
            source,
            prefix + ".depthwise_conv.weight",
            conv_storage_type,
            {channels, 1, enc.subsampling_kernel, enc.subsampling_kernel});
        weights.layers[i].depthwise_bias =
            store.load_f32_tensor(source, prefix + ".depthwise_conv.bias", {channels});
        weights.layers[i].pointwise = load_conv2d(
            store,
            source,
            prefix + ".pointwise_conv",
            conv_storage_type,
            channels,
            channels,
            1,
            1,
            true);
    }
    const int64_t freq_after = config.frontend.feature_size / enc.subsampling_factor + 1;
    weights.linear = load_linear(
        store,
        source,
        "encoder.subsampling.linear",
        matmul_storage_type,
        enc.hidden_size,
        channels * freq_after,
        true);
    return weights;
}

NemotronEncoderLayerWeights load_encoder_layer(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const NemotronConfig & config,
    int64_t layer_index,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    const std::string prefix = "encoder.layers." + std::to_string(layer_index);
    const auto & enc = config.encoder;
    const int64_t hidden = enc.hidden_size;
    const int64_t head_dim = hidden / enc.heads;
    NemotronEncoderLayerWeights layer;
    layer.norm_feed_forward1 = binding::norm_from_source(store, source, prefix + ".norm_feed_forward1", hidden);
    layer.norm_self_att = binding::norm_from_source(store, source, prefix + ".norm_self_att", hidden);
    layer.norm_conv = binding::norm_from_source(store, source, prefix + ".norm_conv", hidden);
    layer.conv_norm = binding::norm_from_source(store, source, prefix + ".conv.norm", hidden);
    layer.norm_feed_forward2 = binding::norm_from_source(store, source, prefix + ".norm_feed_forward2", hidden);
    layer.norm_out = binding::norm_from_source(store, source, prefix + ".norm_out", hidden);
    layer.ff1_linear1 = load_linear(store, source, prefix + ".feed_forward1.linear1", matmul_storage_type, enc.intermediate_size, hidden, false);
    layer.ff1_linear2 = load_linear(store, source, prefix + ".feed_forward1.linear2", matmul_storage_type, hidden, enc.intermediate_size, false);
    layer.ff2_linear1 = load_linear(store, source, prefix + ".feed_forward2.linear1", matmul_storage_type, enc.intermediate_size, hidden, false);
    layer.ff2_linear2 = load_linear(store, source, prefix + ".feed_forward2.linear2", matmul_storage_type, hidden, enc.intermediate_size, false);
    layer.self_attn = load_relative_attention(store, source, prefix + ".self_attn", matmul_storage_type, hidden, enc.heads, head_dim);
    layer.conv_pointwise1 = {
        store.load_tensor_as_shape(
            source,
            prefix + ".conv.pointwise_conv1.weight",
            conv_storage_type,
            {2 * hidden, hidden, 1},
            engine::core::TensorShape::from_dims({2 * hidden, hidden})),
        std::nullopt,
    };
    layer.conv_depthwise = {
        store.load_tensor(source, prefix + ".conv.depthwise_conv.weight", conv_storage_type, {hidden, 1, enc.conv_kernel}),
        std::nullopt,
    };
    layer.conv_pointwise2 = {
        store.load_tensor_as_shape(
            source,
            prefix + ".conv.pointwise_conv2.weight",
            conv_storage_type,
            {hidden, hidden, 1},
            engine::core::TensorShape::from_dims({hidden, hidden})),
        std::nullopt,
    };
    return layer;
}

NemotronEncoderWeights load_encoder_weights(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const NemotronConfig & config,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    NemotronEncoderWeights weights;
    weights.subsampling = load_subsampling(store, source, config, matmul_storage_type, conv_storage_type);
    weights.layers.reserve(static_cast<size_t>(config.encoder.layers));
    for (int64_t i = 0; i < config.encoder.layers; ++i) {
        weights.layers.push_back(load_encoder_layer(store, source, config, i, matmul_storage_type, conv_storage_type));
    }
    weights.prompt_linear1 = load_linear(
        store,
        source,
        "prompt_projector.linear_1",
        matmul_storage_type,
        config.prompt_intermediate_size,
        config.encoder.hidden_size + config.num_prompts,
        true);
    weights.prompt_linear2 = load_linear(
        store,
        source,
        "prompt_projector.linear_2",
        matmul_storage_type,
        config.encoder.hidden_size,
        config.prompt_intermediate_size,
        true);
    weights.encoder_projector = load_linear(
        store,
        source,
        "encoder_projector",
        matmul_storage_type,
        config.decoder_hidden_size,
        config.encoder.hidden_size,
        true);
    return weights;
}

NemotronDecoderWeights load_decoder_weights(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const NemotronConfig & config,
    engine::assets::TensorStorageType storage_type) {
    NemotronDecoderWeights weights;
    weights.embedding = store.load_tensor(
        source,
        "decoder.embedding.weight",
        storage_type,
        {config.vocab_size, config.decoder_hidden_size});
    weights.lstm_layers.reserve(static_cast<size_t>(config.decoder_layers));
    for (int64_t layer = 0; layer < config.decoder_layers; ++layer) {
        const std::string prefix = "decoder.lstm.";
        weights.lstm_layers.push_back({
            store.load_tensor(source, prefix + "weight_ih_l" + std::to_string(layer), storage_type, {4 * config.decoder_hidden_size, config.decoder_hidden_size}),
            store.load_tensor(source, prefix + "weight_hh_l" + std::to_string(layer), storage_type, {4 * config.decoder_hidden_size, config.decoder_hidden_size}),
            store.load_f32_tensor(source, prefix + "bias_ih_l" + std::to_string(layer), {4 * config.decoder_hidden_size}),
            store.load_f32_tensor(source, prefix + "bias_hh_l" + std::to_string(layer), {4 * config.decoder_hidden_size}),
        });
    }
    weights.decoder_projector = load_linear(
        store,
        source,
        "decoder.decoder_projector",
        storage_type,
        config.decoder_hidden_size,
        config.decoder_hidden_size,
        true);
    weights.joint_head = load_linear(
        store,
        source,
        "joint.head",
        storage_type,
        config.vocab_size,
        config.decoder_hidden_size,
        true);
    return weights;
}

}  // namespace

std::shared_ptr<const NemotronWeights> load_nemotron_asr_weights(
    const NemotronAssets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes) {
    if (assets.source == nullptr) {
        throw std::runtime_error("Nemotron ASR requires a tensor source");
    }
    const auto load_start = Clock::now();
    auto weights = std::make_shared<NemotronWeights>();
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        backend,
        backend_type,
        "nemotron_asr.weights",
        weight_context_bytes);
    weights->encoder = load_encoder_weights(
        *weights->store,
        *assets.source,
        assets.config,
        matmul_storage_type,
        conv_storage_type);
    weights->decoder = load_decoder_weights(
        *weights->store,
        *assets.source,
        assets.config,
        matmul_storage_type);
    weights->store->upload();
    assets.source->release_storage();
    debug::timing_log_scalar("nemotron_asr.weights.upload_ms", engine::debug::elapsed_ms(load_start, Clock::now()));
    return weights;
}

}  // namespace engine::models::nemotron_asr
