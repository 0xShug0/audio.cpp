#include "engine/community_models/parakeet_tdt/weights.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/weight_binding.h"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::community_models::parakeet_tdt {
namespace {

using Clock = std::chrono::steady_clock;
namespace binding = engine::modules::binding;

engine::modules::LinearWeights load_linear(
    engine::core::BackendWeightStore & store, const engine::assets::TensorSource & source,
    const std::string & prefix, engine::assets::TensorStorageType st,
    int64_t out_f, int64_t in_f, bool use_bias) {
    engine::modules::LinearWeights w;
    w.weight = store.load_tensor(source, prefix + ".weight", st, {out_f, in_f});
    if (use_bias) w.bias = store.load_f32_tensor(source, prefix + ".bias", {out_f});
    return w;
}

engine::modules::Conv2dWeights load_conv2d(
    engine::core::BackendWeightStore & store, const engine::assets::TensorSource & source,
    const std::string & prefix, engine::assets::TensorStorageType st,
    int64_t oc, int64_t ic, int64_t kh, int64_t kw, bool use_bias) {
    engine::modules::Conv2dWeights w;
    w.weight = store.load_tensor(source, prefix + ".weight", st, {oc, ic, kh, kw});
    if (use_bias) w.bias = store.load_f32_tensor(source, prefix + ".bias", {oc});
    return w;
}

ParakeetSubsamplingWeights load_subsampling(
    engine::core::BackendWeightStore & store, const engine::assets::TensorSource & source,
    const ParakeetConfig & config, engine::assets::TensorStorageType matmul_st,
    engine::assets::TensorStorageType conv_st) {
    ParakeetSubsamplingWeights w;
    const auto & enc = config.encoder;
    const int64_t C = enc.subsampling_channels;
    const std::string sub = "encoder.subsampling";
    w.conv_in = load_conv2d(store, source, sub + ".layers.0", conv_st, C, 1, enc.subsampling_kernel, enc.subsampling_kernel, true);
    w.layers.resize(2);
    w.layers[0].depthwise_weight = store.load_tensor(source, sub + ".layers.2.weight", conv_st, {C, 1, enc.subsampling_kernel, enc.subsampling_kernel});
    w.layers[0].depthwise_bias = store.load_f32_tensor(source, sub + ".layers.2.bias", {C});
    w.layers[0].pointwise = load_conv2d(store, source, sub + ".layers.3", conv_st, C, C, 1, 1, true);
    w.layers[1].depthwise_weight = store.load_tensor(source, sub + ".layers.5.weight", conv_st, {C, 1, enc.subsampling_kernel, enc.subsampling_kernel});
    w.layers[1].depthwise_bias = store.load_f32_tensor(source, sub + ".layers.5.bias", {C});
    w.layers[1].pointwise = load_conv2d(store, source, sub + ".layers.6", conv_st, C, C, 1, 1, true);
    const int64_t ff = config.frontend.feature_size / enc.subsampling_factor;
    w.linear = load_linear(store, source, sub + ".linear", matmul_st, enc.hidden_size, C * ff, true);
    return w;
}

std::pair<std::vector<float>, std::vector<float>> fold_bn(
    const engine::assets::TensorSource & source, const std::string & dw_pfx, const std::string & bn_pfx,
    int64_t d_model, int64_t K) {
    auto bn_w = source.require_f32(bn_pfx + ".weight", {d_model});
    auto bn_b = source.require_f32(bn_pfx + ".bias", {d_model});
    auto bn_m = source.require_f32(bn_pfx + ".running_mean", {d_model});
    auto bn_v = source.require_f32(bn_pfx + ".running_var", {d_model});
    auto dw_w = source.require_f32(dw_pfx + ".weight", {d_model, 1, K});

    // Layout: dw_w is row-major [channels][1][kernel], so element (c,0,ki) is at offset c*K+ki
    // ggml stores [ne0=K, ne1=1, ne2=d_model] → consecutive K elements per channel
    std::vector<float> w_f(static_cast<size_t>(K * d_model));
    for (int64_t c = 0; c < d_model; ++c)
        for (int64_t ki = 0; ki < K; ++ki)
            w_f[static_cast<size_t>(ki + c * K)] = dw_w[static_cast<size_t>(c * K + ki)];

    std::vector<float> dw_b(static_cast<size_t>(d_model), 0.f);
    const float eps = 1e-5f;
    for (int64_t c = 0; c < d_model; ++c) {
        float s = bn_w[static_cast<size_t>(c)] / std::sqrt(std::max(bn_v[static_cast<size_t>(c)], eps) + eps);
        for (int64_t ki = 0; ki < K; ++ki)
            w_f[static_cast<size_t>(ki + c * K)] *= s;
        dw_b[static_cast<size_t>(c)] = -bn_m[static_cast<size_t>(c)] * s + bn_b[static_cast<size_t>(c)];
    }
    return {std::move(w_f), std::move(dw_b)};
}

ParakeetEncoderLayerWeights load_encoder_layer(
    engine::core::BackendWeightStore & store, const engine::assets::TensorSource & source,
    const ParakeetConfig & config, int64_t idx, engine::assets::TensorStorageType matmul_st,
    engine::assets::TensorStorageType conv_st) {
    const std::string p = "encoder.layers." + std::to_string(idx);
    const auto & enc = config.encoder;
    const int64_t h = enc.hidden_size;
    const int64_t hd = h / enc.heads;
    ParakeetEncoderLayerWeights layer;
    layer.norm_ff1 = binding::norm_from_source(store, source, p + ".norm_feed_forward1", h);
    layer.ff1_linear1 = {store.load_tensor(source, p + ".feed_forward1.linear1.weight", matmul_st, {enc.intermediate_size, h}), std::nullopt};
    layer.ff1_linear2 = {store.load_tensor(source, p + ".feed_forward1.linear2.weight", matmul_st, {h, enc.intermediate_size}), std::nullopt};
    layer.norm_attn = binding::norm_from_source(store, source, p + ".norm_self_att", h);
    layer.self_attn.q_weight = store.load_tensor(source, p + ".self_attn.q_proj.weight", matmul_st, {h, h});
    layer.self_attn.k_weight = store.load_tensor(source, p + ".self_attn.k_proj.weight", matmul_st, {h, h});
    layer.self_attn.v_weight = store.load_tensor(source, p + ".self_attn.v_proj.weight", matmul_st, {h, h});
    layer.self_attn.out_weight = store.load_tensor(source, p + ".self_attn.o_proj.weight", matmul_st, {h, h});
    layer.pos_weight = store.load_tensor(source, p + ".self_attn.relative_k_proj.weight", matmul_st, {h, h});
    layer.pos_bias_u = store.load_f32_tensor(source, p + ".self_attn.bias_u", {enc.heads, hd});
    layer.pos_bias_v = store.load_f32_tensor(source, p + ".self_attn.bias_v", {enc.heads, hd});
    layer.norm_conv = binding::norm_from_source(store, source, p + ".norm_conv", h);
    layer.conv_pw1 = {store.load_tensor_as_shape(source, p + ".conv.pointwise_conv1.weight", conv_st, {2*h, h, 1}, engine::core::TensorShape::from_dims({2*h, h})), std::nullopt};
    layer.conv_pw2 = {store.load_tensor_as_shape(source, p + ".conv.pointwise_conv2.weight", conv_st, {h, h, 1}, engine::core::TensorShape::from_dims({h, h})), std::nullopt};
    auto [fd_w, fd_b] = fold_bn(source, p + ".conv.depthwise_conv", p + ".conv.norm", h, enc.conv_kernel);
    layer.conv_dw_weight = store.make_from_f32(engine::core::TensorShape::from_dims({h, 1, enc.conv_kernel}), engine::assets::TensorStorageType::F32, std::move(fd_w));
    layer.conv_dw_bias = store.make_f32(engine::core::TensorShape::from_dims({h}), std::move(fd_b));
    layer.norm_ff2 = binding::norm_from_source(store, source, p + ".norm_feed_forward2", h);
    layer.ff2_linear1 = {store.load_tensor(source, p + ".feed_forward2.linear1.weight", matmul_st, {enc.intermediate_size, h}), std::nullopt};
    layer.ff2_linear2 = {store.load_tensor(source, p + ".feed_forward2.linear2.weight", matmul_st, {h, enc.intermediate_size}), std::nullopt};
    layer.norm_out = binding::norm_from_source(store, source, p + ".norm_out", h);
    return layer;
}

}  // namespace

ParakeetEncoderWeights load_encoder_weights(
    engine::core::BackendWeightStore & store, const engine::assets::TensorSource & source,
    const ParakeetConfig & config, engine::assets::TensorStorageType matmul_st,
    engine::assets::TensorStorageType conv_st) {
    ParakeetEncoderWeights w;
    w.subsampling = load_subsampling(store, source, config, matmul_st, conv_st);
    w.layers.reserve(static_cast<size_t>(config.encoder.layers));
    for (int64_t i = 0; i < config.encoder.layers; ++i)
        w.layers.push_back(load_encoder_layer(store, source, config, i, matmul_st, conv_st));
    return w;
}

ParakeetDecoderWeights load_decoder_weights(
    engine::core::BackendWeightStore & store, const engine::assets::TensorSource & source,
    const ParakeetConfig & config, engine::assets::TensorStorageType st) {
    ParakeetDecoderWeights w;
    const int64_t H = config.decoder_hidden_size;
    w.embedding = store.load_tensor(source, "decoder.embedding.weight", st, {config.vocab_size, H});
    w.lstm_layers.reserve(static_cast<size_t>(config.decoder_layers));
    for (int64_t l = 0; l < config.decoder_layers; ++l) {
        const std::string p = "decoder.lstm";
        w.lstm_layers.push_back({
            store.load_tensor(source, p + ".weight_ih_l" + std::to_string(l), st, {4*H, H}),
            store.load_tensor(source, p + ".weight_hh_l" + std::to_string(l), st, {4*H, H}),
            store.load_f32_tensor(source, p + ".bias_ih_l" + std::to_string(l), {4*H}),
            store.load_f32_tensor(source, p + ".bias_hh_l" + std::to_string(l), {4*H}),
        });
    }
    w.decoder_projector = load_linear(store, source, "decoder.decoder_projector", st, H, H, true);
    w.joint_enc = load_linear(store, source, "encoder_projector", st, H, config.encoder.hidden_size, true);
    const int64_t jout = config.vocab_size + static_cast<int64_t>(config.durations.size());
    w.joint_head = load_linear(store, source, "joint.head", st, jout, H, true);
    return w;
}

std::shared_ptr<const ParakeetWeights> load_parakeet_weights(
    const ParakeetTDTAssets & assets, ggml_backend_t backend, engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_st, engine::assets::TensorStorageType conv_st, size_t ctx_bytes) {
    if (!assets.source) throw std::runtime_error("Parakeet TDT requires a tensor source");
    const auto t0 = Clock::now();
    auto w = std::make_shared<ParakeetWeights>();
    w->store = std::make_shared<engine::core::BackendWeightStore>(backend, backend_type, "parakeet_tdt.weights", ctx_bytes);
    w->encoder = load_encoder_weights(*w->store, *assets.source, assets.config, matmul_st, conv_st);
    w->decoder = load_decoder_weights(*w->store, *assets.source, assets.config, matmul_st);
    w->store->upload();
    assets.source->release_storage();
    debug::timing_log_scalar("parakeet_tdt.weights.upload_ms", engine::debug::elapsed_ms(t0, Clock::now()));
    return w;
}

}  // namespace engine::community_models::parakeet_tdt
