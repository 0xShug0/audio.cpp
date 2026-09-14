#include "engine/models/canary_asr/model.h"

#include "engine/framework/modules/packed_linear_weights.h"
#include "engine/framework/modules/weight_binding.h"

#include <cmath>

namespace engine::models::canary_asr {

std::unique_ptr<CanaryWeights> load_canary_weights(
    const CanaryAssets & assets, core::ExecutionContext & execution, assets::TensorStorageType type) {
    namespace binding = modules::binding;
    auto out = std::make_unique<CanaryWeights>();
    out->store = std::make_unique<core::BackendWeightStore>(execution.backend(), execution.backend_type(),
        "canary_asr.weights", 4 * 1024 * 1024);
    auto & store = *out->store;
    const auto & source = *assets.source;
    const auto norm = [&](const std::string & name) {
        return binding::norm_from_named_source(store, source, name + ".weight", name + ".bias");
    };
    const auto linear = [&](const std::string & name) {
        return binding::linear_from_named_source(store, source, name + ".weight", name + ".bias", type);
    };
    const auto attention = [&](const std::string & name, bool conformer) {
        modules::AttentionWeights w;
        auto q = linear(name + (conformer ? ".linear_q" : ".query_net"));
        auto o = linear(name + (conformer ? ".linear_out" : ".out_projection"));
        w.q_weight = q.weight; w.q_bias = q.bias;
        if (conformer) {
            const auto k = linear(name + ".linear_k");
            const auto v = linear(name + ".linear_v");
            w.k_weight = k.weight; w.k_bias = k.bias;
            w.v_weight = v.weight; w.v_bias = v.bias;
        } else {
            const auto kv = modules::PackedLinearWeightsBuilder({1024, {
                {name + ".key_net.weight", name + ".key_net.bias", 1024},
                {name + ".value_net.weight", name + ".value_net.bias", 1024}}, true}).build(store, source, type);
            w.qkv_weight = kv.weight;
            w.qkv_bias = kv.bias;
        }
        w.out_weight = o.weight; w.out_bias = o.bias;
        return w;
    };
    out->conv0 = binding::conv2d_from_source(store, source, "encoder.pre_encode.conv.0",
        assets::TensorStorageType::F32, 256, 1, 3, 3, true);
    out->pointwise1 = binding::conv2d_from_source(store, source, "encoder.pre_encode.conv.3",
        assets::TensorStorageType::F32, 256, 256, 1, 1, true);
    out->pointwise2 = binding::conv2d_from_source(store, source, "encoder.pre_encode.conv.6",
        assets::TensorStorageType::F32, 256, 256, 1, 1, true);
    out->depthwise1 = store.load_f32_tensor(source, "encoder.pre_encode.conv.2.weight", {256, 1, 3, 3});
    out->depthwise1_bias = store.load_f32_tensor(source, "encoder.pre_encode.conv.2.bias", {256});
    out->depthwise2 = store.load_f32_tensor(source, "encoder.pre_encode.conv.5.weight", {256, 1, 3, 3});
    out->depthwise2_bias = store.load_f32_tensor(source, "encoder.pre_encode.conv.5.bias", {256});
    out->subsampling_out = linear("encoder.pre_encode.out");
    out->encoder_out = linear("encoder_decoder_proj");
    for (int64_t i = 0; i < 17; ++i) {
        const auto p = "encoder.layers." + std::to_string(i);
        modules::RelativeConformerBlockWeights w;
        w.ffn1_norm = norm(p + ".norm_feed_forward1");
        w.ffn1_fc1 = linear(p + ".feed_forward1.linear1");
        w.ffn1_fc2 = linear(p + ".feed_forward1.linear2");
        w.norm1 = norm(p + ".norm_self_att");
        w.self_attention.attention = attention(p + ".self_attn", true);
        w.self_attention.pos_weight = store.load_tensor(source, p + ".self_attn.linear_pos.weight", type, {512, 512});
        w.self_attention.pos_bias_u = store.load_f32_tensor(source, p + ".self_attn.pos_bias_u", {8, 64});
        w.self_attention.pos_bias_v = store.load_f32_tensor(source, p + ".self_attn.pos_bias_v", {8, 64});
        w.conv.norm = norm(p + ".norm_conv");
        w.conv.pointwise_in = {store.load_tensor_as_shape(source, p + ".conv.pointwise_conv1.weight", type,
            {1024, 512, 1}, core::TensorShape::from_dims({1024, 512})),
            store.load_f32_tensor(source, p + ".conv.pointwise_conv1.bias", {1024})};
        w.conv.pointwise_out = {store.load_tensor_as_shape(source, p + ".conv.pointwise_conv2.weight", type,
            {512, 512, 1}, core::TensorShape::from_dims({512, 512})),
            store.load_f32_tensor(source, p + ".conv.pointwise_conv2.bias", {512})};
        w.conv.depthwise = binding::depthwise_conv1d_from_source(store, source,
            p + ".conv.depthwise_conv", assets::TensorStorageType::F32, 512, 9, true);
        const auto gamma = source.require_f32(p + ".conv.batch_norm.weight", {512});
        const auto beta = source.require_f32(p + ".conv.batch_norm.bias", {512});
        const auto mean = source.require_f32(p + ".conv.batch_norm.running_mean", {512});
        const auto variance = source.require_f32(p + ".conv.batch_norm.running_var", {512});
        std::vector<float> scale(512), bias(512);
        for (size_t c = 0; c < 512; ++c) {
            scale[c] = gamma[c] / std::sqrt(variance[c] + 1e-5f);
            bias[c] = beta[c] - mean[c] * scale[c];
        }
        w.conv.depthwise_norm.scale = store.make_from_f32(core::TensorShape::from_dims({512}), assets::TensorStorageType::F32, std::move(scale));
        w.conv.depthwise_norm.bias = store.make_from_f32(core::TensorShape::from_dims({512}), assets::TensorStorageType::F32, std::move(bias));
        w.norm2 = norm(p + ".norm_feed_forward2");
        w.ffn2_fc1 = linear(p + ".feed_forward2.linear1");
        w.ffn2_fc2 = linear(p + ".feed_forward2.linear2");
        w.final_norm = norm(p + ".norm_out");
        out->encoder.push_back(std::move(w));
    }
    out->embedding = store.load_tensor(source, "transf_decoder._embedding.token_embedding.weight", type, {5248, 1024});
    out->positions = store.load_f32_tensor(source, "transf_decoder._embedding.position_embedding.pos_enc", {1024, 1024});
    out->embedding_norm = norm("transf_decoder._embedding.layer_norm");
    out->decoder_norm = norm("transf_decoder._decoder.final_layer_norm");
    for (int64_t i = 0; i < 4; ++i) {
        const auto p = "transf_decoder._decoder.layers." + std::to_string(i);
        CanaryDecoderLayer w;
        w.self_norm = norm(p + ".layer_norm_1");
        w.cross_norm = norm(p + ".layer_norm_2");
        w.ff_norm = norm(p + ".layer_norm_3");
        const auto self = p + ".first_sub_layer";
        const auto packed = modules::PackedLinearWeightsBuilder({1024, {
            {self + ".query_net.weight", self + ".query_net.bias", 1024},
            {self + ".key_net.weight", self + ".key_net.bias", 1024},
            {self + ".value_net.weight", self + ".value_net.bias", 1024}}, true}).build(store, source, type);
        w.self_attention.qkv_weight = packed.weight;
        w.self_attention.qkv_bias = packed.bias;
        const auto self_out = linear(self + ".out_projection");
        w.self_attention.out_weight = self_out.weight;
        w.self_attention.out_bias = self_out.bias;
        w.cross_attention = attention(p + ".second_sub_layer", false);
        w.fc1 = linear(p + ".third_sub_layer.dense_in");
        w.fc2 = linear(p + ".third_sub_layer.dense_out");
        out->decoder.push_back(std::move(w));
    }
    out->head = linear("log_softmax.mlp.layer0");
    store.upload();
    return out;
}

}  // namespace engine::models::canary_asr
