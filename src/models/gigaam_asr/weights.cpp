#include "engine/models/gigaam_asr/model.h"

#include "engine/framework/modules/weight_binding.h"

namespace engine::models::gigaam_asr {

std::unique_ptr<GigaAMWeights> load_gigaam_weights(
    const GigaAMAssets & assets, core::ExecutionContext & execution, assets::TensorStorageType type) {
    namespace binding = modules::binding;
    auto out = std::make_unique<GigaAMWeights>();
    // Only tensor descriptors live here, not weight data.
    out->store = std::make_unique<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "gigaam_asr.weights", 512 * 1024);
    auto & store = *out->store;
    const auto & source = *assets.source;
    const auto norm = [&](const std::string & p) {
        return binding::norm_from_named_source(store, source, p + ".weight", p + ".bias");
    };
    const auto linear = [&](const std::string & p) {
        return binding::linear_from_named_source(store, source, p + ".weight", p + ".bias", type);
    };
    const auto pointwise = [&](const std::string & p, int64_t input, int64_t output) {
        return modules::LinearWeights{
            store.load_tensor_as_shape(source, p + ".weight", type, {output, input, 1},
                core::TensorShape::from_dims({output, input})),
            store.load_f32_tensor(source, p + ".bias", {output})};
    };
    const auto & config = assets.encoder;
    const auto hidden = config.hidden_size;
    out->encoder.subsampling1 = binding::conv1d_from_source(store, source,
        "encoder.pre_encode.conv.0", type, hidden, config.features, config.subsampling_kernel, true);
    out->encoder.subsampling2 = binding::conv1d_from_source(store, source,
        "encoder.pre_encode.conv.2", type, hidden, hidden, config.subsampling_kernel, true);
    for (int64_t i = 0; i < assets.layers; ++i) {
        const auto p = "encoder.layers." + std::to_string(i);
        ConformerLayerWeights w;
        w.ffn1_norm = norm(p + ".norm_feed_forward1");
        w.attention_norm = norm(p + ".norm_self_att");
        w.conv_norm = norm(p + ".norm_conv");
        w.ffn2_norm = norm(p + ".norm_feed_forward2");
        w.output_norm = norm(p + ".norm_out");
        w.ffn1_in = linear(p + ".feed_forward1.linear1");
        w.ffn1_out = linear(p + ".feed_forward1.linear2");
        w.ffn2_in = linear(p + ".feed_forward2.linear1");
        w.ffn2_out = linear(p + ".feed_forward2.linear2");
        w.query = linear(p + ".self_attn.linear_q");
        w.key = linear(p + ".self_attn.linear_k");
        w.value = linear(p + ".self_attn.linear_v");
        w.attention_out = linear(p + ".self_attn.linear_out");
        w.pointwise_in = pointwise(p + ".conv.pointwise_conv1", hidden, 2 * hidden);
        w.pointwise_out = pointwise(p + ".conv.pointwise_conv2", hidden, hidden);
        w.depthwise = binding::depthwise_conv1d_from_source(store, source,
            p + ".conv.depthwise_conv", type, hidden, config.conv_kernel, true);
        w.depthwise_norm = norm(p + ".conv.batch_norm");
        out->encoder.layers.push_back(std::move(w));
    }
    if (assets.rnnt) {
        out->predictor = binding::lstm_cell_from_source(store, source, "head.decoder.lstm", 0,
            assets.predictor_hidden, assets.predictor_hidden, type);
        out->embedding = store.load_tensor(source, "head.decoder.embed.weight", type,
            {assets.classes, assets.predictor_hidden});
        out->joint_encoder = linear("head.joint.enc");
        out->joint_predictor = linear("head.joint.pred");
        out->joint_output = linear("head.joint.joint_net.1");
    } else {
        out->ctc = pointwise("head.decoder_layers.0", hidden, assets.classes);
    }
    store.upload();
    return out;
}

}  // namespace engine::models::gigaam_asr
