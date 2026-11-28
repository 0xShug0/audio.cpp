#include "engine/models/gigaam_asr/model.h"

#include "engine/framework/modules/weight_binding.h"

#include <cmath>

namespace engine::models::gigaam_asr {

namespace {

float scalar(const assets::TensorSource & source, const std::string & name) {
    return source.require_f32(name, {1}).at(0);
}

void load_zapformer_weights(GigaAMWeights & out, const GigaAMAssets & assets, core::BackendWeightStore & store,
    const assets::TensorSource & source, assets::TensorStorageType type, size_t head) {
    namespace binding = modules::binding;
    const auto & z = assets.zapformer_encoder;
    auto & w = out.zapformer;
    const auto linear = [&](const std::string & p, bool bias = true) {
        return binding::linear_from_named_source(store, source, p + ".weight",
            bias ? std::optional<std::string>(p + ".bias") : std::nullopt, type);
    };
    const auto norm = [&](const std::string & p) {
        return binding::norm_from_named_source(store, source, p + ".weight", std::nullopt);
    };
    const auto conv2d = [&](const std::string & p, int64_t out_c, int64_t in_c, int64_t k) {
        return binding::conv2d_from_source(store, source, p, type, out_c, in_c, k, k, true);
    };
    const std::string fe = "encoder.encoder_embed";
    const auto c1 = z.frontend_channels[0], c2 = z.frontend_channels[1], c3 = z.frontend_channels[2];
    w.conv1 = conv2d(fe + ".conv1", c1, 1, 3);
    w.conv2 = conv2d(fe + ".conv2", c2, c1, 3);
    w.conv3 = conv2d(fe + ".conv3", c3, c2, 3);
    w.convnext_depthwise = conv2d(fe + ".convnext.depthwise_conv", c3, 1, 3);
    w.convnext_pointwise1 = conv2d(fe + ".convnext.pointwise_conv1", 3 * c3, c3, 1);
    w.convnext_pointwise2 = conv2d(fe + ".convnext.pointwise_conv2", c3, 3 * c3, 1);
    w.out = linear(fe + ".out");
    for (size_t s = 0; s < z.stacks.size(); ++s) {
        const auto & c = z.stacks[s];
        const auto p = "encoder.encoder.encoders." + std::to_string(s);
        ZapformerStackWeights stack;
        stack.proj = binding::linear_from_named_source(store, source, p + ".proj.weight", std::nullopt, type);
        stack.proj_t = binding::linear_from_named_source(store, source, p + ".proj.weight_t", std::nullopt, type);
        const int64_t input = z.embed_dim * c.downsampling;
        stack.proj_scale = 1.0f / (std::sqrt(static_cast<float>(input)) * 0.3f);
        stack.input_scale = scalar(source, p + ".input_scale");
        stack.residual_scales = source.require_f32(p + ".residual_scales", {c.layers + 1});
        for (int64_t i = 0; i < c.layers; ++i) {
            const auto l = p + ".layers." + std::to_string(i);
            ZapformerLayerWeights layer;
            layer.ff1_in = linear(l + ".feed_forward1.in_proj");
            layer.ff1_out = linear(l + ".feed_forward1.out_proj");
            layer.attention_norm = norm(l + ".self_attn.in_norm");
            layer.qk = linear(l + ".self_attn.qk_in_proj");
            layer.vg = linear(l + ".self_attn.vg_in_proj");
            layer.attention_out = linear(l + ".self_attn.out_proj");
            layer.output_norm = norm(l + ".norm");
            layer.has_conv = c.conv_kernel > 0;
            if (layer.has_conv) {
                layer.ff2_in = linear(l + ".feed_forward2.in_proj");
                layer.ff2_out = linear(l + ".feed_forward2.out_proj");
                layer.conv_in = linear(l + ".conv_module.in_proj");
                layer.conv_out = linear(l + ".conv_module.out_proj");
                layer.depthwise = binding::depthwise_conv1d_from_source(store, source,
                    l + ".conv_module.depthwise_conv", type, c.dim, c.conv_kernel, true);
                layer.mean_weights = store.load_f32_tensor(source, l + ".conv_module.weighted_mean.weights", {c.dim});
            }
            stack.layers.push_back(std::move(layer));
        }
        w.stacks.push_back(std::move(stack));
    }
    w.out_scale = scalar(source, "encoder.encoder.out_norm.scale");
    w.out_eps = scalar(source, "encoder.encoder.out_norm.eps");

    const auto h = "heads." + assets.head_names.at(head);
    out.head = head;
    out.predictor = binding::lstm_cell_from_source(store, source, h + ".decoder.lstm", 0,
        assets.predictor_hidden, assets.predictor_hidden, type);
    out.embedding = store.load_tensor(source, h + ".decoder.embed.weight", type,
        {assets.classes, assets.predictor_hidden});
    out.joint_encoder = linear(h + ".joint.enc");
    out.joint_predictor = linear(h + ".joint.pred");
    out.joint_output = linear(h + ".joint.joint_net.1");
}

}  // namespace

std::unique_ptr<GigaAMWeights> load_gigaam_weights(
    const GigaAMAssets & assets, core::ExecutionContext & execution, assets::TensorStorageType type, size_t head) {
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
    if (assets.zapformer) {
        load_zapformer_weights(*out, assets, store, source, type, head);
        store.upload();
        return out;
    }
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
