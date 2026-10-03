#include "engine/models/owsm/model.h"

#include "engine/framework/modules/packed_linear_weights.h"
#include "engine/framework/modules/weight_binding.h"

#include <cmath>
#include <stdexcept>

namespace engine::models::owsm {

std::unique_ptr<OWSMV4Weights> load_owsm_weights(
    const OWSMV4Assets & assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType type) {
    const auto & config = assets.config;
    auto out = std::make_unique<OWSMV4Weights>();
    out->store = std::make_unique<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "owsm.weights", 4 * 1024 * 1024);
    auto & store = *out->store;
    const auto & source = *assets.source;
    const auto d = config.hidden_size;
    const auto ff = config.intermediate_size;

    const auto linear = [&](const std::string & name, int64_t output, int64_t input) {
        return modules::binding::linear_from_source(store, source, name, type, output, input, true);
    };
    const auto norm = [&](const std::string & name, int64_t size) {
        return modules::binding::norm_from_source(store, source, name, size);
    };
    const auto feed_forward = [&](const std::string & name) {
        const auto first = linear(name + ".w_1", ff, d);
        const auto second = linear(name + ".w_2", d, ff);
        return modules::FeedForwardWeights{first.weight, first.bias, second.weight, second.bias};
    };
    const auto attention = [&](const std::string & name, bool packed_qkv, bool packed_kv) {
        modules::AttentionWeights weights;
        if (packed_qkv) {
            const auto packed = modules::PackedLinearWeightsBuilder({d, {
                {name + ".linear_q.weight", name + ".linear_q.bias", d},
                {name + ".linear_k.weight", name + ".linear_k.bias", d},
                {name + ".linear_v.weight", name + ".linear_v.bias", d}}, true}).build(store, source, type);
            weights.qkv_weight = packed.weight;
            weights.qkv_bias = packed.bias;
        } else {
            const auto query = linear(name + ".linear_q", d, d);
            weights.q_weight = query.weight;
            weights.q_bias = query.bias;
            if (packed_kv) {
                const auto packed = modules::PackedLinearWeightsBuilder({d, {
                    {name + ".linear_k.weight", name + ".linear_k.bias", d},
                    {name + ".linear_v.weight", name + ".linear_v.bias", d}}, true}).build(store, source, type);
                weights.qkv_weight = packed.weight;
                weights.qkv_bias = packed.bias;
            }
        }
        const auto output = linear(name + ".linear_out", d, d);
        weights.out_weight = output.weight;
        weights.out_bias = output.bias;
        return weights;
    };

    out->subsampling.conv0 = modules::binding::conv2d_from_source(
        store, source, "encoder.embed.conv.0", assets::TensorStorageType::F32, d, 1, 3, 3, true);
    out->subsampling.conv1 = modules::binding::conv2d_from_source(
        store, source, "encoder.embed.conv.2", assets::TensorStorageType::F32, d, d, 3, 3, true);
    out->subsampling.conv2 = modules::binding::conv2d_from_source(
        store, source, "encoder.embed.conv.4", assets::TensorStorageType::F32, d, d, 3, 3, true);
    out->subsampling.projection = linear("encoder.embed.out", d, d * 15);

    out->encoder.reserve(static_cast<size_t>(config.encoder_layers));
    for (int64_t index = 0; index < config.encoder_layers; ++index) {
        const auto prefix = "encoder.encoders." + std::to_string(index);
        OWSMV4EBranchformerLayerWeights layer;
        layer.macaron_ffn = feed_forward(prefix + ".feed_forward_macaron");
        layer.macaron_norm = norm(prefix + ".norm_ff_macaron", d);
        layer.attention_norm = norm(prefix + ".norm_mha", d);
        layer.attention = attention(prefix + ".attn", true, false);
        layer.cgmlp_norm = norm(prefix + ".norm_mlp", d);
        layer.cgmlp.input_projection = linear(prefix + ".cgmlp.channel_proj1.0", ff, d);
        layer.cgmlp.gate_norm = norm(prefix + ".cgmlp.csgu.norm", ff / 2);
        layer.cgmlp.gate_conv = modules::binding::depthwise_conv1d_from_source(
            store, source, prefix + ".cgmlp.csgu.conv", assets::TensorStorageType::F32, ff / 2, 31, true);
        layer.cgmlp.output_projection = linear(prefix + ".cgmlp.channel_proj2", d, ff / 2);
        layer.merge_conv = modules::binding::depthwise_conv1d_from_source(
            store, source, prefix + ".depthwise_conv_fusion", assets::TensorStorageType::F32, d * 2, 31, true);
        layer.merge_projection = linear(prefix + ".merge_proj", d, d * 2);
        layer.final_ffn = feed_forward(prefix + ".feed_forward");
        layer.final_ffn_norm = norm(prefix + ".norm_ff", d);
        layer.output_norm = norm(prefix + ".norm_final", d);
        out->encoder.push_back(std::move(layer));
    }
    out->encoder_norm = norm("encoder.after_norm", d);

    out->embedding = store.load_tensor(source, "decoder.embed.0.weight", type, {config.vocabulary_size, d});
    out->decoder.reserve(static_cast<size_t>(config.decoder_layers));
    for (int64_t index = 0; index < config.decoder_layers; ++index) {
        const auto prefix = "decoder.decoders." + std::to_string(index);
        modules::TransformerDecoderBlockWeights layer;
        layer.norm1 = norm(prefix + ".norm1", d);
        layer.self_attention = attention(prefix + ".self_attn", true, false);
        layer.norm2 = norm(prefix + ".norm2", d);
        layer.cross_attention = attention(prefix + ".src_attn", false, true);
        layer.norm3 = norm(prefix + ".norm3", d);
        layer.feed_forward = feed_forward(prefix + ".feed_forward");
        out->decoder.push_back(std::move(layer));
    }
    out->decoder_norm = norm("decoder.after_norm", d);
    out->output = linear("decoder.output_layer", config.vocabulary_size, d);
    out->half_scale = store.make_f32(core::TensorShape::from_dims({d}), std::vector<float>(static_cast<size_t>(d), 0.5F));
    out->embedding_scale = store.make_f32(
        core::TensorShape::from_dims({d}), std::vector<float>(static_cast<size_t>(d), std::sqrt(static_cast<float>(d))));
    store.upload();
    return out;
}

}  // namespace engine::models::owsm
