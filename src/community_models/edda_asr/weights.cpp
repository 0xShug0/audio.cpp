#include "engine/community_models/edda_asr/model.h"

#include "engine/framework/modules/packed_linear_weights.h"
#include "engine/framework/modules/weight_binding.h"

#include "ggml.h"

#include <stdexcept>
#include <tuple>

namespace engine::community_models::edda_asr {
namespace {

namespace binding = modules::binding;
using StorageType = assets::TensorStorageType;

struct Loader {
    core::BackendWeightStore & store;
    const assets::TensorSource & source;
    StorageType type;
    int64_t d;

    modules::NormWeights norm(const std::string & name, bool bias = true) const {
        return binding::norm_from_named_source(store, source, name + ".weight",
            bias ? std::optional<std::string>(name + ".bias") : std::nullopt);
    }
    modules::LinearWeights linear(const std::string & name, bool bias = true) const {
        return binding::linear_from_named_source(store, source, name + ".weight",
            bias ? std::optional<std::string>(name + ".bias") : std::nullopt, type);
    }
    core::TensorValue f32(const std::string & name, std::initializer_list<int64_t> shape) const {
        return store.load_f32_tensor(source, name, shape);
    }
    // Packed [K; V] cross-attention projections, the layout CrossAttentionModule caches.
    modules::AttentionWeights cross_attention(const std::string & prefix) const {
        modules::AttentionWeights w;
        const auto q = linear(prefix + ".q_proj");
        w.q_weight = q.weight;
        w.q_bias = q.bias;
        std::vector<modules::PackedLinearSource> kv{
            {prefix + ".k_proj.weight", std::nullopt, d}, {prefix + ".v_proj.weight", std::nullopt, d}};
        w.qkv_weight = modules::PackedLinearWeightsBuilder({d, kv, false}).build(store, source, type).weight;
        // Whisper has no key bias, so the packed bias carries a zero key lane.
        std::vector<float> biases(static_cast<size_t>(2 * d), 0.0f);
        const auto v = source.require_f32_tensor(prefix + ".v_proj.bias", {d});
        std::copy(v.values.begin(), v.values.end(), biases.begin() + d);
        w.qkv_bias = store.make_f32(core::TensorShape::from_dims({2 * d}), std::move(biases));
        const auto o = linear(prefix + ".out_proj");
        w.out_weight = o.weight;
        w.out_bias = o.bias;
        return w;
    }
    // Packed [Q; K; V] self-attention projections with a zero key-bias lane.
    std::pair<core::TensorValue, core::TensorValue> packed_qkv(const std::string & prefix) const {
        std::vector<modules::PackedLinearSource> qkv{{prefix + ".q_proj.weight", std::nullopt, d},
            {prefix + ".k_proj.weight", std::nullopt, d}, {prefix + ".v_proj.weight", std::nullopt, d}};
        auto weight = modules::PackedLinearWeightsBuilder({d, qkv, false}).build(store, source, type).weight;
        std::vector<float> biases(static_cast<size_t>(3 * d), 0.0f);
        const auto q = source.require_f32_tensor(prefix + ".q_proj.bias", {d});
        const auto v = source.require_f32_tensor(prefix + ".v_proj.bias", {d});
        std::copy(q.values.begin(), q.values.end(), biases.begin());
        std::copy(v.values.begin(), v.values.end(), biases.begin() + 2 * d);
        return {weight, store.make_f32(core::TensorShape::from_dims({3 * d}), std::move(biases))};
    }
    modules::FeedForwardWeights mlp(const std::string & prefix) const {
        const auto fc1 = linear(prefix + ".fc1"), fc2 = linear(prefix + ".fc2");
        return {fc1.weight, fc1.bias, fc2.weight, fc2.bias};
    }
};

std::vector<uint16_t> host_f16(const assets::TensorSource & source, const std::string & name,
    std::initializer_list<int64_t> shape) {
    const auto values = source.require_f32_tensor(name, shape).values;
    std::vector<uint16_t> out(values.size());
    ggml_fp32_to_fp16_row(values.data(), reinterpret_cast<ggml_fp16_t *>(out.data()), static_cast<int64_t>(values.size()));
    return out;
}

DrafterWeights load_drafter(const EddaAssets & assets, core::BackendWeightStore & store, StorageType type) {
    const auto & config = *assets.drafter;
    const auto & source = *assets.drafter_source;
    const int64_t d = assets.encoder.n_audio_state;
    const Loader load{store, source, type, d};
    DrafterWeights out;
    for (int64_t i = 0; i < config.layers; ++i) {
        const auto p = "layers." + std::to_string(i);
        DrafterLayerWeights layer;
        layer.attention_norm = load.norm(p + ".self_attn_layer_norm");
        layer.cross_norm = load.norm(p + ".encoder_attn_layer_norm");
        layer.mlp_norm = load.norm(p + ".final_layer_norm");
        std::tie(layer.qkv_weight, layer.qkv_bias) = load.packed_qkv(p + ".self_attn");
        const auto o = load.linear(p + ".self_attn.out_proj");
        layer.out_weight = o.weight;
        layer.out_bias = *o.bias;
        layer.cross_attention = load.cross_attention(p + ".encoder_attn");
        layer.mlp = load.mlp(p);
        const auto groups = d / config.conv_group;
        for (auto [name, conv] : {std::pair{".attention_conv", &layer.attention_conv},
                                  std::pair{".mlp_conv", &layer.mlp_conv}}) {
            conv->base = load.f32(p + name + ".base_kernel", {2, config.conv_kernel, d});
            conv->projection = load.linear(p + name + ".kernel_projection", false).weight;
            if (conv->projection.shape.dims[0] != 2 * config.conv_kernel * groups) {
                throw std::runtime_error("Edda drafter kernel projection has an unexpected shape");
            }
        }
        out.layers.push_back(std::move(layer));
    }
    out.context_fc = load.linear("fc", false).weight;
    out.context_norm = load.norm("hidden_norm", false);
    out.final_norm = load.norm("norm");
    out.mask_embedding = load.f32("mask_embedding", {d});
    out.selector_projection = load.linear("candidate_selector.hidden_projection", false).weight;
    out.predecessor_codebook = host_f16(source, "candidate_selector.predecessor_codebook.weight",
        {assets.vocabulary_size, config.selector_rank});
    out.successor_codebook = host_f16(source, "candidate_selector.successor_codebook.weight",
        {assets.vocabulary_size, config.selector_rank});
    return out;
}

}  // namespace

std::unique_ptr<EddaWeights> load_weights(const EddaAssets & assets, core::ExecutionContext & execution,
    StorageType type, bool with_drafter) {
    auto out = std::make_unique<EddaWeights>();
    out->store = std::make_unique<core::BackendWeightStore>(execution.backend(), execution.backend_type(),
        "edda.weights", 4 * 1024 * 1024);
    auto & store = *out->store;
    const auto & source = *assets.source;
    const int64_t d = assets.encoder.n_audio_state;
    const Loader load{store, source, type, d};
    auto & e = out->encoder;
    e.conv1 = binding::conv1d_from_source(store, source, "model.encoder.conv1", StorageType::F32,
        d, assets.encoder.n_mels, 3, true);
    e.conv2 = binding::conv1d_from_source(store, source, "model.encoder.conv2", StorageType::F32, d, d, 3, true);
    e.positions = load.f32("model.encoder.embed_positions.weight", {assets.encoder.n_audio_ctx, d});
    e.final_norm = load.norm("model.encoder.layer_norm");
    for (int64_t i = 0; i < assets.encoder.n_audio_layer; ++i) {
        const auto p = "model.encoder.layers." + std::to_string(i);
        EncoderLayerWeights layer;
        layer.attention_norm = load.norm(p + ".self_attn_layer_norm");
        std::tie(layer.qkv_weight, layer.qkv_bias) = load.packed_qkv(p + ".self_attn");
        const auto o = load.linear(p + ".self_attn.out_proj");
        layer.out_weight = o.weight;
        layer.out_bias = *o.bias;
        layer.mlp_norm = load.norm(p + ".final_layer_norm");
        layer.mlp = load.mlp(p);
        e.layers.push_back(std::move(layer));
    }
    out->embedding = store.load_tensor(source, "model.decoder.embed_tokens.weight", type, {assets.vocabulary_size, d});
    out->positions = load.f32("model.decoder.embed_positions.weight", {assets.max_positions, d});
    out->decoder_norm = load.norm("model.decoder.layer_norm");
    for (int64_t i = 0; i < assets.decoder_layers; ++i) {
        const auto p = "model.decoder.layers." + std::to_string(i);
        TargetLayerWeights layer;
        layer.attention_norm = load.norm(p + ".self_attn_layer_norm");
        layer.cross_norm = load.norm(p + ".encoder_attn_layer_norm");
        layer.mlp_norm = load.norm(p + ".final_layer_norm");
        const auto a = p + ".self_attn";
        std::tie(layer.qkv_weight, layer.qkv_bias) = load.packed_qkv(a);
        const auto o = load.linear(a + ".out_proj");
        layer.out_weight = o.weight;
        layer.out_bias = *o.bias;
        layer.cross_attention = load.cross_attention(p + ".encoder_attn");
        layer.mlp = load.mlp(p);
        out->decoder.push_back(std::move(layer));
    }
    if (with_drafter && assets.drafter) {
        out->drafter = load_drafter(assets, store, type);
    }
    store.upload();
    return out;
}

}  // namespace engine::community_models::edda_asr
