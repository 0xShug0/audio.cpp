#include "engine/models/foundation_1/assets.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/io/config.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::foundation_1 {
namespace json = engine::io::json;

std::shared_ptr<const FoundationAssets> load_foundation_assets(const std::filesystem::path & model_path) {
    auto out = std::make_shared<FoundationAssets>();
    out->resources = model_spec::load_resource_bundle_for_family(model_path, "foundation_1");
    const auto root = out->resources.parse_json("config");
    const auto & model = root.require("model");
    if (json::require_string(root, "model_type") != "diffusion_cond")
        throw std::runtime_error("Foundation-1 requires diffusion_cond");
    out->sample_rate = json::require_i64(root, "sample_rate");
    out->sample_size = json::require_i64(root, "sample_size");
    io::require_positive(out->sample_rate, "sample_rate");
    io::require_positive(out->sample_size, "sample_size");

    const auto & conditioning = model.require("conditioning");
    const auto cond_dim = json::require_i64(conditioning, "cond_dim");
    bool prompt_found = false, start_found = false, total_found = false;
    for (const auto & item : conditioning.require("configs").as_array()) {
        const auto id = json::require_string(item, "id");
        const auto type = json::require_string(item, "type");
        const auto & c = item.require("config");
        if (id == "prompt" && type == "t5") {
            if (json::require_string(c, "t5_model_name") != "t5-base" ||
                json::optional_bool(c, "project_out", false) ||
                json::optional_string(c, "padding_mode", "zero") != "zero")
                throw std::runtime_error("Foundation-1 requires unprojected, zero-padded T5-base conditioning");
            out->max_text_tokens = json::require_i64(c, "max_length");
            prompt_found = true;
        } else if ((id == "seconds_start" || id == "seconds_total") && type == "number") {
            if (json::require_f32(c, "min_val") != 0 || json::require_f32(c, "max_val") != 512 ||
                json::optional_string(c, "fourier_features_type", "learned") != "learned")
                throw std::runtime_error("Foundation-1 requires learned seconds conditioning over [0,512]");
            if (id == "seconds_start") start_found = true;
            else total_found = true;
        } else {
            throw std::runtime_error("Unsupported Foundation-1 conditioner: " + id);
        }
    }
    if (!prompt_found || !start_found || !total_found)
        throw std::runtime_error("Foundation-1 requires prompt, seconds_start and seconds_total");
    io::require_positive(out->max_text_tokens, "max_length");

    const auto text = out->resources.parse_json("text_config");
    auto & t = out->text_config;
    t.hidden_size = json::require_i64(text, "d_model");
    t.layers = json::require_i64(text, "num_layers");
    t.attention_heads = json::require_i64(text, "num_heads");
    t.head_dim = json::require_i64(text, "d_kv");
    t.intermediate_size = json::require_i64(text, "d_ff");
    t.vocab_size = json::require_i64(text, "vocab_size");
    t.relative_attention_num_buckets = json::require_i64(text, "relative_attention_num_buckets");
    t.relative_attention_max_distance = json::optional_i64(text, "relative_attention_max_distance", 128);
    t.rms_norm_eps = json::require_f32(text, "layer_norm_epsilon");
    if (json::optional_string(text, "feed_forward_proj", "relu") != "relu" || t.hidden_size != cond_dim)
        throw std::runtime_error("Foundation-1 requires ReLU T5-base with matching conditioning width");
    t.feed_forward_kind = modules::T5BaseFeedForwardKind::Relu;
    t.shared_relative_position_bias = true;
    (void) out->resources.require_file("tokenizer"); // Spec resource is text_encoder/spiece.model.

    const auto & diffusion = model.require("diffusion");
    const auto & c = diffusion.require("config");
    if (json::require_string(diffusion, "type") != "dit" ||
        json::require_string(c, "transformer_type") != "continuous_transformer" ||
        json::require_bool(c, "project_cond_tokens") ||
        json::optional_string(diffusion, "diffusion_objective", "v") != "v" ||
        json::optional_string(c, "global_cond_type", "prepend") != "prepend" ||
        json::optional_string(c, "timestep_features_type", "learned") != "learned")
        throw std::runtime_error("Foundation-1 requires velocity DiT with learned time features and global prepend");
    std::vector<std::string> cross_ids, global_ids;
    for (const auto & id : diffusion.require("cross_attention_cond_ids").as_array()) cross_ids.push_back(id.as_string());
    for (const auto & id : diffusion.require("global_cond_ids").as_array()) global_ids.push_back(id.as_string());
    if (cross_ids != std::vector<std::string>{"prompt", "seconds_start", "seconds_total"} ||
        global_ids != std::vector<std::string>{"seconds_start", "seconds_total"})
        throw std::runtime_error("Foundation-1 conditioning order must be prompt/start/total and start/total");
    auto & d = out->dit_config;
    d.input_channels = d.output_channels = json::require_i64(c, "io_channels");
    d.hidden_size = json::require_i64(c, "embed_dim");
    d.layers = json::require_i64(c, "depth");
    d.context_dim = json::require_i64(c, "cond_token_dim");
    d.global_dim = json::require_i64(c, "global_cond_dim");
    d.intermediate_size = 4 * d.hidden_size;
    d.time_features = 256;
    d.norm_eps = 1.e-5F;
    d.normalization = modules::DitNormalization::Layer;
    d.modulation = modules::DitModulation::None;
    d.feed_forward = modules::DitFeedForward::SwiGlu;
    d.prepend_condition_token = true;
    const auto heads = json::require_i64(c, "num_heads");
    io::require_positive(heads, "num_heads");
    io::require_divisible(d.hidden_size, heads, "embed_dim / num_heads");
    const auto head_dim = d.hidden_size / heads;
    io::require_divisible(d.context_dim, head_dim, "cond_token_dim / head_dim");
    if (d.context_dim != cond_dim || d.global_dim != 2 * cond_dim)
        throw std::runtime_error("Foundation-1 DiT conditioning dimensions do not match conditioners");
    d.self_attention.heads = d.self_attention.kv_heads = heads;
    d.self_attention.head_dim = head_dim;
    d.self_attention.lowering = modules::ScaledDotProductAttentionLowering::FlashPreserveViews;
    d.self_attention.rotary = {(std::max)(head_dim / 2, int64_t{32}), GGML_ROPE_TYPE_NEOX, 10000.F};
    d.cross_attention.heads = heads;
    d.cross_attention.kv_heads = d.context_dim / head_dim;
    d.cross_attention.head_dim = head_dim;
    d.cross_attention.lowering = modules::ScaledDotProductAttentionLowering::FlashPreserveViews;
    auto & n = out->dit_names;
    n.input = {"model.model.transformer.project_in.weight", ""};
    n.output = {"model.model.transformer.project_out.weight", ""};
    n.preprocess = {"model.model.preprocess_conv.weight", ""};
    n.postprocess = {"model.model.postprocess_conv.weight", ""};
    n.time_frequencies = "model.model.timestep_features.weight";
    n.rotary_inv_frequencies = "model.model.transformer.rotary_pos_emb.inv_freq";
    n.time_embedding = {{"model.model.to_timestep_embed.0.weight", "model.model.to_timestep_embed.0.bias"},
                        {"model.model.to_timestep_embed.2.weight", "model.model.to_timestep_embed.2.bias"}};
    n.context_embedding = {{"model.model.to_cond_embed.0.weight", ""}, {"model.model.to_cond_embed.2.weight", ""}};
    n.global_embedding = {{"model.model.to_global_embed.0.weight", ""}, {"model.model.to_global_embed.2.weight", ""}};
    n.layer.self_norm = {"model.model.transformer.layers.{layer}.pre_norm.gamma", "model.model.transformer.layers.{layer}.pre_norm.beta"};
    n.layer.cross_norm = {"model.model.transformer.layers.{layer}.cross_attend_norm.gamma", "model.model.transformer.layers.{layer}.cross_attend_norm.beta"};
    n.layer.feed_forward_norm = {"model.model.transformer.layers.{layer}.ff_norm.gamma", "model.model.transformer.layers.{layer}.ff_norm.beta"};
    n.layer.self_attention.qkv = {"model.model.transformer.layers.{layer}.self_attn.to_qkv.weight", ""};
    n.layer.self_attention.output = {"model.model.transformer.layers.{layer}.self_attn.to_out.weight", ""};
    n.layer.cross_attention.query = {"model.model.transformer.layers.{layer}.cross_attn.to_q.weight", ""};
    n.layer.cross_attention.kv = {"model.model.transformer.layers.{layer}.cross_attn.to_kv.weight", ""};
    n.layer.cross_attention.output = {"model.model.transformer.layers.{layer}.cross_attn.to_out.weight", ""};
    n.layer.feed_forward_in = {"model.model.transformer.layers.{layer}.ff.ff.0.proj.weight", "model.model.transformer.layers.{layer}.ff.ff.0.proj.bias"};
    n.layer.feed_forward_out = {"model.model.transformer.layers.{layer}.ff.ff.2.weight", "model.model.transformer.layers.{layer}.ff.ff.2.bias"};

    const auto & pre = model.require("pretransform");
    const auto & codec = pre.require("config");
    if (json::require_string(pre, "type") != "autoencoder" ||
        json::require_string(codec.require("encoder"), "type") != "oobleck" ||
        json::require_string(codec.require("decoder"), "type") != "oobleck" ||
        json::require_string(codec.require("bottleneck"), "type") != "vae")
        throw std::runtime_error("Foundation-1 requires the Oobleck VAE pretransform");
    const auto & enc = codec.require("encoder").require("config");
    const auto & dec = codec.require("decoder").require("config");
    auto & v = out->codec_config;
    v.sample_rate = out->sample_rate;
    v.audio_channels = json::require_i64(root, "audio_channels");
    v.channels = json::require_i64(enc, "channels");
    v.c_mults = json::require_i64_array(enc, "c_mults");
    v.strides = json::require_i64_array(enc, "strides");
    v.encoder_latent_dim = json::require_i64(enc, "latent_dim");
    v.decoder_latent_dim = json::require_i64(dec, "latent_dim");
    v.use_snake = json::require_bool(enc, "use_snake");
    v.final_tanh = json::require_bool(dec, "final_tanh");
    v.encoder_prefix = "pretransform.model.encoder";
    v.decoder_prefix = "pretransform.model.decoder";
    int64_t ratio = 1;
    io::require_all_positive(v.strides, "codec strides");
    for (const auto stride : v.strides) ratio *= stride;
    if (!v.use_snake || !json::require_bool(dec, "use_snake") ||
        json::require_i64(dec, "channels") != v.channels ||
        json::require_i64_array(dec, "c_mults") != v.c_mults || json::require_i64_array(dec, "strides") != v.strides ||
        json::require_i64(enc, "in_channels") != v.audio_channels || json::require_i64(dec, "out_channels") != v.audio_channels ||
        json::require_i64(codec, "downsampling_ratio") != ratio || v.encoder_latent_dim != 2 * v.decoder_latent_dim ||
        v.decoder_latent_dim != d.output_channels)
        throw std::runtime_error("Foundation-1 Oobleck encoder/decoder dimensions do not match");
    out->weights = out->resources.open_tensor_source("weights");
    out->text_weights = out->resources.open_tensor_source("text_weights");
    return out;
}

NumberConditioningWeights load_foundation_number_weights(
    const FoundationAssets & assets, core::ExecutionContext & execution) {
    NumberConditioningWeights out;
    out.storage = std::make_shared<core::BackendWeightStore>(execution.backend(), execution.backend_type(),
        "foundation_1.numbers", 1ull << 20);
    const std::string base = "conditioner.conditioners.";
    out.duration_frequencies = assets.weights->require_f32(base + "seconds_total.embedder.embedding.0.weights", {128});
    out.start = modules::binding::linear_from_source(*out.storage, *assets.weights,
        base + "seconds_start.embedder.embedding.1", assets::TensorStorageType::Native, 768, 257, true);
    out.duration = modules::binding::linear_from_source(*out.storage, *assets.weights,
        base + "seconds_total.embedder.embedding.1", assets::TensorStorageType::Native, 768, 257, true);
    out.storage->upload();
    return out;
}

modules::T5EncoderRuntimeWeights load_foundation_text_weights(
    std::shared_ptr<const FoundationAssets> assets, core::ExecutionContext & execution,
    assets::TensorStorageType storage_type) {
    const auto & c = assets->text_config;
    const auto & source = *assets->text_weights;
    auto store = std::make_shared<core::BackendWeightStore>(execution.backend(), execution.backend_type(),
        "foundation_1.text.weights", 8ull * 1024ull * 1024ull);
    modules::T5BaseEncoderWeights weights;
    weights.embed_tokens = store->load_tensor(source, "shared.weight", storage_type, {c.vocab_size, c.hidden_size});
    weights.relative_attention_bias = store->load_f32_tensor(source,
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
        {c.relative_attention_num_buckets, c.attention_heads});
    weights.final_layer_norm = store->load_f32_tensor(source, "encoder.final_layer_norm.weight", {c.hidden_size});
    for (int64_t i = 0; i < c.layers; ++i) {
        const auto prefix = "encoder.block." + std::to_string(i);
        modules::T5BaseEncoderLayerWeights layer;
        layer.self_attention_layer_norm = store->load_f32_tensor(source, prefix + ".layer.0.layer_norm.weight", {c.hidden_size});
        layer.ffn_layer_norm = store->load_f32_tensor(source, prefix + ".layer.1.layer_norm.weight", {c.hidden_size});
        layer.q_proj.weight = store->load_tensor(source, prefix + ".layer.0.SelfAttention.q.weight", storage_type, {c.attention_heads * c.head_dim, c.hidden_size});
        layer.k_proj.weight = store->load_tensor(source, prefix + ".layer.0.SelfAttention.k.weight", storage_type, {c.attention_heads * c.head_dim, c.hidden_size});
        layer.v_proj.weight = store->load_tensor(source, prefix + ".layer.0.SelfAttention.v.weight", storage_type, {c.attention_heads * c.head_dim, c.hidden_size});
        layer.o_proj.weight = store->load_tensor(source, prefix + ".layer.0.SelfAttention.o.weight", storage_type, {c.hidden_size, c.attention_heads * c.head_dim});
        layer.wi_proj.weight = store->load_tensor(source, prefix + ".layer.1.DenseReluDense.wi.weight", storage_type, {c.intermediate_size, c.hidden_size});
        layer.wo_proj.weight = store->load_tensor(source, prefix + ".layer.1.DenseReluDense.wo.weight", storage_type, {c.hidden_size, c.intermediate_size});
        weights.layers.push_back(std::move(layer));
    }
    store->upload();
    return {std::move(store), std::move(weights), std::nullopt};
}

} // namespace engine::models::foundation_1
