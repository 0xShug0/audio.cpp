#include "engine/models/dots_tts/assets.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/config.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <string>

namespace engine::models::dots_tts {
namespace {

namespace json = engine::io::json;

DotsTransformerConfig parse_transformer_config(const json::Value & root, const std::string & key) {
    const auto & value = root.require(key);
    DotsTransformerConfig config;
    config.num_layers = json::require_i64(value, "num_layers");
    config.num_heads = json::require_i64(value, "num_heads");
    config.hidden_size = json::require_i64(value, "hidden_size");
    config.ffn_hidden_size = json::require_i64(value, "ffn_hidden_size");
    config.modulation = json::optional_bool(value, "modulation", config.modulation);
    config.qkv_bias = json::optional_bool(value, "qkv_bias", config.qkv_bias);
    config.qk_norm = json::optional_bool(value, "qk_norm", config.qk_norm);
    config.attn_dropout = json::optional_f32(value, "attn_dropout", config.attn_dropout);
    config.dropout = json::optional_f32(value, "dropout", config.dropout);
    config.norm_layer = json::optional_string(value, "norm_layer", config.norm_layer);
    config.alibi_bias = json::optional_bool(value, "alibi_bias", config.alibi_bias);
    config.rotary_bias = json::optional_bool(value, "rotary_bias", config.rotary_bias);
    config.rotary_theta = json::optional_f32(value, "rotary_theta", config.rotary_theta);
    config.input_dim = json::optional_i64(value, "input_dim", config.input_dim);
    config.causal = json::optional_bool(value, "causal", config.causal);
    return config;
}

DotsVocoderConfig parse_vocoder_config(const json::Value & root) {
    const auto & value = root.require("vocoder");
    DotsVocoderConfig config;
    config.sample_rate = static_cast<int>(json::require_i64(value, "sample_rate"));
    config.upsample_rates = json::require_i64_array(value, "upsample_rates");
    config.upsample_kernel_sizes = json::require_i64_array(value, "upsample_kernel_sizes");
    config.upsample_initial_channel = json::require_i64(value, "upsample_initial_channel");
    config.resblock = json::require_string(value, "resblock");
    config.resblock_kernel_sizes = json::require_i64_array(value, "resblock_kernel_sizes");
    const auto & dilation_rows = value.require("resblock_dilation_sizes").as_array();
    config.resblock_dilation_sizes.reserve(dilation_rows.size());
    for (const auto & row : dilation_rows) {
        config.resblock_dilation_sizes.push_back(json::number_array_as<int64_t>(row));
    }
    config.downsample_rates = json::require_i64_array(value, "downsample_rates");
    config.downsample_channels = json::require_i64_array(value, "downsample_channels");
    config.activation = json::require_string(value, "activation");
    config.snake_logscale = json::optional_bool(value, "snake_logscale", config.snake_logscale);
    config.latent_dim = json::require_i64(value, "latent_dim");
    config.causal = json::optional_bool(value, "causal", config.causal);
    config.mi_num_layers = json::require_i64(value, "mi_num_layers");
    config.causal_encoder = json::optional_bool(value, "causal_encoder", config.causal_encoder);
    config.use_bias_at_final = json::optional_bool(value, "use_bias_at_final", config.use_bias_at_final);
    config.use_tanh_at_final = json::optional_bool(value, "use_tanh_at_final", config.use_tanh_at_final);
    return config;
}

DotsLlmConfig parse_llm_config(const json::Value & root) {
    DotsLlmConfig config;
    config.vocab_size = json::require_i64(root, "vocab_size");
    config.max_position_embeddings = json::require_i64(root, "max_position_embeddings");
    config.hidden_size = json::require_i64(root, "hidden_size");
    config.intermediate_size = json::require_i64(root, "intermediate_size");
    config.num_hidden_layers = json::require_i64(root, "num_hidden_layers");
    config.num_attention_heads = json::require_i64(root, "num_attention_heads");
    config.num_key_value_heads = json::require_i64(root, "num_key_value_heads");
    config.rms_norm_eps = json::optional_f32(root, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = json::optional_f32(root, "rope_theta", config.rope_theta);
    config.bos_token_id = json::optional_i64(root, "bos_token_id", config.bos_token_id);
    config.eos_token_id = json::optional_i64(root, "eos_token_id", config.eos_token_id);
    return config;
}

DotsConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    DotsConfig config;
    config.latent_dim = json::require_i64(root, "latent_dim");
    config.patch_size = json::require_i64(root, "patch_size");
    config.cfg_droprate = json::optional_f32(root, "cfg_droprate", config.cfg_droprate);
    config.patch_encoder = parse_transformer_config(root, "PatchEncoder");
    config.dit = parse_transformer_config(root, "DiT");
    config.vocoder = parse_vocoder_config(root);
    config.fm_sigma = json::optional_f32(root, "fm_sigma", config.fm_sigma);
    config.xvec_drop_rate = json::optional_f32(root, "xvec_drop_rate", config.xvec_drop_rate);
    config.campplus_embedding_size = json::optional_i64(root, "campplus_embedding_size", config.campplus_embedding_size);
    config.xvec_max_audio_seconds = json::optional_f32(root, "xvec_max_audio_seconds", config.xvec_max_audio_seconds);
    if (const auto * meanflow = root.find("meanflow"); meanflow != nullptr && meanflow->is_object()) {
        DotsMeanFlowConfig meanflow_config;
        meanflow_config.enabled = json::optional_bool(*meanflow, "enabled", meanflow_config.enabled);
        meanflow_config.use_duration_embedding =
            json::optional_bool(*meanflow, "use_duration_embedding", meanflow_config.use_duration_embedding);
        config.meanflow = meanflow_config;
    }
    config.llm = parse_llm_config(resources.parse_json("llm_config"));
    return config;
}

void validate_transformer(const DotsTransformerConfig & config, const std::string & name) {
    engine::io::require_positive(config.num_layers, name + " num_layers");
    engine::io::require_positive(config.num_heads, name + " num_heads");
    engine::io::require_positive(config.hidden_size, name + " hidden_size");
    engine::io::require_positive(config.ffn_hidden_size, name + " ffn_hidden_size");
    engine::io::require_divisible(config.hidden_size, config.num_heads, name + " hidden_size / num_heads");
}

void validate_config(const DotsConfig & config) {
    engine::io::require_positive(config.latent_dim, "DotTTS latent_dim");
    engine::io::require_positive(config.patch_size, "DotTTS patch_size");
    engine::io::require_divisible(config.patch_size, int64_t{2}, "DotTTS patch_size / semantic encoder downsample");
    validate_transformer(config.patch_encoder, "DotTTS PatchEncoder");
    validate_transformer(config.dit, "DotTTS DiT");
    engine::io::require_positive(config.vocoder.sample_rate, "DotTTS vocoder sample_rate");
    engine::io::require_positive(config.vocoder.latent_dim, "DotTTS vocoder latent_dim");
    engine::io::require_positive(config.vocoder.upsample_initial_channel, "DotTTS vocoder upsample_initial_channel");
    if (config.latent_dim != config.vocoder.latent_dim || config.patch_encoder.input_dim != config.latent_dim) {
        throw std::runtime_error("DotTTS latent dimensions must match config, PatchEncoder, and vocoder");
    }
    if (config.patch_encoder.hidden_size <= 0 || config.patch_encoder.hidden_size * (config.patch_size / 2) <= 0) {
        throw std::runtime_error("DotTTS PatchEncoder output projection dimensions are invalid");
    }
    if (config.llm.hidden_size <= 0 || config.llm.vocab_size <= 0 || config.llm.num_hidden_layers <= 0) {
        throw std::runtime_error("DotTTS LLM config contains invalid dimensions");
    }
    engine::io::require_divisible(config.llm.hidden_size, config.llm.num_attention_heads, "DotTTS LLM hidden_size / heads");
    engine::io::require_divisible(config.llm.num_attention_heads, config.llm.num_key_value_heads, "DotTTS LLM heads / kv_heads");
}

void validate_weight_anchors(const DotsAssets & assets) {
    const auto & config = assets.config;
    assets::require_tensor_shape(*assets.core_weights, "llm.model.embed_tokens.weight", {config.llm.vocab_size, config.llm.hidden_size});
    assets::require_tensor_shape(
        *assets.core_weights,
        "llm.model.layers.0.self_attn.q_proj.weight",
        {config.llm.hidden_size, config.llm.hidden_size});
    assets::require_tensor_shape(*assets.core_weights, "patch_encoder.ds_proj.weight", {config.latent_dim, config.latent_dim, 2});
    assets::require_tensor_shape(
        *assets.core_weights,
        "velocity_field_predictor.input_layer.weight",
        {config.dit.hidden_size, config.dit.hidden_size});
    assets::require_tensor_shape(*assets.core_weights, "xvec_proj.0.weight", {config.dit.hidden_size, config.campplus_embedding_size});
    assets::require_tensor_shape(*assets.vocoder_weights, "post_proj.weight", {config.latent_dim, config.latent_dim, 1});
    assets::require_tensor_shape(
        *assets.vocoder_weights,
        "decoder.conv_pre.weight",
        {config.vocoder.upsample_initial_channel, config.latent_dim, 5});
    assets::require_tensor_shape(*assets.speaker_encoder_weights, "model.head.conv1.weight", {32, 1, 3, 3});
    assets::require_tensor_shape(*assets.latent_stats, "mean", {config.latent_dim});
    assets::require_tensor_shape(*assets.latent_stats, "var", {config.latent_dim});
}

}  // namespace

std::shared_ptr<const DotsAssets> load_dots_assets(const std::filesystem::path & model_path) {
    DotsAssets assets;
    assets.resources = model_spec::load_resource_bundle(
        model_path,
        model_spec::default_spec_path("dots_tts"));
    assets.config = parse_config(assets.resources);
    validate_config(assets.config);
    assets.core_weights = assets.resources.open_tensor_source("core_weights");
    assets.vocoder_weights = assets.resources.open_tensor_source("vocoder_weights");
    assets.speaker_encoder_weights = assets.resources.open_tensor_source("speaker_encoder_weights");
    assets.latent_stats = assets.resources.open_tensor_source("latent_stats");
    validate_weight_anchors(assets);
    return std::make_shared<DotsAssets>(std::move(assets));
}

}  // namespace engine::models::dots_tts
