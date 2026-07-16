#include "engine/models/vibevoice_asr/assets.h"

#include "engine/framework/assets/model_package.h"
#include "engine/framework/io/config.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::vibevoice_asr {
namespace json = engine::io::json;
namespace {

VibeVoiceTokenizerConfig parse_tokenizer_config(const json::Value & value, const char * label) {
    VibeVoiceTokenizerConfig config;
    config.channels = json::optional_i64(value, "channels", config.channels);
    config.causal = json::optional_bool(value, "causal", config.causal);
    config.vae_dim = json::require_i64(value, "vae_dim");
    config.fix_std = json::optional_f32(value, "fix_std", config.fix_std);
    config.std_dist_type = json::optional_string(value, "std_dist_type", config.std_dist_type);
    config.mixer_layer = json::optional_string(value, "mixer_layer", config.mixer_layer);
    config.conv_norm = json::optional_string(value, "conv_norm", config.conv_norm);
    config.pad_mode = json::optional_string(value, "pad_mode", config.pad_mode);
    config.disable_last_norm = json::optional_bool(value, "disable_last_norm", config.disable_last_norm);
    config.layernorm = json::optional_string(value, "layernorm", config.layernorm);
    config.layernorm_eps = json::optional_f32(value, "layernorm_eps", config.layernorm_eps);
    config.layernorm_elementwise_affine =
        json::optional_bool(value, "layernorm_elementwise_affine", config.layernorm_elementwise_affine);
    config.conv_bias = json::optional_bool(value, "conv_bias", config.conv_bias);
    config.layer_scale_init_value =
        json::optional_f32(value, "layer_scale_init_value", config.layer_scale_init_value);
    config.weight_init_value = json::optional_f32(value, "weight_init_value", config.weight_init_value);
    config.encoder_n_filters = json::require_i64(value, "encoder_n_filters");
    config.encoder_ratios = json::optional_i64_array(value, "encoder_ratios");
    config.encoder_depths = json::optional_string(value, "encoder_depths", config.encoder_depths);
    config.decoder_n_filters = json::optional_i64(value, "decoder_n_filters", config.encoder_n_filters);
    config.decoder_ratios = json::optional_i64_array(value, "decoder_ratios", config.encoder_ratios);
    config.decoder_depths = json::optional_string(value, "decoder_depths", config.decoder_depths);
    engine::io::require_positive(config.channels, label);
    engine::io::require_positive(config.vae_dim, label);
    engine::io::require_positive(config.encoder_n_filters, label);
    engine::io::require_positive(config.decoder_n_filters, label);
    if (!config.causal) {
        throw std::runtime_error(std::string("VibeVoice ") + label + " must be causal");
    }
    if (config.encoder_ratios.empty()) {
        throw std::runtime_error(std::string("VibeVoice ") + label + " encoder ratios must be non-empty");
    }
    if (config.decoder_ratios.empty()) {
        config.decoder_ratios = config.encoder_ratios;
    }
    for (const auto ratio : config.encoder_ratios) {
        engine::io::require_positive(ratio, label);
    }
    for (const auto ratio : config.decoder_ratios) {
        engine::io::require_positive(ratio, label);
    }
    if (config.mixer_layer != "depthwise_conv") {
        throw std::runtime_error(std::string("VibeVoice config ") + label + " mixer_layer mismatch");
    }
    if (config.conv_norm != "none") {
        throw std::runtime_error(std::string("VibeVoice config ") + label + " conv_norm mismatch");
    }
    if (config.layernorm != "RMSNorm") {
        throw std::runtime_error(std::string("VibeVoice config ") + label + " layernorm mismatch");
    }
    return config;
}

VibeVoiceDecoderConfig parse_decoder_config(const json::Value & value) {
    const auto model_type = json::optional_string(value, "model_type", "");
    if (model_type != "qwen2") {
        throw std::runtime_error("VibeVoice config decoder model_type mismatch");
    }
    VibeVoiceDecoderConfig config;
    config.hidden_size = json::require_i64(value, "hidden_size");
    config.intermediate_size = json::require_i64(value, "intermediate_size");
    config.max_position_embeddings = json::require_i64(value, "max_position_embeddings");
    config.max_window_layers = json::optional_i64(value, "max_window_layers", config.max_window_layers);
    config.num_attention_heads = json::require_i64(value, "num_attention_heads");
    config.num_hidden_layers = json::require_i64(value, "num_hidden_layers");
    config.num_key_value_heads = json::require_i64(value, "num_key_value_heads");
    config.vocab_size = json::require_i64(value, "vocab_size");
    config.rms_norm_eps = json::optional_f32(value, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = json::optional_f32(value, "rope_theta", config.rope_theta);
    config.tie_word_embeddings = json::optional_bool(value, "tie_word_embeddings", config.tie_word_embeddings);
    config.use_cache = json::optional_bool(value, "use_cache", config.use_cache);
    config.use_sliding_window = json::optional_bool(value, "use_sliding_window", config.use_sliding_window);
    engine::io::require_positive(config.hidden_size, "decoder hidden_size");
    engine::io::require_positive(config.intermediate_size, "decoder intermediate_size");
    engine::io::require_positive(config.max_position_embeddings, "decoder max_position_embeddings");
    engine::io::require_positive(config.num_attention_heads, "decoder num_attention_heads");
    engine::io::require_positive(config.num_hidden_layers, "decoder num_hidden_layers");
    engine::io::require_positive(config.num_key_value_heads, "decoder num_key_value_heads");
    engine::io::require_positive(config.vocab_size, "decoder vocab_size");
    engine::io::require_divisible(config.hidden_size, config.num_attention_heads, "decoder hidden_size / heads");
    engine::io::require_divisible(config.num_attention_heads, config.num_key_value_heads, "decoder grouped query heads");
    config.head_dim = config.hidden_size / config.num_attention_heads;
    if (config.use_sliding_window) {
        throw std::runtime_error("VibeVoice decoder sliding-window attention is not expected for 1.5B");
    }
    return config;
}

VibeVoiceDiffusionHeadConfig parse_diffusion_head_config(const json::Value & value) {
    VibeVoiceDiffusionHeadConfig config;
    config.ddpm_batch_mul = json::optional_i64(value, "ddpm_batch_mul", config.ddpm_batch_mul);
    config.ddpm_beta_schedule = json::optional_string(value, "ddpm_beta_schedule", config.ddpm_beta_schedule);
    config.ddpm_num_inference_steps =
        json::optional_i64(value, "ddpm_num_inference_steps", config.ddpm_num_inference_steps);
    config.ddpm_num_steps = json::optional_i64(value, "ddpm_num_steps", config.ddpm_num_steps);
    config.diffusion_type = json::optional_string(value, "diffusion_type", config.diffusion_type);
    config.head_ffn_ratio = json::optional_f32(value, "head_ffn_ratio", config.head_ffn_ratio);
    config.head_layers = json::optional_i64(value, "head_layers", config.head_layers);
    config.hidden_size = json::require_i64(value, "hidden_size");
    config.latent_size = json::require_i64(value, "latent_size");
    config.prediction_type = json::optional_string(value, "prediction_type", config.prediction_type);
    config.rms_norm_eps = json::optional_f32(value, "rms_norm_eps", config.rms_norm_eps);
    config.speech_vae_dim = json::optional_i64(value, "speech_vae_dim", config.latent_size);
    engine::io::require_positive(config.ddpm_batch_mul, "diffusion ddpm_batch_mul");
    engine::io::require_positive(config.ddpm_num_inference_steps, "diffusion ddpm_num_inference_steps");
    engine::io::require_positive(config.ddpm_num_steps, "diffusion ddpm_num_steps");
    engine::io::require_positive(config.head_layers, "diffusion head_layers");
    engine::io::require_positive(config.hidden_size, "diffusion hidden_size");
    engine::io::require_positive(config.latent_size, "diffusion latent_size");
    engine::io::require_positive(config.speech_vae_dim, "diffusion speech_vae_dim");
    if (config.diffusion_type != "ddpm") {
        throw std::runtime_error("VibeVoice config diffusion_type mismatch");
    }
    if (config.prediction_type != "v_prediction") {
        throw std::runtime_error("VibeVoice config diffusion prediction_type mismatch");
    }
    if (config.ddpm_beta_schedule != "cosine") {
        throw std::runtime_error("VibeVoice config diffusion beta schedule mismatch");
    }
    if (config.speech_vae_dim != config.latent_size) {
        throw std::runtime_error("VibeVoice diffusion speech_vae_dim must match latent_size");
    }
    return config;
}

int64_t require_acoustic_vae_dim(const json::Value & root) {
    // VibeVoice-7B's config.json spells this key "acostic_vae_dim".
    for (const char * key : {"acoustic_vae_dim", "acostic_vae_dim"}) {
        if (const auto * value = root.find(key); value != nullptr) {
            return value->as_i64();
        }
    }
    throw std::runtime_error("VibeVoice config is missing acoustic_vae_dim");
}

VibeVoiceConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    VibeVoiceConfig config;
    config.model_type = json::optional_string(root, "model_type", "");
    if (config.model_type != "vibevoice") {
        throw std::runtime_error("VibeVoice config model_type mismatch");
    }
    config.torch_dtype = json::optional_string(root, "torch_dtype", config.torch_dtype);
    config.acoustic_vae_dim = require_acoustic_vae_dim(root);
    config.semantic_vae_dim = json::require_i64(root, "semantic_vae_dim");
    config.acoustic_tokenizer = parse_tokenizer_config(root.require("acoustic_tokenizer_config"), "acoustic tokenizer");
    config.semantic_tokenizer = parse_tokenizer_config(root.require("semantic_tokenizer_config"), "semantic tokenizer");
    config.decoder = parse_decoder_config(root.require("decoder_config"));
    config.diffusion_head = parse_diffusion_head_config(root.require("diffusion_head_config"));
    engine::io::require_positive(config.acoustic_vae_dim, "acoustic_vae_dim");
    engine::io::require_positive(config.semantic_vae_dim, "semantic_vae_dim");
    if (config.acoustic_vae_dim != config.acoustic_tokenizer.vae_dim) {
        throw std::runtime_error("VibeVoice acoustic_vae_dim does not match acoustic tokenizer vae_dim");
    }
    if (config.semantic_vae_dim != config.semantic_tokenizer.vae_dim) {
        throw std::runtime_error("VibeVoice semantic_vae_dim does not match semantic tokenizer vae_dim");
    }
    if (config.diffusion_head.hidden_size != config.decoder.hidden_size) {
        throw std::runtime_error("VibeVoice diffusion hidden_size must match decoder hidden_size");
    }
    if (config.diffusion_head.latent_size != config.acoustic_vae_dim) {
        throw std::runtime_error("VibeVoice diffusion latent_size must match acoustic_vae_dim");
    }
    return config;
}

VibeVoiceProcessorConfig parse_processor_config(const assets::ResourceBundle & resources) {
    if (resources.find_file("preprocessor_config") == nullptr) {
        return {};
    }
    const auto root = resources.parse_json("preprocessor_config");
    VibeVoiceProcessorConfig config;
    config.speech_tok_compress_ratio =
        json::optional_i64(root, "speech_tok_compress_ratio", config.speech_tok_compress_ratio);
    config.db_normalize = json::optional_bool(root, "db_normalize", config.db_normalize);
    config.language_model_pretrained_name =
        json::optional_string(root, "language_model_pretrained_name", config.language_model_pretrained_name);
    if (const auto * audio = root.find("audio_processor"); audio != nullptr) {
        config.audio_processor.sample_rate =
            static_cast<int>(json::optional_i64(*audio, "sampling_rate", config.audio_processor.sample_rate));
        config.audio_processor.normalize_audio =
            json::optional_bool(*audio, "normalize_audio", config.audio_processor.normalize_audio);
        config.audio_processor.target_db_fs =
            json::optional_f32(*audio, "target_dB_FS", config.audio_processor.target_db_fs);
        config.audio_processor.eps = json::optional_f32(*audio, "eps", config.audio_processor.eps);
    }
    engine::io::require_positive(config.speech_tok_compress_ratio, "processor speech_tok_compress_ratio");
    engine::io::require_positive(config.audio_processor.sample_rate, "processor sampling_rate");
    if (config.language_model_pretrained_name.empty()) {
        throw std::runtime_error("VibeVoice processor language_model_pretrained_name must not be empty");
    }
    return config;
}

void validate_weight_anchors(const VibeVoiceASRAssets & assets) {
    const auto & config = assets.config;
    const auto & weights = *assets.model_weights;
    const auto & decoder = config.decoder;
    assets::require_tensor_shape(weights, "model.language_model.embed_tokens.weight", {decoder.vocab_size, decoder.hidden_size});
    if (!decoder.tie_word_embeddings) {
        const auto lm_head_name = weights.require_tensor_name({"lm_head.weight", "model.lm_head.weight"});
        assets::require_tensor_shape(weights, lm_head_name, {decoder.vocab_size, decoder.hidden_size});
    }
    assets::require_tensor_shape(weights, "model.language_model.norm.weight", {decoder.hidden_size});
    assets::require_tensor_shape(weights, "model.language_model.layers.0.self_attn.q_proj.weight", {decoder.hidden_size, decoder.hidden_size});
    assets::require_tensor_shape(
        weights,
        "model.language_model.layers.0.self_attn.k_proj.weight",
        {decoder.num_key_value_heads * decoder.head_dim, decoder.hidden_size});
    assets::require_tensor_shape(
        weights,
        "model.language_model.layers.0.self_attn.v_proj.weight",
        {decoder.num_key_value_heads * decoder.head_dim, decoder.hidden_size});
    assets::require_tensor_shape(weights, "model.language_model.layers.0.self_attn.o_proj.weight", {decoder.hidden_size, decoder.hidden_size});
    assets::require_tensor_shape(weights, "model.language_model.layers.0.mlp.gate_proj.weight", {decoder.intermediate_size, decoder.hidden_size});
    assets::require_tensor_shape(weights, "model.language_model.layers.0.mlp.up_proj.weight", {decoder.intermediate_size, decoder.hidden_size});
    assets::require_tensor_shape(weights, "model.language_model.layers.0.mlp.down_proj.weight", {decoder.hidden_size, decoder.intermediate_size});

    assets::require_tensor_shape(weights, "model.acoustic_connector.fc1.weight", {decoder.hidden_size, config.acoustic_vae_dim});
    assets::require_tensor_shape(weights, "model.acoustic_connector.norm.weight", {decoder.hidden_size});
    assets::require_tensor_shape(weights, "model.acoustic_connector.fc2.weight", {decoder.hidden_size, decoder.hidden_size});
    assets::require_tensor_shape(weights, "model.semantic_connector.fc1.weight", {decoder.hidden_size, config.semantic_vae_dim});
    assets::require_tensor_shape(weights, "model.semantic_connector.norm.weight", {decoder.hidden_size});
    assets::require_tensor_shape(weights, "model.semantic_connector.fc2.weight", {decoder.hidden_size, decoder.hidden_size});

    assets::require_tensor_shape(
        weights,
        "model.acoustic_tokenizer.encoder.downsample_layers.0.0.conv.conv.weight",
        {config.acoustic_tokenizer.encoder_n_filters, config.acoustic_tokenizer.channels, 7});
    assets::require_tensor_shape(
        weights,
        "model.semantic_tokenizer.encoder.downsample_layers.0.0.conv.conv.weight",
        {config.semantic_tokenizer.encoder_n_filters, config.semantic_tokenizer.channels, 7});
}

}  // namespace

std::shared_ptr<const VibeVoiceASRAssets> load_vibevoice_asr_assets(const std::filesystem::path & model_path) {
    auto resources = assets::load_resource_bundle_from_package_spec(
        model_path,
        assets::default_model_package_spec_path("vibevoice_asr"));
    VibeVoiceASRAssets assets;
    assets.resources = std::move(resources);
    assets.config = parse_config(assets.resources);
    assets.processor = parse_processor_config(assets.resources);
    assets.model_weights = assets.resources.open_tensor_source("model_weights");
    validate_weight_anchors(assets);
    return std::make_shared<VibeVoiceASRAssets>(std::move(assets));
}

}  // namespace engine::models::vibevoice_asr
