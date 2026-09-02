#include "engine/community_models/voxcpm1/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/config.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <ggml.h>
#include <gguf.h>

#include <stdexcept>
#include <string>
#include <memory>
#include <optional>

namespace engine::community_models::voxcpm1 {
namespace json = engine::io::json;
namespace {

VoxCPM1RopeScalingConfig parse_rope_scaling(const json::Value & value) {
    VoxCPM1RopeScalingConfig config;
    config.type = json::optional_string(value, "type", "");
    config.long_factor = json::optional_f32_array(value, "long_factor");
    config.short_factor = json::optional_f32_array(value, "short_factor");
    config.original_max_position_embeddings =
        json::optional_i64(value, "original_max_position_embeddings", 0);
    return config;
}

VoxCPM1MiniCPMConfig parse_lm_config(const json::Value & value) {
    VoxCPM1MiniCPMConfig config;
    config.bos_token_id = json::optional_i64(value, "bos_token_id", config.bos_token_id);
    config.eos_token_id = json::optional_i64(value, "eos_token_id", config.eos_token_id);
    config.hidden_size = json::require_i64(value, "hidden_size");
    config.intermediate_size = json::require_i64(value, "intermediate_size");
    config.max_position_embeddings = json::require_i64(value, "max_position_embeddings");
    config.num_attention_heads = json::require_i64(value, "num_attention_heads");
    config.num_hidden_layers = json::require_i64(value, "num_hidden_layers");
    config.num_key_value_heads = json::require_i64(value, "num_key_value_heads");
    config.kv_channels = json::optional_i64(value, "kv_channels", config.hidden_size / config.num_attention_heads);
    config.vocab_size = json::require_i64(value, "vocab_size");
    config.scale_emb = json::optional_i64(value, "scale_emb", config.scale_emb);
    config.dim_model_base = json::optional_i64(value, "dim_model_base", config.dim_model_base);
    config.rms_norm_eps = json::optional_f32(value, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = json::optional_f32(value, "rope_theta", config.rope_theta);
    config.scale_depth = json::optional_f32(value, "scale_depth", config.scale_depth);
    config.use_mup = json::optional_bool(value, "use_mup", config.use_mup);
    if (const auto * rope_scaling = value.find("rope_scaling"); rope_scaling != nullptr) {
        config.rope_scaling = parse_rope_scaling(*rope_scaling);
    }
    engine::io::require_positive(config.hidden_size, "lm hidden_size");
    engine::io::require_positive(config.intermediate_size, "lm intermediate_size");
    engine::io::require_positive(config.max_position_embeddings, "lm max_position_embeddings");
    engine::io::require_positive(config.num_attention_heads, "lm num_attention_heads");
    engine::io::require_positive(config.num_hidden_layers, "lm num_hidden_layers");
    engine::io::require_positive(config.num_key_value_heads, "lm num_key_value_heads");
    engine::io::require_positive(config.kv_channels, "lm kv_channels");
    engine::io::require_positive(config.vocab_size, "lm vocab_size");
    engine::io::require_divisible(config.hidden_size, config.num_attention_heads, "lm hidden_size / num_attention_heads");
    engine::io::require_divisible(config.num_attention_heads, config.num_key_value_heads, "lm attention heads");
    if (!config.rope_scaling.type.empty()) {
        if (config.rope_scaling.type != "longrope") {
            throw std::runtime_error("VoxCPM1 currently expects longrope rope_scaling");
        }
        const int64_t expected = config.hidden_size / config.num_attention_heads / 2;
        if (static_cast<int64_t>(config.rope_scaling.long_factor.size()) != expected ||
            static_cast<int64_t>(config.rope_scaling.short_factor.size()) != expected) {
            throw std::runtime_error("VoxCPM1 rope_scaling factor length does not match head_dim / 2");
        }
    }
    return config;
}

VoxCPM1LocalTransformerConfig parse_local_transformer_config(
    const json::Value & value,
    const char * label) {
    VoxCPM1LocalTransformerConfig config;
    config.hidden_dim = json::require_i64(value, "hidden_dim");
    config.ffn_dim = json::require_i64(value, "ffn_dim");
    config.num_heads = json::require_i64(value, "num_heads");
    config.num_layers = json::require_i64(value, "num_layers");
    config.kv_channels = json::optional_i64(value, "kv_channels", config.hidden_dim / config.num_heads);
    engine::io::require_positive(config.hidden_dim, label);
    engine::io::require_positive(config.ffn_dim, label);
    engine::io::require_positive(config.num_heads, label);
    engine::io::require_positive(config.num_layers, label);
    engine::io::require_positive(config.kv_channels, label);
    engine::io::require_divisible(config.hidden_dim, config.num_heads, label);
    return config;
}

VoxCPM1DiTConfig parse_dit_config(const json::Value & value) {
    const auto base = parse_local_transformer_config(value, "dit transformer");
    VoxCPM1DiTConfig config;
    config.hidden_dim = base.hidden_dim;
    config.ffn_dim = base.ffn_dim;
    config.num_heads = base.num_heads;
    config.num_layers = base.num_layers;
    config.kv_channels = base.kv_channels;
    config.mean_mode = json::optional_bool(value, "dit_mean_mode", json::optional_bool(value, "mean_mode", false));
    const auto & cfm = value.require("cfm_config");
    config.cfm.sigma_min = json::optional_f32(cfm, "sigma_min", config.cfm.sigma_min);
    config.cfm.solver = json::optional_string(cfm, "solver", config.cfm.solver);
    config.cfm.t_scheduler = json::optional_string(cfm, "t_scheduler", config.cfm.t_scheduler);
    config.cfm.inference_cfg_rate = json::optional_f32(cfm, "inference_cfg_rate", config.cfm.inference_cfg_rate);
    if (config.cfm.solver != "euler") {
        throw std::runtime_error("VoxCPM1 CFM currently expects euler solver");
    }
    if (config.cfm.t_scheduler != "log-norm") {
        throw std::runtime_error("VoxCPM1 CFM currently expects log-norm scheduler");
    }
    return config;
}

VoxCPM1AudioVAEConfig parse_audio_vae_config(const json::Value & value) {
    VoxCPM1AudioVAEConfig config;
    config.encoder_dim = json::require_i64(value, "encoder_dim");
    config.encoder_rates = json::require_i64_array(value, "encoder_rates");
    config.latent_dim = json::require_i64(value, "latent_dim");
    config.decoder_dim = json::require_i64(value, "decoder_dim");
    config.decoder_rates = json::require_i64_array(value, "decoder_rates");
    config.sample_rate_bin_boundaries = json::optional_i64_array(value, "sr_bin_boundaries");
    config.sample_rate = static_cast<int>(json::require_i64(value, "sample_rate"));
    config.output_sample_rate = static_cast<int>(json::require_i64(value, "out_sample_rate"));
    engine::io::require_positive(config.encoder_dim, "AudioVAE encoder_dim");
    engine::io::require_positive(config.latent_dim, "AudioVAE latent_dim");
    engine::io::require_positive(config.decoder_dim, "AudioVAE decoder_dim");
    engine::io::require_positive(config.sample_rate, "AudioVAE sample_rate");
    engine::io::require_positive(config.output_sample_rate, "AudioVAE out_sample_rate");
    if (config.encoder_rates.empty() || config.decoder_rates.empty()) {
        throw std::runtime_error("VoxCPM1 AudioVAE rates must be non-empty");
    }
    for (const auto rate : config.encoder_rates) {
        engine::io::require_positive(rate, "AudioVAE encoder rate");
    }
    for (const auto rate : config.decoder_rates) {
        engine::io::require_positive(rate, "AudioVAE decoder rate");
    }
    return config;
}

std::optional<std::filesystem::path> locate_gguf(const assets::ResourceBundle & resources) {
    try {
        auto source = resources.open_tensor_source("weights");
        if (source) {
            return source->source_path();
        }
    } catch (...) {}
    try {
        auto source = resources.open_tensor_source("audiovae_weights");
        if (source) {
            return source->source_path();
        }
    } catch (...) {}
    const auto root = resources.model_root();
    if (auto found = assets::find_directory_gguf(root)) {
        return *found;
    }
    auto files = assets::directory_gguf_files(root);
    if (!files.empty()) {
        return files.front();
    }
    return std::nullopt;
}

int64_t gguf_get_i64(gguf_context * gguf, const char * key, int64_t default_value) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return default_value;
    const enum gguf_type type = gguf_get_kv_type(gguf, id);
    if (type == GGUF_TYPE_INT32) return gguf_get_val_i32(gguf, id);
    if (type == GGUF_TYPE_INT64) return gguf_get_val_i64(gguf, id);
    if (type == GGUF_TYPE_UINT32) return static_cast<int64_t>(gguf_get_val_u32(gguf, id));
    if (type == GGUF_TYPE_UINT64) return static_cast<int64_t>(gguf_get_val_u64(gguf, id));
    if (type == GGUF_TYPE_FLOAT32) return static_cast<int64_t>(gguf_get_val_f32(gguf, id));
    return default_value;
}

float gguf_get_f32_or(gguf_context * gguf, const char * key, float default_value) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return default_value;
    const enum gguf_type type = gguf_get_kv_type(gguf, id);
    if (type == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(gguf, id);
    if (type == GGUF_TYPE_FLOAT64) return static_cast<float>(gguf_get_val_f64(gguf, id));
    if (type == GGUF_TYPE_INT32) return static_cast<float>(gguf_get_val_i32(gguf, id));
    if (type == GGUF_TYPE_UINT32) return static_cast<float>(gguf_get_val_u32(gguf, id));
    return default_value;
}

std::string gguf_get_str_or(gguf_context * gguf, const char * key, const std::string & default_value) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return default_value;
    if (gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) return default_value;
    return gguf_get_val_str(gguf, id);
}

std::vector<int64_t> gguf_get_i64_array(gguf_context * gguf, const char * key) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return {};
    if (gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY) return {};
    const size_t n = gguf_get_arr_n(gguf, id);
    const enum gguf_type arr_type = gguf_get_arr_type(gguf, id);
    std::vector<int64_t> out;
    out.reserve(n);
    const void * data = gguf_get_arr_data(gguf, id);
    if (arr_type == GGUF_TYPE_INT32) {
        const auto * p = static_cast<const int32_t *>(data);
        for (size_t i = 0; i < n; ++i) out.push_back(p[i]);
    } else if (arr_type == GGUF_TYPE_UINT32) {
        const auto * p = static_cast<const uint32_t *>(data);
        for (size_t i = 0; i < n; ++i) out.push_back(static_cast<int64_t>(p[i]));
    } else if (arr_type == GGUF_TYPE_INT64) {
        const auto * p = static_cast<const int64_t *>(data);
        for (size_t i = 0; i < n; ++i) out.push_back(p[i]);
    } else if (arr_type == GGUF_TYPE_UINT64) {
        const auto * p = static_cast<const uint64_t *>(data);
        for (size_t i = 0; i < n; ++i) out.push_back(static_cast<int64_t>(p[i]));
    }
    return out;
}

std::vector<float> gguf_get_f32_array(gguf_context * gguf, const char * key) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return {};
    if (gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY) return {};
    if (gguf_get_arr_type(gguf, id) != GGUF_TYPE_FLOAT32) return {};
    const size_t n = gguf_get_arr_n(gguf, id);
    const auto * data = static_cast<const float *>(gguf_get_arr_data(gguf, id));
    return std::vector<float>(data, data + n);
}

VoxCPM1Config parse_config_from_gguf(const std::filesystem::path & gguf_path) {
    ggml_context * ctx = nullptr;
    gguf_context * gguf = gguf_init_from_file(gguf_path.string().c_str(), gguf_init_params{true, &ctx});
    if (gguf == nullptr) {
        if (ctx) ggml_free(ctx);
        throw std::runtime_error("failed to read GGUF for VoxCPM1 config: " + gguf_path.string());
    }
    try {
        VoxCPM1Config config;
        config.architecture = gguf_get_str_or(gguf, "voxcpm_architecture", "");
        if (config.architecture.empty()) {
            // Fallback to general.architecture if voxcpm_ missing but file is still voxcpm
            config.architecture = "voxcpm";
        }
        if (config.architecture != "voxcpm") {
            gguf_free(gguf);
            if (ctx) ggml_free(ctx);
            throw std::runtime_error("VoxCPM1 GGUF architecture mismatch: " + config.architecture);
        }
        // LM config
        config.lm.bos_token_id = gguf_get_i64(gguf, "voxcpm_lm_config_bos_token_id", config.lm.bos_token_id);
        config.lm.eos_token_id = gguf_get_i64(gguf, "voxcpm_lm_config_eos_token_id", config.lm.eos_token_id);
        config.lm.hidden_size = gguf_get_i64(gguf, "voxcpm_lm_config_hidden_size", 0);
        config.lm.intermediate_size = gguf_get_i64(gguf, "voxcpm_lm_config_intermediate_size", 0);
        config.lm.max_position_embeddings = gguf_get_i64(gguf, "voxcpm_lm_config_max_position_embeddings", 0);
        config.lm.num_attention_heads = gguf_get_i64(gguf, "voxcpm_lm_config_num_attention_heads", 0);
        config.lm.num_hidden_layers = gguf_get_i64(gguf, "voxcpm_lm_config_num_hidden_layers", 0);
        config.lm.num_key_value_heads = gguf_get_i64(gguf, "voxcpm_lm_config_num_key_value_heads", 0);
        config.lm.vocab_size = gguf_get_i64(gguf, "voxcpm_lm_config_vocab_size", 0);
        config.lm.kv_channels = gguf_get_i64(gguf, "voxcpm_lm_config_kv_channels", config.lm.hidden_size / std::max<int64_t>(config.lm.num_attention_heads, 1));
        config.lm.scale_emb = gguf_get_i64(gguf, "voxcpm_lm_config_scale_emb", config.lm.scale_emb);
        config.lm.dim_model_base = gguf_get_i64(gguf, "voxcpm_lm_config_dim_model_base", config.lm.dim_model_base);
        config.lm.rms_norm_eps = gguf_get_f32_or(gguf, "voxcpm_lm_config_rms_norm_eps", config.lm.rms_norm_eps);
        config.lm.rope_theta = static_cast<float>(gguf_get_i64(gguf, "voxcpm_lm_config_rope_theta", static_cast<int64_t>(config.lm.rope_theta)));
        // rope_theta might be stored as float or int; try both
        if (gguf_find_key(gguf, "voxcpm_lm_config_rope_theta") >= 0 && gguf_get_kv_type(gguf, gguf_find_key(gguf, "voxcpm_lm_config_rope_theta")) == GGUF_TYPE_FLOAT32) {
            config.lm.rope_theta = gguf_get_f32_or(gguf, "voxcpm_lm_config_rope_theta", config.lm.rope_theta);
        }
        config.lm.scale_depth = gguf_get_f32_or(gguf, "voxcpm_lm_config_scale_depth", config.lm.scale_depth);
        config.lm.use_mup = gguf_get_i64(gguf, "voxcpm_lm_config_use_mup", 0) != 0;
        // rope scaling
        const auto long_factor = gguf_get_f32_array(gguf, "voxcpm_lm_config_rope_scaling_long_factor");
        const auto short_factor = gguf_get_f32_array(gguf, "voxcpm_lm_config_rope_scaling_short_factor");
        if (!long_factor.empty() || !short_factor.empty()) {
            config.lm.rope_scaling.type = gguf_get_str_or(gguf, "voxcpm_lm_config_rope_scaling_type", "longrope");
            config.lm.rope_scaling.long_factor = long_factor;
            config.lm.rope_scaling.short_factor = short_factor;
            config.lm.rope_scaling.original_max_position_embeddings = gguf_get_i64(gguf, "voxcpm_lm_config_rope_scaling_original_max_position_embeddings", config.lm.rope_scaling.original_max_position_embeddings);
        }
        // Validate LM
        engine::io::require_positive(config.lm.hidden_size, "lm hidden_size");
        engine::io::require_positive(config.lm.intermediate_size, "lm intermediate_size");
        engine::io::require_positive(config.lm.max_position_embeddings, "lm max_position_embeddings");
        engine::io::require_positive(config.lm.num_attention_heads, "lm num_attention_heads");
        engine::io::require_positive(config.lm.num_hidden_layers, "lm num_hidden_layers");
        engine::io::require_positive(config.lm.num_key_value_heads, "lm num_key_value_heads");
        engine::io::require_positive(config.lm.kv_channels, "lm kv_channels");
        engine::io::require_positive(config.lm.vocab_size, "lm vocab_size");
        // Common
        config.patch_size = gguf_get_i64(gguf, "voxcpm_patch_size", config.patch_size);
        config.feat_dim = gguf_get_i64(gguf, "voxcpm_feat_dim", config.feat_dim);
        config.residual_lm_num_layers = gguf_get_i64(gguf, "voxcpm_residual_lm_num_layers", config.residual_lm_num_layers);
        // residual_lm_no_rope not stored, keep default false
        config.scalar_quantization_latent_dim = gguf_get_i64(gguf, "voxcpm_scalar_quantization_latent_dim", config.scalar_quantization_latent_dim);
        config.scalar_quantization_scale = gguf_get_i64(gguf, "voxcpm_scalar_quantization_scale", config.scalar_quantization_scale);
        // encoder
        config.encoder.hidden_dim = gguf_get_i64(gguf, "voxcpm_encoder_config_hidden_dim", 0);
        config.encoder.ffn_dim = gguf_get_i64(gguf, "voxcpm_encoder_config_ffn_dim", 0);
        config.encoder.num_heads = gguf_get_i64(gguf, "voxcpm_encoder_config_num_heads", 0);
        config.encoder.num_layers = gguf_get_i64(gguf, "voxcpm_encoder_config_num_layers", 0);
        config.encoder.kv_channels = config.encoder.hidden_dim / std::max<int64_t>(config.encoder.num_heads, 1);
        // dit
        config.dit.hidden_dim = gguf_get_i64(gguf, "voxcpm_dit_config_hidden_dim", 0);
        config.dit.ffn_dim = gguf_get_i64(gguf, "voxcpm_dit_config_ffn_dim", 0);
        config.dit.num_heads = gguf_get_i64(gguf, "voxcpm_dit_config_num_heads", 0);
        config.dit.num_layers = gguf_get_i64(gguf, "voxcpm_dit_config_num_layers", 0);
        config.dit.kv_channels = config.dit.hidden_dim / std::max<int64_t>(config.dit.num_heads, 1);
        config.dit.cfm.sigma_min = gguf_get_f32_or(gguf, "voxcpm_dit_config_cfm_config_sigma_min", config.dit.cfm.sigma_min);
        config.dit.cfm.solver = gguf_get_str_or(gguf, "voxcpm_dit_config_cfm_config_solver", config.dit.cfm.solver);
        config.dit.cfm.t_scheduler = gguf_get_str_or(gguf, "voxcpm_dit_config_cfm_config_t_scheduler", config.dit.cfm.t_scheduler);
        config.dit.cfm.inference_cfg_rate = gguf_get_f32_or(gguf, "voxcpm_dit_config_cfm_config_inference_cfg_rate", config.dit.cfm.inference_cfg_rate);
        // audio vae
        config.audio_vae.encoder_dim = gguf_get_i64(gguf, "voxcpm_audio_vae_config_encoder_dim", 0);
        config.audio_vae.encoder_rates = gguf_get_i64_array(gguf, "voxcpm_audio_vae_config_encoder_rates");
        config.audio_vae.latent_dim = gguf_get_i64(gguf, "voxcpm_audio_vae_config_latent_dim", 0);
        config.audio_vae.decoder_dim = gguf_get_i64(gguf, "voxcpm_audio_vae_config_decoder_dim", 0);
        config.audio_vae.decoder_rates = gguf_get_i64_array(gguf, "voxcpm_audio_vae_config_decoder_rates");
        config.audio_vae.sample_rate = static_cast<int>(gguf_get_i64(gguf, "voxcpm_audio_vae_config_sample_rate", 16000));
        config.audio_vae.output_sample_rate = config.audio_vae.sample_rate;
        // Try to get out_sample_rate if present (for 1.5B)
        if (gguf_find_key(gguf, "voxcpm_audio_vae_config_out_sample_rate") >= 0) {
            config.audio_vae.output_sample_rate = static_cast<int>(gguf_get_i64(gguf, "voxcpm_audio_vae_config_out_sample_rate", config.audio_vae.sample_rate));
        }
        // sr_bin_boundaries not in GGUF, keep empty
        config.max_length = gguf_get_i64(gguf, "voxcpm_max_length", config.max_length);
        config.device = gguf_get_str_or(gguf, "voxcpm_device", config.device);
        config.dtype = gguf_get_str_or(gguf, "voxcpm_dtype", config.dtype);
        gguf_free(gguf);
        if (ctx) ggml_free(ctx);
        // Validate remaining
        engine::io::require_positive(config.encoder.hidden_dim, "encoder hidden_dim");
        engine::io::require_positive(config.encoder.ffn_dim, "encoder ffn_dim");
        engine::io::require_positive(config.encoder.num_heads, "encoder num_heads");
        engine::io::require_positive(config.encoder.num_layers, "encoder num_layers");
        engine::io::require_positive(config.dit.hidden_dim, "dit hidden_dim");
        engine::io::require_positive(config.dit.ffn_dim, "dit ffn_dim");
        engine::io::require_positive(config.dit.num_heads, "dit num_heads");
        engine::io::require_positive(config.dit.num_layers, "dit num_layers");
        engine::io::require_positive(config.audio_vae.encoder_dim, "AudioVAE encoder_dim");
        engine::io::require_positive(config.audio_vae.latent_dim, "AudioVAE latent_dim");
        engine::io::require_positive(config.audio_vae.decoder_dim, "AudioVAE decoder_dim");
        if (config.audio_vae.encoder_rates.empty() || config.audio_vae.decoder_rates.empty()) {
            throw std::runtime_error("VoxCPM1 AudioVAE rates must be non-empty from GGUF");
        }
        engine::io::require_positive(config.patch_size, "patch_size");
        engine::io::require_positive(config.feat_dim, "feat_dim");
        engine::io::require_positive(config.residual_lm_num_layers, "residual_lm_num_layers");
        engine::io::require_positive(config.scalar_quantization_latent_dim, "scalar_quantization_latent_dim");
        engine::io::require_positive(config.scalar_quantization_scale, "scalar_quantization_scale");
        engine::io::require_positive(config.max_length, "max_length");
        if (config.feat_dim != config.audio_vae.latent_dim) {
            throw std::runtime_error("VoxCPM1 feat_dim must match AudioVAE latent_dim");
        }
        return config;
    } catch (...) {
        gguf_free(gguf);
        if (ctx) ggml_free(ctx);
        throw;
    }
}

VoxCPM1Config parse_config(const assets::ResourceBundle & resources) {
    if (resources.has_file("config")) {
        const auto root = resources.parse_json("config");
        VoxCPM1Config config;
        config.architecture = json::require_string(root, "architecture");
        if (config.architecture != "voxcpm") {
            throw std::runtime_error("VoxCPM1 config architecture mismatch: " + config.architecture);
        }
        config.lm = parse_lm_config(root.require("lm_config"));
        config.patch_size = json::optional_i64(root, "patch_size", config.patch_size);
        config.feat_dim = json::optional_i64(root, "feat_dim", config.feat_dim);
        config.residual_lm_num_layers =
            json::optional_i64(root, "residual_lm_num_layers", config.residual_lm_num_layers);
        config.residual_lm_no_rope = json::optional_bool(root, "residual_lm_no_rope", config.residual_lm_no_rope);
        config.scalar_quantization_latent_dim =
            json::optional_i64(root, "scalar_quantization_latent_dim", config.scalar_quantization_latent_dim);
        config.scalar_quantization_scale =
            json::optional_i64(root, "scalar_quantization_scale", config.scalar_quantization_scale);
        config.encoder = parse_local_transformer_config(root.require("encoder_config"), "local encoder transformer");
        config.dit = parse_dit_config(root.require("dit_config"));
        config.audio_vae = parse_audio_vae_config(root.require("audio_vae_config"));
        config.max_length = json::optional_i64(root, "max_length", config.max_length);
        config.device = json::optional_string(root, "device", config.device);
        config.dtype = json::optional_string(root, "dtype", config.dtype);
        engine::io::require_positive(config.patch_size, "patch_size");
        engine::io::require_positive(config.feat_dim, "feat_dim");
        engine::io::require_positive(config.residual_lm_num_layers, "residual_lm_num_layers");
        engine::io::require_positive(config.scalar_quantization_latent_dim, "scalar_quantization_latent_dim");
        engine::io::require_positive(config.scalar_quantization_scale, "scalar_quantization_scale");
        engine::io::require_positive(config.max_length, "max_length");
        if (config.feat_dim != config.audio_vae.latent_dim) {
            throw std::runtime_error("VoxCPM1 feat_dim must match AudioVAE latent_dim");
        }
        if (config.residual_lm_num_layers > config.lm.num_hidden_layers) {
            throw std::runtime_error("VoxCPM1 residual_lm_num_layers exceeds lm num_hidden_layers");
        }
        return config;
    }
    auto gguf_path = locate_gguf(resources);
    if (!gguf_path) {
        throw std::runtime_error("VoxCPM1 config requires either config.json or a GGUF with embedded voxcpm config");
    }
    return parse_config_from_gguf(*gguf_path);
}

namespace assets = engine::assets;

void validate_weight_anchors(const VoxCPM1Assets & assets) {
    const auto & config = assets.config;
    const auto & weights = *assets.model_weights;
    assets::require_tensor_shape(weights, "token_embd.weight", {config.lm.vocab_size, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "output_norm.weight", {config.lm.hidden_size});
    assets::require_tensor_shape(weights, "blk.0.attn_q.weight", {config.lm.hidden_size, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "blk.0.attn_k.weight",
        {config.lm.num_key_value_heads * config.lm.kv_channels, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "blk.0.ffn_gate.weight", {config.lm.intermediate_size, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "residual_lm.output_norm.weight", {config.lm.hidden_size});
    {
        const auto meta = weights.require_metadata("locenc.special_token");
        int64_t expected = config.encoder.hidden_dim;
        int64_t actual = 1;
        for (auto d : meta.shape) actual *= d;
        if (actual != expected) {
            throw std::runtime_error("tensor shape mismatch for locenc.special_token: expected " +
                                     std::to_string(expected) + " elements, got " + std::to_string(actual));
        }
    }
    assets::require_tensor_shape(weights, "locenc.in_proj.weight", {config.encoder.hidden_dim, config.feat_dim});
    assets::require_tensor_shape(weights, "locenc.output_norm.weight", {config.encoder.hidden_dim});
    assets::require_tensor_shape(weights, "locdit.in_proj.weight", {config.dit.hidden_dim, config.feat_dim});
    assets::require_tensor_shape(weights, "locdit.cond_proj.weight", {config.dit.hidden_dim, config.feat_dim});
    assets::require_tensor_shape(weights, "locdit.out_proj.weight", {config.feat_dim, config.dit.hidden_dim});
    assets::require_tensor_shape(weights, "locdit.output_norm.weight", {config.dit.hidden_dim});
    assets::require_tensor_shape(weights, "fsq.in_proj.weight", {config.scalar_quantization_latent_dim, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "fsq.out_proj.weight", {config.lm.hidden_size, config.scalar_quantization_latent_dim});
    assets::require_tensor_shape(weights, "proj.enc_to_lm.weight", {config.lm.hidden_size, config.encoder.hidden_dim});
    assets::require_tensor_shape(weights, "proj.lm_to_dit.weight", {config.dit.hidden_dim, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "proj.res_to_dit.weight", {config.dit.hidden_dim, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "stop.stop_proj.weight", {config.lm.hidden_size, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "stop.stop_head.weight", {2, config.lm.hidden_size});

    const auto & vae = *assets.audiovae_weights;
    int64_t encoder_in_channels = config.audio_vae.encoder_dim;
    for (size_t i = 0; i < config.audio_vae.encoder_rates.size(); ++i) {
        encoder_in_channels *= 2;
    }
    assets::require_tensor_shape(vae, "audio_vae.encoder.fc_mu.weight", {config.audio_vae.latent_dim, encoder_in_channels, 3});
    assets::require_tensor_shape(vae, "audio_vae.encoder.fc_mu.bias", {config.audio_vae.latent_dim});
    assets::require_tensor_shape(vae, "audio_vae.decoder.model.0.weight", {config.audio_vae.latent_dim, 1, 7});
    assets::require_tensor_shape(vae, "audio_vae.decoder.model.1.weight", {config.audio_vae.decoder_dim, config.audio_vae.latent_dim, 1});
}

}

std::shared_ptr<const VoxCPM1Assets> load_voxcpm1_assets(const std::filesystem::path & model_path) {
    auto out = std::make_shared<VoxCPM1Assets>();
    try {
        out->resources = engine::model_spec::load_resource_bundle(
            model_path,
            engine::model_spec::default_spec_path("voxcpm1"));
    } catch (const std::exception & e) {
        const std::string msg = e.what();
        const bool missing_tokenizer =
            msg.find("tokenizer_config") != std::string::npos ||
            msg.find("tokenizer_json") != std::string::npos ||
            msg.find("config.json") != std::string::npos;
        if (!missing_tokenizer) {
            throw;
        }
        // Standalone GGUF without sidecars: build a minimal bundle that
        // provides the GGUF tensors and falls back to GGUF-embedded config/tokenizer
        auto prepared = engine::assets::prepare_model_directory(model_path);
        if (!prepared.standalone_gguf) {
            throw;
        }
        assets::ResourceBundle fallback(prepared.model_root);
        fallback.add_tensor_source("weights", *prepared.standalone_gguf, "");
        fallback.add_tensor_source("audiovae_weights", *prepared.standalone_gguf, "");
        auto try_add = [&](const char * id, const char * rel) {
            const auto p = prepared.model_root / rel;
            if (engine::io::is_existing_file(p)) {
                fallback.add_file(id, p);
            }
        };
        try_add("config", "config.json");
        try_add("tokenizer_json", "tokenizer.json");
        try_add("tokenizer_config", "tokenizer_config.json");
        try_add("special_tokens_map", "special_tokens_map.json");
        out->resources = std::move(fallback);
    }
    out->config = parse_config(out->resources);
    out->config.v1 = true;
    out->model_weights = out->resources.open_tensor_source("weights");
    out->audiovae_weights = out->resources.open_tensor_source("audiovae_weights");
    validate_weight_anchors(*out);
    return out;
}

}  // namespace engine::community_models::voxcpm1
