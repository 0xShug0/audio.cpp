#include "engine/models/voxcpm2/config_gguf.h"

#include "engine/framework/assets/tensor_source.h"

#include <stdexcept>
#include <string>

namespace engine::models::voxcpm2 {

bool has_voxcpm1_config_metadata(const engine::assets::TensorSource & source) {
    // Check for at least one VoxCPM1-specific metadata key
    return source.optional_string("voxcpm_architecture").has_value() ||
           source.optional_string("voxcpm_lm_config_hidden_size").has_value() ||
           source.optional_u32("voxcpm_lm_config_hidden_size").has_value();
}

VoxCPM2Config load_voxcpm1_config_from_gguf(const engine::assets::TensorSource & source) {
    VoxCPM2Config config;
    config.v1 = true;
    config.architecture = "voxcpm";

    // Helper lambda to get optional i64 from GGUF metadata (via u32 or i64)
    auto get_optional_i64 = [&source](const char * key) -> std::optional<int64_t> {
        auto u32 = source.optional_u32(key);
        if (u32) return static_cast<int64_t>(*u32);
        // Try i64 scalar if it's a tensor
        if (source.has_tensor(key)) {
            try {
                return source.require_i64_scalar(key);
            } catch (...) {
                // Not a scalar tensor
            }
        }
        return std::nullopt;
    };

    // Helper lambda to get optional bool from GGUF metadata
    auto get_optional_bool = [&source](const char * key) -> std::optional<bool> {
        auto u32 = source.optional_u32(key);
        if (u32) return *u32 != 0;
        return std::nullopt;
    };

    // Helper lambda to get optional int64 array from GGUF metadata
    auto get_optional_i64_array = [&source](const char * key) -> std::optional<std::vector<int64_t>> {
        auto i32_arr = source.optional_i32_array(key);
        if (i32_arr) {
            std::vector<int64_t> result;
            result.reserve(i32_arr->size());
            for (int32_t v : *i32_arr) {
                result.push_back(static_cast<int64_t>(v));
            }
            return result;
        }
        return std::nullopt;
    };

    // Architecture
    auto arch = source.optional_string("voxcpm_architecture");
    if (arch) config.architecture = *arch;

    // LM Config
    config.lm.bos_token_id = get_optional_i64("voxcpm_lm_config_bos_token_id").value_or(1);
    config.lm.eos_token_id = get_optional_i64("voxcpm_lm_config_eos_token_id").value_or(2);
    config.lm.hidden_size = get_optional_i64("voxcpm_lm_config_hidden_size").value_or(1024);
    config.lm.intermediate_size = get_optional_i64("voxcpm_lm_config_intermediate_size").value_or(4096);
    config.lm.max_position_embeddings = get_optional_i64("voxcpm_lm_config_max_position_embeddings").value_or(2048);
    config.lm.num_attention_heads = get_optional_i64("voxcpm_lm_config_num_attention_heads").value_or(16);
    config.lm.num_hidden_layers = get_optional_i64("voxcpm_lm_config_num_hidden_layers").value_or(24);
    config.lm.num_key_value_heads = get_optional_i64("voxcpm_lm_config_num_key_value_heads").value_or(16);
    config.lm.kv_channels = get_optional_i64("voxcpm_lm_config_kv_channels").value_or(config.lm.hidden_size / config.lm.num_attention_heads);
    config.lm.vocab_size = get_optional_i64("voxcpm_lm_config_vocab_size").value_or(73448);
    config.lm.scale_emb = get_optional_i64("voxcpm_lm_config_scale_emb").value_or(1);
    config.lm.dim_model_base = get_optional_i64("voxcpm_lm_config_dim_model_base").value_or(256);
    config.lm.rms_norm_eps = 1e-5f;  // Default, GGUF doesn't have native float
    config.lm.rope_theta = 10000.0f;  // Default
    config.lm.scale_depth = 1.0f;  // Default
    config.lm.use_mup = get_optional_bool("voxcpm_lm_config_use_mup").value_or(false);

    // Rope scaling (longrope for VoxCPM1)
    config.lm.rope_scaling.type = "longrope";
    // GGUF doesn't have native float arrays, use defaults
    const int64_t head_dim = config.lm.hidden_size / config.lm.num_attention_heads;
    const int64_t factor_size = head_dim / 2;
    config.lm.rope_scaling.long_factor.assign(factor_size, 1.0f);
    config.lm.rope_scaling.short_factor.assign(factor_size, 1.0f);
    config.lm.rope_scaling.original_max_position_embeddings = 
        get_optional_i64("voxcpm_lm_config_rope_scaling_original_max_position_embeddings").value_or(2048);

    // Patch size
    config.patch_size = get_optional_i64("voxcpm_patch_size").value_or(1);

    // Feature dimension
    config.feat_dim = get_optional_i64("voxcpm_feat_dim").value_or(512);

    // Residual LM
    config.residual_lm_num_layers = get_optional_i64("voxcpm_residual_lm_num_layers").value_or(6);
    config.residual_lm_no_rope = get_optional_bool("voxcpm_residual_lm_no_rope").value_or(false);

    // Scalar quantization
    config.scalar_quantization_latent_dim = get_optional_i64("voxcpm_scalar_quantization_latent_dim").value_or(8);
    config.scalar_quantization_scale = get_optional_i64("voxcpm_scalar_quantization_scale").value_or(8);

    // Encoder config (local encoder)
    config.encoder.hidden_dim = get_optional_i64("voxcpm_encoder_config_hidden_dim").value_or(512);
    config.encoder.ffn_dim = get_optional_i64("voxcpm_encoder_config_ffn_dim").value_or(2048);
    config.encoder.num_heads = get_optional_i64("voxcpm_encoder_config_num_heads").value_or(8);
    config.encoder.num_layers = get_optional_i64("voxcpm_encoder_config_num_layers").value_or(4);
    config.encoder.kv_channels = get_optional_i64("voxcpm_encoder_config_kv_channels").value_or(config.encoder.hidden_dim / config.encoder.num_heads);

    // DiT config (local DiT)
    config.dit.hidden_dim = get_optional_i64("voxcpm_dit_config_hidden_dim").value_or(512);
    config.dit.ffn_dim = get_optional_i64("voxcpm_dit_config_ffn_dim").value_or(2048);
    config.dit.num_heads = get_optional_i64("voxcpm_dit_config_num_heads").value_or(8);
    config.dit.num_layers = get_optional_i64("voxcpm_dit_config_num_layers").value_or(4);
    config.dit.kv_channels = get_optional_i64("voxcpm_dit_config_kv_channels").value_or(config.dit.hidden_dim / config.dit.num_heads);
    config.dit.mean_mode = get_optional_bool("voxcpm_dit_config_mean_mode").value_or(false);
    config.dit.cfm.sigma_min = 1e-4f;  // Default
    config.dit.cfm.solver = "euler";
    config.dit.cfm.t_scheduler = "log-norm";
    config.dit.cfm.inference_cfg_rate = 0.5f;  // Default

    // Audio VAE config
    config.audio_vae.encoder_dim = get_optional_i64("voxcpm_audio_vae_config_encoder_dim").value_or(64);
    config.audio_vae.encoder_rates = get_optional_i64_array("voxcpm_audio_vae_config_encoder_rates").value_or(std::vector<int64_t>{2, 2, 2, 2});
    config.audio_vae.latent_dim = get_optional_i64("voxcpm_audio_vae_config_latent_dim").value_or(512);
    config.audio_vae.decoder_dim = get_optional_i64("voxcpm_audio_vae_config_decoder_dim").value_or(512);
    config.audio_vae.decoder_rates = get_optional_i64_array("voxcpm_audio_vae_config_decoder_rates").value_or(std::vector<int64_t>{2, 2, 2, 2});
    config.audio_vae.sample_rate_bin_boundaries = get_optional_i64_array("voxcpm_audio_vae_config_sr_bin_boundaries").value_or(std::vector<int64_t>{});
    config.audio_vae.sample_rate = static_cast<int>(get_optional_i64("voxcpm_audio_vae_config_sample_rate").value_or(16000));
    config.audio_vae.output_sample_rate = static_cast<int>(get_optional_i64("voxcpm_audio_vae_config_out_sample_rate").value_or(16000));

    // Max length
    config.max_length = get_optional_i64("voxcpm_max_length").value_or(2048);

    // Device and dtype
    config.device = source.optional_string("voxcpm_device").value_or("cpu");
    config.dtype = source.optional_string("voxcpm_dtype").value_or("fp16");

    // Validate required fields
    if (config.lm.hidden_size <= 0) {
        throw std::runtime_error("voxcpm_lm_config_hidden_size must be positive");
    }
    if (config.lm.vocab_size <= 0) {
        throw std::runtime_error("voxcpm_lm_config_vocab_size must be positive");
    }
    if (config.feat_dim != config.audio_vae.latent_dim) {
        throw std::runtime_error("voxcpm_feat_dim must match voxcpm_audio_vae_config_latent_dim");
    }
    if (config.residual_lm_num_layers > config.lm.num_hidden_layers) {
        throw std::runtime_error("residual_lm_num_layers exceeds lm num_hidden_layers");
    }

    return config;
}

}  // namespace engine::models::voxcpm2