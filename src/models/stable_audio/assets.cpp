#include "engine/models/stable_audio/assets.h"

#include "engine/framework/assets/model_package.h"
#include "engine/framework/io/json.h"

#include <stdexcept>

namespace engine::models::stable_audio {
namespace json = engine::io::json;
namespace {

StableAudioConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("model_config");
    StableAudioConfig config;
    config.sample_size = json::require_i64(root, "sample_size");
    config.sample_rate = json::require_i32(root, "sample_rate");
    config.audio_channels = json::require_i32(root, "audio_channels");

    const auto & model = root.require("model");
    config.io_channels = json::require_i64(model, "io_channels");

    const auto & pretransform = model.require("pretransform");
    config.pretransform_type = json::require_string(pretransform, "type");
    const auto & pretransform_config = pretransform.require("config");
    config.latent_dim = json::require_i64(pretransform_config, "latent_dim");
    config.downsampling_ratio = json::require_i64(pretransform_config, "downsampling_ratio");
    const auto & encoder_config = pretransform_config.require("encoder").require("config");
    const auto & decoder_config = pretransform_config.require("decoder").require("config");
    const bool has_sa3_pretransform = pretransform_config.find("pretransform") != nullptr;
    config.stable_audio_open_v1 = !has_sa3_pretransform;
    if (has_sa3_pretransform) {
        config.pretransform_patch_size =
            json::require_i64(pretransform_config.require("pretransform").require("config"), "patch_size");
    } else {
        config.pretransform_patch_size = 1;
    }
    config.pretransform_encoder_chunk_size = json::optional_i64(encoder_config, "chunk_size", 32);
    const auto & encoder_strides = encoder_config.require("strides").as_array();
    if (encoder_strides.empty()) {
        throw std::runtime_error("Stable Audio pretransform encoder strides must not be empty");
    }
    config.pretransform_encoder_stride = encoder_strides.front().as_i64();
    config.same_encoder_in_channels = json::require_i64(encoder_config, "in_channels");
    config.same_decoder_out_channels = json::require_i64(decoder_config, "out_channels");
    config.same_channels = json::require_i64(encoder_config, "channels");
    if (config.same_channels != json::require_i64(decoder_config, "channels")) {
        throw std::runtime_error("Stable Audio SAME encoder/decoder channel counts must match");
    }
    config.same_dim_heads = json::optional_i64(encoder_config, "dim_heads", 64);
    if (config.same_dim_heads != json::optional_i64(decoder_config, "dim_heads", config.same_dim_heads)) {
        throw std::runtime_error("Stable Audio SAME encoder/decoder head dimensions must match");
    }
    config.same_c_mults = json::require_i64_array(encoder_config, "c_mults");
    config.same_strides = json::require_i64_array(encoder_config, "strides");
    config.same_encoder_transformer_depths = has_sa3_pretransform
        ? json::require_i64_array(encoder_config, "transformer_depths")
        : std::vector<int64_t>(config.same_c_mults.size(), 0);
    config.same_decoder_transformer_depths = has_sa3_pretransform
        ? json::require_i64_array(decoder_config, "transformer_depths")
        : std::vector<int64_t>(config.same_c_mults.size(), 0);
    config.same_decoder_sinusoidal_blocks = json::optional_i64_array(decoder_config, "sinusoidal_blocks", std::vector<int64_t>(config.same_decoder_transformer_depths.size(), 0));
    config.same_sliding_window = json::optional_i64_array(encoder_config, "sliding_window");
    config.same_differential = json::optional_bool(encoder_config, "differential", true);
    if (config.same_differential != json::optional_bool(decoder_config, "differential", config.same_differential)) {
        throw std::runtime_error("Stable Audio SAME encoder/decoder differential settings must match");
    }
    config.same_variable_stride = json::optional_bool(encoder_config, "variable_stride", false);
    if (config.same_variable_stride != json::optional_bool(decoder_config, "variable_stride", config.same_variable_stride)) {
        throw std::runtime_error("Stable Audio SAME encoder/decoder variable_stride settings must match");
    }
    config.same_encoder_conv_mapping = json::optional_bool(encoder_config, "conv_mapping", false);
    config.same_decoder_conv_mapping = json::optional_bool(decoder_config, "conv_mapping", false);
    config.same_chunk_midpoint_shift =
        json::optional_bool(encoder_config, "chunk_midpoint_shift", false) ||
        json::optional_bool(decoder_config, "chunk_midpoint_shift", false);
    config.same_encoder_mask_noise = json::optional_f32(encoder_config, "mask_noise", 0.0F);
    config.same_decoder_mask_noise = json::optional_f32(decoder_config, "mask_noise", 0.0F);
    const auto & bottleneck = pretransform_config.require("bottleneck");
    if (const auto * bottleneck_config = bottleneck.find("config"); bottleneck_config != nullptr && bottleneck_config->is_object()) {
        config.same_bottleneck_noise_regularize = json::optional_bool(*bottleneck_config, "noise_regularize", false);
        config.same_bottleneck_auto_scale = json::optional_bool(*bottleneck_config, "auto_scale", false);
        config.same_bottleneck_noise_augment_dim = json::optional_i64(*bottleneck_config, "noise_augment_dim", 0);
    }

    const auto & conditioning = model.require("conditioning");
    config.cond_dim = json::require_i64(conditioning, "cond_dim");
    for (const auto & item : conditioning.require("configs").as_array()) {
        const std::string id = json::require_string(item, "id");
        const std::string type = json::require_string(item, "type");
        const auto & item_config = item.require("config");
        if (id == "prompt") {
            config.prompt_conditioner_type = type;
            config.prompt_max_length = json::require_i64(item_config, "max_length");
        } else if (id == "seconds_total" && type == "number") {
            config.seconds_min = json::require_f32(item_config, "min_val");
            config.seconds_max = json::require_f32(item_config, "max_val");
        }
    }
    if (config.prompt_conditioner_type.empty()) {
        throw std::runtime_error("Stable Audio config is missing prompt conditioner");
    }
    if (config.prompt_conditioner_type != "t5gemma" && config.prompt_conditioner_type != "t5") {
        throw std::runtime_error("Stable Audio unsupported prompt conditioner type: " + config.prompt_conditioner_type);
    }

    const auto & diffusion = model.require("diffusion");
    config.diffusion_type = json::require_string(diffusion, "type");
    config.diffusion_objective = json::optional_string(diffusion, "diffusion_objective", "v");
    if (const auto * shift = diffusion.find("sampling_distribution_shift_options"); shift != nullptr && !shift->is_null()) {
        config.distribution_shift_type = json::optional_string(*shift, "type", "full");
        config.distribution_shift_base_shift = json::optional_f32(*shift, "base_shift", 0.5F);
        config.distribution_shift_max_shift = json::optional_f32(*shift, "max_shift", 1.15F);
        config.distribution_shift_min_length = json::optional_i64(*shift, "min_length", 256);
        config.distribution_shift_max_length = json::optional_i64(*shift, "max_length", 4096);
        config.distribution_shift_use_sine = json::optional_bool(*shift, "use_sine", false);
    }
    const auto & diffusion_config = diffusion.require("config");
    config.diffusion_io_channels = json::require_i64(diffusion_config, "io_channels");
    config.embed_dim = json::require_i64(diffusion_config, "embed_dim");
    config.depth = json::require_i64(diffusion_config, "depth");
    config.num_heads = json::require_i64(diffusion_config, "num_heads");
    config.cond_token_dim = json::require_i64(diffusion_config, "cond_token_dim");
    config.global_cond_dim = json::require_i64(diffusion_config, "global_cond_dim");
    config.local_add_cond_dim = json::optional_i64(diffusion_config, "local_add_cond_dim", 0);
    config.num_memory_tokens = json::optional_i64(diffusion_config, "num_memory_tokens", 0);
    config.transformer_type = json::optional_string(diffusion_config, "transformer_type", "");
    if (const auto * norm_kwargs = diffusion_config.find("norm_kwargs"); norm_kwargs != nullptr && norm_kwargs->is_object()) {
        config.force_fp32_norm = json::optional_bool(*norm_kwargs, "force_fp32", false);
    }
    if (const auto * attn_kwargs = diffusion_config.find("attn_kwargs"); attn_kwargs != nullptr && attn_kwargs->is_object()) {
        config.differential_attention = json::optional_bool(*attn_kwargs, "differential", false);
    }
    config.medium_architecture = config.embed_dim >= 1536 || config.depth >= 24;
    return config;
}

void parse_t5gemma_config(const assets::ResourceBundle & resources, StableAudioConfig & config) {
    const auto root = resources.parse_json("t5_config");
    const auto & encoder = root.require("encoder");
    config.t5_hidden_size = json::require_i64(encoder, "hidden_size");
    config.t5_layers = json::require_i64(encoder, "num_hidden_layers");
    config.t5_attention_heads = json::require_i64(encoder, "num_attention_heads");
    config.t5_kv_heads = json::require_i64(encoder, "num_key_value_heads");
    config.t5_head_dim = json::require_i64(encoder, "head_dim");
    config.t5_intermediate_size = json::require_i64(encoder, "intermediate_size");
    config.t5_vocab_size = json::require_i64(encoder, "vocab_size");
    config.t5_sliding_window = json::require_i64(encoder, "sliding_window");
    config.t5_rope_theta = json::require_f32(encoder, "rope_theta");
    config.t5_rms_norm_eps = json::require_f32(encoder, "rms_norm_eps");
    config.t5_attn_logit_softcap = json::require_f32(encoder, "attn_logit_softcapping");
    config.t5_query_pre_attn_scalar = json::require_f32(encoder, "query_pre_attn_scalar");
    config.t5_pad_token_id = json::require_i64(root, "pad_token_id");
}

void parse_t5_base_config(const assets::ResourceBundle & resources, StableAudioConfig & config) {
    const auto root = resources.parse_json("t5_config");
    config.t5_hidden_size = json::require_i64(root, "d_model");
    config.t5_layers = json::require_i64(root, "num_layers");
    config.t5_attention_heads = json::require_i64(root, "num_heads");
    config.t5_kv_heads = config.t5_attention_heads;
    config.t5_head_dim = json::require_i64(root, "d_kv");
    config.t5_intermediate_size = json::require_i64(root, "d_ff");
    config.t5_vocab_size = json::require_i64(root, "vocab_size");
    config.t5_rms_norm_eps = json::require_f32(root, "layer_norm_epsilon");
    config.t5_pad_token_id = json::require_i64(root, "pad_token_id");
}

}  // namespace

std::shared_ptr<const StableAudioAssets> load_stable_audio_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<StableAudioAssets>();
    assets->resources = assets::load_resource_bundle_from_package_spec(
        model_path,
        assets::default_model_package_spec_path("stable_audio"));
    assets->t5_tokenizer_model_path = assets->resources.require_file("t5_tokenizer_model");
    assets->config = parse_config(assets->resources);
    if (assets->config.prompt_conditioner_type == "t5gemma") {
        parse_t5gemma_config(assets->resources, assets->config);
    } else {
        parse_t5_base_config(assets->resources, assets->config);
    }
    assets->model_weights = assets->resources.open_tensor_source("model_weights");
    assets->t5_weights = assets->resources.open_tensor_source("t5_weights");
    return assets;
}

}  // namespace engine::models::stable_audio
