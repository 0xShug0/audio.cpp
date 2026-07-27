#include "engine/models/dramabox/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>

namespace engine::models::dramabox {
namespace json = engine::io::json;
namespace {

bool has_prefix(const assets::TensorSource & source, std::string_view prefix) {
    const auto tensors = source.tensors();
    return std::any_of(tensors.begin(), tensors.end(), [&](const auto & item) {
        return item.name.rfind(prefix, 0) == 0;
    });
}

void require_prefix(const assets::TensorSource & source, std::string_view prefix, const char * label) {
    if (!has_prefix(source, prefix)) {
        throw std::runtime_error(
            std::string("DramaBox tensor source missing ") + label + " tensors with prefix " + std::string(prefix));
    }
}

std::vector<int64_t> optional_i64_array_or_empty(const json::Value & object, const std::string & key) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return {};
    }
    return json::number_array_as<int64_t>(*value);
}

DramaBoxConfig parse_dramabox_config(const assets::ResourceBundle & resources) {
    DramaBoxConfig config;
    const auto top = resources.parse_json("config");
    config.model_type = json::require_string(top, "model_type");
    if (config.model_type != "dramabox-tts") {
        throw std::runtime_error("unsupported DramaBox model_type: " + config.model_type);
    }

    const auto root = resources.parse_json("audio_components_config");
    const auto & transformer = root.require("transformer");
    config.transformer.num_layers = json::require_i64(transformer, "num_layers");
    config.transformer.in_channels = json::optional_i64(transformer, "audio_in_channels", 128);
    config.transformer.out_channels = json::require_i64(transformer, "audio_out_channels");
    config.transformer.num_attention_heads = json::require_i64(transformer, "audio_num_attention_heads");
    config.transformer.attention_head_dim = json::require_i64(transformer, "audio_attention_head_dim");
    config.transformer.hidden_size = config.transformer.num_attention_heads * config.transformer.attention_head_dim;
    config.transformer.cross_attention_dim = json::require_i64(transformer, "audio_cross_attention_dim");
    config.transformer.timestep_scale_multiplier = json::require_i64(transformer, "timestep_scale_multiplier");
    config.transformer.positional_embedding_theta = json::require_f32(transformer, "positional_embedding_theta");
    config.transformer.positional_embedding_max_pos =
        json::require_i64_array(transformer, "audio_positional_embedding_max_pos");
    config.transformer.connector_num_layers = json::require_i64(transformer, "connector_num_layers");
    config.transformer.connector_num_attention_heads = json::require_i64(transformer, "audio_connector_num_attention_heads");
    config.transformer.connector_attention_head_dim = json::require_i64(transformer, "audio_connector_attention_head_dim");
    config.transformer.connector_positional_embedding_max_pos =
        json::require_i64_array(transformer, "connector_positional_embedding_max_pos");
    config.transformer.connector_num_learnable_registers =
        json::require_i64(transformer, "connector_num_learnable_registers");
    config.transformer.apply_gated_attention = json::require_bool(transformer, "apply_gated_attention");
    config.transformer.connector_apply_gated_attention =
        json::require_bool(transformer, "connector_apply_gated_attention");
    config.transformer.cross_attention_adaln = json::require_bool(transformer, "cross_attention_adaln");
    config.transformer.use_middle_indices_grid = json::require_bool(transformer, "use_middle_indices_grid");
    config.transformer.rope_type = json::require_string(transformer, "rope_type");
    config.transformer.double_precision_rope =
        json::require_string(transformer, "frequencies_precision") == "float64";

    const auto & audio_vae = root.require("audio_vae");
    const auto & ddconfig = audio_vae.require("model").require("params").require("ddconfig");
    const auto & stft = audio_vae.require("preprocessing").require("stft");
    config.audio_vae.latent_channels = json::require_i64(ddconfig, "z_channels");
    config.audio_vae.mel_bins = json::require_i64(ddconfig, "mel_bins");
    config.audio_vae.ch = json::require_i64(ddconfig, "ch");
    config.audio_vae.ch_mult = json::require_i64_array(ddconfig, "ch_mult");
    config.audio_vae.num_res_blocks = json::require_i64(ddconfig, "num_res_blocks");
    config.audio_vae.attn_resolutions = optional_i64_array_or_empty(ddconfig, "attn_resolutions");
    config.audio_vae.resolution = json::require_i64(ddconfig, "resolution");
    config.audio_vae.out_channels = json::require_i64(ddconfig, "out_ch");
    config.audio_vae.sample_rate = json::require_i64(audio_vae.require("model").require("params"), "sampling_rate");
    config.audio_vae.hop_length = json::require_i64(stft, "hop_length");
    config.audio_vae.n_fft = json::require_i64(stft, "filter_length");
    config.audio_vae.causal = json::require_bool(stft, "causal");
    config.audio_vae.norm_type = json::require_string(ddconfig, "norm_type");
    config.audio_vae.causality_axis = json::require_string(ddconfig, "causality_axis");

    const auto & vocoder = root.require("vocoder");
    const auto & vocoder_cfg = vocoder.require("vocoder");
    const auto & bwe_cfg = vocoder.require("bwe");
    config.vocoder.input_sample_rate = json::require_i64(bwe_cfg, "input_sampling_rate");
    config.vocoder.output_sample_rate = json::require_i64(bwe_cfg, "output_sampling_rate");
    config.vocoder.hop_length = json::require_i64(bwe_cfg, "hop_length");
    config.vocoder.n_fft = json::require_i64(bwe_cfg, "n_fft");
    config.vocoder.num_mels = json::require_i64(bwe_cfg, "num_mels");
    config.vocoder.vocoder_initial_channel = json::require_i64(vocoder_cfg, "upsample_initial_channel");
    config.vocoder.bwe_initial_channel = json::require_i64(bwe_cfg, "upsample_initial_channel");
    config.vocoder.vocoder_upsample_rates = json::require_i64_array(vocoder_cfg, "upsample_rates");
    config.vocoder.vocoder_upsample_kernel_sizes = json::require_i64_array(vocoder_cfg, "upsample_kernel_sizes");
    config.vocoder.bwe_upsample_rates = json::require_i64_array(bwe_cfg, "upsample_rates");
    config.vocoder.bwe_upsample_kernel_sizes = json::require_i64_array(bwe_cfg, "upsample_kernel_sizes");
    config.vocoder.resblock_kernel_sizes = json::require_i64_array(vocoder_cfg, "resblock_kernel_sizes");

    const auto gemma_root = resources.parse_json("gemma_config");
    const auto & text_config = gemma_root.require("text_config");
    config.gemma.hidden_size = json::require_i64(text_config, "hidden_size");
    config.gemma.intermediate_size = json::require_i64(text_config, "intermediate_size");
    config.gemma.num_hidden_layers = json::require_i64(text_config, "num_hidden_layers");
    config.gemma.num_attention_heads = json::require_i64(text_config, "num_attention_heads");
    config.gemma.num_key_value_heads = json::require_i64(text_config, "num_key_value_heads");
    config.gemma.head_dim = json::require_i64(text_config, "head_dim");
    config.gemma.vocab_size = json::require_i64(text_config, "vocab_size");
    config.gemma.max_position_embeddings = json::require_i64(text_config, "max_position_embeddings");
    config.gemma.sliding_window = json::require_i64(text_config, "sliding_window");
    config.gemma.sliding_window_pattern = json::require_i64(text_config, "sliding_window_pattern");
    config.gemma.rope_theta = json::require_f32(text_config, "rope_theta");
    config.gemma.rope_local_base_freq = json::optional_f32(text_config, "rope_local_base_freq", 10000.0F);
    config.gemma.rms_norm_eps = json::require_f32(text_config, "rms_norm_eps");
    config.gemma.query_pre_attn_scalar = json::require_f32(text_config, "query_pre_attn_scalar");
    config.gemma.rope_scaling_factor = json::require_f32(text_config.require("rope_scaling"), "factor");
    return config;
}

void validate_weight_anchors(const DramaBoxAssets & assets) {
    require_prefix(*assets.dit_weights, "model.diffusion_model.audio_patchify_proj.", "DiT audio input projection");
    require_prefix(*assets.dit_weights, "model.diffusion_model.audio_adaln_single.", "DiT AdaLN");
    require_prefix(*assets.dit_weights, "model.diffusion_model.transformer_blocks.", "DiT transformer blocks");
    require_prefix(*assets.dit_weights, "model.diffusion_model.audio_proj_out.", "DiT audio output projection");
    require_prefix(*assets.audio_weights, "text_embedding_projection.audio_aggregate_embed.", "audio text projection");
    require_prefix(*assets.audio_weights, "model.diffusion_model.audio_embeddings_connector.", "audio embeddings connector");
    require_prefix(*assets.audio_weights, "audio_vae.decoder.", "audio VAE decoder");
    require_prefix(*assets.audio_weights, "audio_vae.encoder.", "audio VAE encoder");
    require_prefix(*assets.audio_weights, "vocoder.vocoder.", "vocoder");
    require_prefix(*assets.audio_weights, "vocoder.bwe_generator.", "BWE vocoder");
    require_prefix(*assets.audio_weights, "vocoder.mel_stft.", "BWE mel STFT");
    require_prefix(*assets.silence_latent, "silence_latent_frame", "silence latent");
    require_prefix(*assets.gemma_weights.front(), "language_model.model.", "Gemma language model");
    assets::require_tensor_shape(
        *assets.gemma_weights.front(),
        "language_model.model.layers.0.self_attn.q_proj.weight",
        {assets.config.gemma.num_attention_heads * assets.config.gemma.head_dim, assets.config.gemma.hidden_size});
}

}  // namespace

std::shared_ptr<const DramaBoxAssets> load_dramabox_assets(const std::filesystem::path & model_path) {
    DramaBoxAssets assets;
    assets.resources = model_spec::load_resource_bundle(
        model_path,
        model_spec::default_spec_path("dramabox"));
    assets.config = parse_dramabox_config(assets.resources);
    assets.dit_weights = assets.resources.open_tensor_source("dit_weights");
    assets.audio_weights = assets.resources.open_tensor_source("audio_weights");
    assets.silence_latent = assets.resources.open_tensor_source("silence_latent");
    assets.gemma_weights.push_back(assets.resources.open_tensor_source("gemma_weights"));
    validate_weight_anchors(assets);
    return std::make_shared<DramaBoxAssets>(std::move(assets));
}

}  // namespace engine::models::dramabox
