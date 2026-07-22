#include "engine/models/ace_step/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <string_view>
#include <utility>

namespace engine::models::ace_step {
namespace json = engine::io::json;
namespace {

void parse_qwen_common(
    const engine::io::json::Value & value,
    int64_t & vocab_size,
    int64_t & hidden_size,
    int64_t & intermediate_size,
    int64_t & num_hidden_layers,
    int64_t & num_attention_heads,
    int64_t & num_key_value_heads,
    int64_t & head_dim,
    int64_t & max_position_embeddings,
    float & rms_norm_eps,
    float & rope_theta) {
    vocab_size = json::require_i64(value, "vocab_size");
    hidden_size = json::require_i64(value, "hidden_size");
    intermediate_size = json::require_i64(value, "intermediate_size");
    num_hidden_layers = json::require_i64(value, "num_hidden_layers");
    num_attention_heads = json::require_i64(value, "num_attention_heads");
    num_key_value_heads = json::require_i64(value, "num_key_value_heads");
    head_dim = json::optional_i64(value, "head_dim", hidden_size / num_attention_heads);
    max_position_embeddings = json::require_i64(value, "max_position_embeddings");
    rms_norm_eps = json::optional_f32(value, "rms_norm_eps", rms_norm_eps);
    rope_theta = json::optional_f32(value, "rope_theta", rope_theta);
}

AceStepPlannerConfig parse_planner_config(const engine::io::json::Value & value) {
    AceStepPlannerConfig config;
    config.lm_family = json::optional_string(value, "model_type", config.lm_family);
    parse_qwen_common(
        value,
        config.vocab_size,
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.max_position_embeddings,
        config.rms_norm_eps,
        config.rope_theta);
    config.bos_token_id = json::optional_i64(value, "bos_token_id", config.bos_token_id);
    config.eos_token_id = json::optional_i64(value, "eos_token_id", config.eos_token_id);
    config.pad_token_id = json::optional_i64(value, "pad_token_id", config.pad_token_id);
    return config;
}

AceStepTextEncoderConfig parse_text_encoder_config(const engine::io::json::Value & value) {
    AceStepTextEncoderConfig config;
    config.encoder_family = json::optional_string(value, "model_type", config.encoder_family);
    parse_qwen_common(
        value,
        config.vocab_size,
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.max_position_embeddings,
        config.rms_norm_eps,
        config.rope_theta);
    return config;
}

AceStepDiffusionConfig parse_diffusion_config(const engine::io::json::Value & value) {
    AceStepDiffusionConfig config;
    config.model_type = json::optional_string(value, "model_type", config.model_type);
    config.model_version = json::optional_string(value, "model_version", config.model_version);
    config.hidden_size = json::require_i64(value, "hidden_size");
    config.intermediate_size = json::require_i64(value, "intermediate_size");
    config.num_hidden_layers = json::require_i64(value, "num_hidden_layers");
    config.num_attention_heads = json::require_i64(value, "num_attention_heads");
    config.num_key_value_heads = json::require_i64(value, "num_key_value_heads");
    config.head_dim = json::optional_i64(value, "head_dim", config.hidden_size / config.num_attention_heads);
    config.text_hidden_dim = json::require_i64(value, "text_hidden_dim");
    config.in_channels = json::require_i64(value, "in_channels");
    config.patch_size = json::require_i64(value, "patch_size");
    config.latent_channels = json::require_i64(value, "audio_acoustic_hidden_dim");
    config.pool_window_size = json::require_i64(value, "pool_window_size");
    config.timbre_hidden_dim = json::require_i64(value, "timbre_hidden_dim");
    config.timbre_fix_frame = json::require_i64(value, "timbre_fix_frame");
    config.num_lyric_encoder_hidden_layers = json::require_i64(value, "num_lyric_encoder_hidden_layers");
    config.num_timbre_encoder_hidden_layers = json::require_i64(value, "num_timbre_encoder_hidden_layers");
    config.num_attention_pooler_hidden_layers = json::require_i64(value, "num_attention_pooler_hidden_layers");
    config.fsq_input_num_quantizers = json::require_i64(value, "fsq_input_num_quantizers");
    config.fsq_input_levels = json::optional_i64_array(value, "fsq_input_levels");
    config.fsq_dim = static_cast<int64_t>(config.fsq_input_levels.size());
    config.sliding_window = json::optional_i64(value, "sliding_window", 0);
    config.use_sliding_window = json::optional_bool(value, "use_sliding_window", false);
    config.layer_types = json::optional_string_array(value, "layer_types");
    config.is_turbo = json::optional_bool(value, "is_turbo", config.is_turbo);
    config.rms_norm_eps = json::optional_f32(value, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = json::optional_f32(value, "rope_theta", config.rope_theta);
    return config;
}

AceStepVAEConfig parse_vae_config(const engine::io::json::Value & value) {
    AceStepVAEConfig config;
    config.sample_rate = static_cast<int>(json::optional_i64(value, "sampling_rate", config.sample_rate));
    config.audio_channels = static_cast<int>(json::optional_i64(value, "audio_channels", config.audio_channels));
    config.encoder_hidden_size = json::optional_i64(value, "encoder_hidden_size", config.encoder_hidden_size);
    config.decoder_channels = json::optional_i64(value, "decoder_channels", config.decoder_channels);
    config.decoder_input_channels = json::optional_i64(value, "decoder_input_channels", config.decoder_input_channels);
    config.downsampling_ratios = json::optional_i64_array(value, "downsampling_ratios");
    config.channel_multiples = json::optional_i64_array(value, "channel_multiples");
    return config;
}

std::string dit_resource_id(const AceStepModelSelection & selection, std::string_view suffix) {
    if (selection.dit_model_path == "acestep-v15-turbo") {
        return "dit_turbo_" + std::string(suffix);
    }
    if (selection.dit_model_path == "acestep-v15-base") {
        return "dit_base_" + std::string(suffix);
    }
    throw std::runtime_error("ACE-Step package spec supports only acestep-v15-turbo and acestep-v15-base DiT variants");
}

void validate_selection(const AceStepModelSelection & selection) {
    (void)dit_resource_id(selection, "config");
}

AceStepConfig parse_config(const assets::ResourceBundle & resources, const AceStepModelSelection & selection) {
    AceStepConfig config;
    config.diffusion = parse_diffusion_config(resources.parse_json(dit_resource_id(selection, "config")));
    config.planner = parse_planner_config(resources.parse_json("lm_config"));
    config.text_encoder = parse_text_encoder_config(resources.parse_json("text_encoder_config"));
    config.vae = parse_vae_config(resources.parse_json("vae_config"));
    return config;
}

void validate_config(const AceStepConfig & config) {
    if (config.diffusion.model_type != "acestep") {
        throw std::runtime_error("ACE-Step diffusion config must have model_type=acestep");
    }
    if (config.planner.lm_family != "qwen3") {
        throw std::runtime_error("ACE-Step planner LM currently supports only qwen3");
    }
    if (config.text_encoder.encoder_family != "qwen3") {
        throw std::runtime_error("ACE-Step text encoder currently supports only qwen3");
    }
    if (config.vae.sample_rate != config.diffusion.sample_rate) {
        throw std::runtime_error("ACE-Step VAE sample rate must match diffusion sample rate");
    }
}

}  // namespace

std::shared_ptr<const AceStepAssets> load_ace_step_assets(
    const std::filesystem::path & model_path,
    const AceStepModelSelection & selection) {
    auto assets = std::make_shared<AceStepAssets>();
    assets->selection = selection;
    validate_selection(assets->selection);
    assets->resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("ace_step"));
    assets->config = parse_config(assets->resources, assets->selection);
    validate_config(assets->config);
    assets->dit_weights = assets->resources.open_tensor_source(dit_resource_id(assets->selection, "weights"));
    assets->dit_silence_latent =
        assets->resources.open_tensor_source(dit_resource_id(assets->selection, "silence_latent"));
    assets->lm_weights = assets->resources.open_tensor_source("lm_weights");
    assets->text_encoder_weights = assets->resources.open_tensor_source("text_encoder_weights");
    assets->vae_weights = assets->resources.open_tensor_source("vae_weights");
    return assets;
}

}  // namespace engine::models::ace_step
