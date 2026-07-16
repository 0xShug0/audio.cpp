#include "engine/models/irodori_tts/assets.h"

#include "engine/framework/assets/model_package.h"
#include "engine/framework/io/config.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <string>

namespace engine::models::irodori_tts {
namespace json = engine::io::json;
namespace {

IrodoriModelConfig parse_model_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("model_config");

    IrodoriModelConfig config;
    config.latent_dim = json::optional_i64(root, "latent_dim", config.latent_dim);
    config.latent_patch_size = json::optional_i64(root, "latent_patch_size", config.latent_patch_size);
    config.model_dim = json::optional_i64(root, "model_dim", config.model_dim);
    config.num_layers = json::optional_i64(root, "num_layers", config.num_layers);
    config.num_heads = json::optional_i64(root, "num_heads", config.num_heads);
    config.mlp_ratio = json::optional_f32(root, "mlp_ratio", config.mlp_ratio);
    config.text_mlp_ratio = json::optional_nullable_f32(root, "text_mlp_ratio", config.mlp_ratio);
    config.speaker_mlp_ratio = json::optional_nullable_f32(root, "speaker_mlp_ratio", config.mlp_ratio);
    config.dropout = json::optional_f32(root, "dropout", config.dropout);
    config.text_vocab_size = json::optional_i64(root, "text_vocab_size", config.text_vocab_size);
    config.text_add_bos = json::optional_bool(root, "text_add_bos", config.text_add_bos);
    config.text_dim = json::optional_i64(root, "text_dim", config.text_dim);
    config.text_layers = json::optional_i64(root, "text_layers", config.text_layers);
    config.text_heads = json::optional_i64(root, "text_heads", config.text_heads);
    config.use_caption_condition = json::optional_bool(root, "use_caption_condition", config.use_caption_condition);
    const bool legacy_speaker_default = !config.use_caption_condition;
    config.use_speaker_condition = json::optional_nullable_bool(root, "use_speaker_condition", legacy_speaker_default);
    config.caption_vocab_size = json::optional_nullable_i64(root, "caption_vocab_size", 0);
    config.caption_dim = json::optional_nullable_i64(root, "caption_dim", 0);
    config.caption_layers = json::optional_nullable_i64(root, "caption_layers", 0);
    config.caption_heads = json::optional_nullable_i64(root, "caption_heads", 0);
    config.caption_mlp_ratio = json::optional_nullable_f32(root, "caption_mlp_ratio", 0.0F);
    config.speaker_dim = json::optional_i64(root, "speaker_dim", config.speaker_dim);
    config.speaker_layers = json::optional_i64(root, "speaker_layers", config.speaker_layers);
    config.speaker_heads = json::optional_i64(root, "speaker_heads", config.speaker_heads);
    config.speaker_patch_size = json::optional_i64(root, "speaker_patch_size", config.speaker_patch_size);
    config.timestep_embed_dim = json::optional_i64(root, "timestep_embed_dim", config.timestep_embed_dim);
    config.adaln_rank = json::optional_i64(root, "adaln_rank", config.adaln_rank);
    config.norm_eps = json::optional_f32(root, "norm_eps", config.norm_eps);
    config.use_duration_predictor = json::optional_bool(root, "use_duration_predictor", config.use_duration_predictor);
    config.duration_aux_dim = json::optional_i64(root, "duration_aux_dim", config.duration_aux_dim);
    config.duration_hidden_dim = json::optional_i64(root, "duration_hidden_dim", config.duration_hidden_dim);
    config.duration_layers = json::optional_i64(root, "duration_layers", config.duration_layers);
    config.duration_dropout = json::optional_f32(root, "duration_dropout", config.duration_dropout);
    config.duration_attention_heads =
        json::optional_i64(root, "duration_attention_heads", config.duration_attention_heads);
    config.duration_architecture =
        json::optional_string(root, "duration_architecture", config.duration_architecture);
    config.duration_token_init_frames =
        json::optional_f32(root, "duration_token_init_frames", config.duration_token_init_frames);
    config.duration_speaker_fusion =
        json::optional_string(root, "duration_speaker_fusion", config.duration_speaker_fusion);
    config.duration_caption_fusion =
        json::optional_nullable_string(root, "duration_caption_fusion", config.duration_caption_fusion);
    config.duration_caption_pooling =
        json::optional_nullable_string(root, "duration_caption_pooling", config.duration_caption_pooling);
    config.max_text_len = json::optional_i64(root, "max_text_len", config.max_text_len);
    config.max_caption_len = json::optional_i64(root, "max_caption_len", config.max_text_len);

    engine::io::require_positive(config.latent_dim, "latent_dim");
    engine::io::require_positive(config.latent_patch_size, "latent_patch_size");
    engine::io::require_positive(config.model_dim, "model_dim");
    engine::io::require_positive(config.num_layers, "num_layers");
    engine::io::require_positive(config.num_heads, "num_heads");
    engine::io::require_positive(config.mlp_ratio, "mlp_ratio");
    engine::io::require_positive(config.text_vocab_size, "text_vocab_size");
    engine::io::require_positive(config.text_dim, "text_dim");
    engine::io::require_positive(config.text_layers, "text_layers");
    engine::io::require_positive(config.text_heads, "text_heads");
    engine::io::require_positive(config.timestep_embed_dim, "timestep_embed_dim");
    engine::io::require_positive(config.adaln_rank, "adaln_rank");
    engine::io::require_positive(config.norm_eps, "norm_eps");
    engine::io::require_divisible(config.model_dim, config.num_heads, "model_dim / num_heads");
    engine::io::require_divisible(config.text_dim, config.text_heads, "text_dim / text_heads");
    if (config.use_speaker_condition) {
        engine::io::require_positive(config.speaker_dim, "speaker_dim");
        engine::io::require_positive(config.speaker_layers, "speaker_layers");
        engine::io::require_positive(config.speaker_heads, "speaker_heads");
        engine::io::require_positive(config.speaker_patch_size, "speaker_patch_size");
        engine::io::require_divisible(config.speaker_dim, config.speaker_heads, "speaker_dim / speaker_heads");
    }
    if (config.use_caption_condition) {
        engine::io::require_positive(config.caption_vocab_size_resolved(), "caption_vocab_size");
        engine::io::require_positive(config.caption_dim_resolved(), "caption_dim");
        engine::io::require_positive(config.caption_layers_resolved(), "caption_layers");
        engine::io::require_positive(config.caption_heads_resolved(), "caption_heads");
        engine::io::require_divisible(config.caption_dim_resolved(), config.caption_heads_resolved(), "caption_dim / caption_heads");
    }
    if (config.use_duration_predictor) {
        engine::io::require_positive(config.duration_aux_dim, "duration_aux_dim");
        engine::io::require_positive(config.duration_hidden_dim, "duration_hidden_dim");
        engine::io::require_positive(config.duration_layers, "duration_layers");
    }
    return config;
}

IrodoriModelConfig parse_config(const assets::ResourceBundle & resources) {
    return parse_model_config(resources);
}

void validate_model_weights(const IrodoriModelConfig & config, const assets::TensorSource & source) {
    assets::require_tensor_shape(source, "text_encoder.text_embedding.weight", {config.text_vocab_size, config.text_dim});
    assets::require_tensor_shape(source, "text_norm.weight", {config.text_dim});
    assets::require_tensor_shape(source, "in_proj.weight", {config.model_dim, config.patched_latent_dim()});
    assets::require_tensor_shape(source, "in_proj.bias", {config.model_dim});
    assets::require_tensor_shape(source, "out_norm.weight", {config.model_dim});
    assets::require_tensor_shape(source, "out_proj.weight", {config.patched_latent_dim(), config.model_dim});
    assets::require_tensor_shape(source, "out_proj.bias", {config.patched_latent_dim()});
    assets::require_tensor_shape(source, "blocks.0.attention.wq.weight", {config.model_dim, config.model_dim});
    assets::require_tensor_shape(source, "blocks.0.attention.wk_text.weight", {config.model_dim, config.text_dim});
    if (config.use_speaker_condition) {
        assets::require_tensor_shape(source, "speaker_encoder.in_proj.weight", {config.speaker_dim, config.speaker_patched_latent_dim()});
        assets::require_tensor_shape(source, "speaker_norm.weight", {config.speaker_dim});
        assets::require_tensor_shape(source, "blocks.0.attention.wk_speaker.weight", {config.model_dim, config.speaker_dim});
    }
    if (config.use_caption_condition) {
        assets::require_tensor_shape(source, "caption_encoder.text_embedding.weight", {config.caption_vocab_size_resolved(), config.caption_dim_resolved()});
        assets::require_tensor_shape(source, "caption_norm.weight", {config.caption_dim_resolved()});
        assets::require_tensor_shape(source, "blocks.0.attention.wk_caption.weight", {config.model_dim, config.caption_dim_resolved()});
        assets::require_tensor_shape(source, "blocks.0.attention.wv_caption.weight", {config.model_dim, config.caption_dim_resolved()});
        assets::require_tensor_shape(source, "duration_predictor.null_caption", {config.caption_dim_resolved()});
        assets::require_tensor_shape(
            source,
            "duration_predictor.token_blocks.0.caption_modulation.weight",
            {3 * config.duration_hidden_dim, config.caption_dim_resolved()});
    }
    if (config.use_duration_predictor) {
        assets::require_tensor_shape(source, "duration_predictor.token_input_proj.weight", {config.duration_hidden_dim, config.text_dim});
        assets::require_tensor_shape(source, "duration_predictor.token_out_proj.weight", {1, config.duration_hidden_dim});
        assets::require_tensor_shape(source, "duration_predictor.token_out_proj.bias", {1});
    }
}

void validate_codec_weights(const IrodoriCodecConfig & config, const assets::TensorSource & source) {
    assets::require_tensor_shape(source, "quantizer.out_proj.weight_v", {config.latent_dim, config.codebook_dim, 1});
    assets::require_tensor_shape(source, "quantizer.out_proj.weight_g", {config.latent_dim, 1, 1});
    assets::require_tensor_shape(source, "quantizer.out_proj.bias", {config.latent_dim});
    assets::require_tensor_shape(source, "decoder.model.0.weight_v", {config.decoder_dim, config.latent_dim, 7});
    assets::require_tensor_shape(source, "decoder.model.0.weight_g", {config.decoder_dim, 1, 1});
    assets::require_tensor_shape(source, "decoder.model.0.bias", {config.decoder_dim});
}

void validate_weight_anchors(const IrodoriTTSAssets & assets) {
    validate_model_weights(assets.config, *assets.model_weights);
    validate_codec_weights(assets.codec, *assets.codec_weights);
}

}  // namespace

int64_t IrodoriModelConfig::patched_latent_dim() const noexcept {
    return latent_dim * latent_patch_size;
}

int64_t IrodoriModelConfig::speaker_patched_latent_dim() const noexcept {
    return patched_latent_dim() * speaker_patch_size;
}

int64_t IrodoriModelConfig::caption_vocab_size_resolved() const noexcept {
    return caption_vocab_size > 0 ? caption_vocab_size : text_vocab_size;
}

int64_t IrodoriModelConfig::caption_dim_resolved() const noexcept {
    return caption_dim > 0 ? caption_dim : text_dim;
}

int64_t IrodoriModelConfig::caption_layers_resolved() const noexcept {
    return caption_layers > 0 ? caption_layers : text_layers;
}

int64_t IrodoriModelConfig::caption_heads_resolved() const noexcept {
    return caption_heads > 0 ? caption_heads : text_heads;
}

float IrodoriModelConfig::caption_mlp_ratio_resolved() const noexcept {
    return caption_mlp_ratio > 0.0F ? caption_mlp_ratio : text_mlp_ratio;
}

std::shared_ptr<const IrodoriTTSAssets> load_irodori_tts_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<IrodoriTTSAssets>();
    assets->resources = assets::load_resource_bundle_from_package_spec(
        model_path,
        assets::default_model_package_spec_path("irodori_tts"));

    assets->config = parse_config(assets->resources);
    assets->model_weights = assets->resources.open_tensor_source("model_weights");
    assets->codec_weights = assets->resources.open_tensor_source("codec_weights");
    validate_weight_anchors(*assets);
    return assets;
}

}  // namespace engine::models::irodori_tts
