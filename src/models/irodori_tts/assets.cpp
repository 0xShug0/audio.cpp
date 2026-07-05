#include "engine/models/irodori_tts/assets.h"

#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/safetensors.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace engine::models::irodori_tts {
namespace json = engine::io::json;
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("Irodori-TTS model path does not exist: " + model_path.string());
}

std::filesystem::path repo_root_from_model_root(const std::filesystem::path & model_root) {
    const auto parent = model_root.parent_path();
    if (parent.filename() == "models") {
        return parent.parent_path();
    }
    return std::filesystem::current_path();
}

std::filesystem::path require_file(const std::filesystem::path & path, const char * label) {
    if (!engine::io::is_existing_file(path)) {
        throw std::runtime_error(std::string("missing Irodori-TTS ") + label + ": " + path.string());
    }
    return std::filesystem::weakly_canonical(path);
}

std::filesystem::path require_dir(const std::filesystem::path & path, const char * label) {
    if (!engine::io::is_existing_directory(path)) {
        throw std::runtime_error(std::string("missing Irodori-TTS ") + label + ": " + path.string());
    }
    return std::filesystem::weakly_canonical(path);
}

void require_positive(int64_t value, const char * label) {
    if (value <= 0) {
        throw std::runtime_error(std::string("Irodori-TTS config contains non-positive ") + label);
    }
}

void require_positive_float(float value, const char * label) {
    if (!(value > 0.0F) || !std::isfinite(value)) {
        throw std::runtime_error(std::string("Irodori-TTS config contains invalid ") + label);
    }
}

void require_divisible(int64_t value, int64_t divisor, const char * label) {
    if (divisor <= 0 || value % divisor != 0) {
        throw std::runtime_error(std::string("Irodori-TTS config invalid divisibility for ") + label);
    }
}

bool optional_nullable_bool(const engine::io::json::Value & object, const std::string & key, bool fallback) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return fallback;
    }
    return value->as_bool();
}

int64_t optional_nullable_i64(const engine::io::json::Value & object, const std::string & key, int64_t fallback) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return fallback;
    }
    return value->as_i64();
}

float optional_nullable_f32(const engine::io::json::Value & object, const std::string & key, float fallback) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return fallback;
    }
    return value->as_f32();
}

std::string optional_nullable_string(
    const engine::io::json::Value & object,
    const std::string & key,
    const std::string & fallback) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return fallback;
    }
    return value->as_string();
}

IrodoriModelConfig parse_model_config_json(const std::string & raw) {
    const auto root = engine::io::json::parse(raw);
    IrodoriModelConfig config;
    config.latent_dim = json::optional_i64(root, "latent_dim", config.latent_dim);
    config.latent_patch_size = json::optional_i64(root, "latent_patch_size", config.latent_patch_size);
    config.model_dim = json::optional_i64(root, "model_dim", config.model_dim);
    config.num_layers = json::optional_i64(root, "num_layers", config.num_layers);
    config.num_heads = json::optional_i64(root, "num_heads", config.num_heads);
    config.mlp_ratio = json::optional_f32(root, "mlp_ratio", config.mlp_ratio);
    config.text_mlp_ratio = optional_nullable_f32(root, "text_mlp_ratio", config.mlp_ratio);
    config.speaker_mlp_ratio = optional_nullable_f32(root, "speaker_mlp_ratio", config.mlp_ratio);
    config.dropout = json::optional_f32(root, "dropout", config.dropout);
    config.text_vocab_size = json::optional_i64(root, "text_vocab_size", config.text_vocab_size);
    config.text_tokenizer_repo = json::optional_string(root, "text_tokenizer_repo", config.text_tokenizer_repo);
    config.text_add_bos = json::optional_bool(root, "text_add_bos", config.text_add_bos);
    config.text_dim = json::optional_i64(root, "text_dim", config.text_dim);
    config.text_layers = json::optional_i64(root, "text_layers", config.text_layers);
    config.text_heads = json::optional_i64(root, "text_heads", config.text_heads);
    config.use_caption_condition = json::optional_bool(root, "use_caption_condition", config.use_caption_condition);
    const bool legacy_speaker_default = !config.use_caption_condition;
    config.use_speaker_condition = optional_nullable_bool(root, "use_speaker_condition", legacy_speaker_default);
    config.caption_vocab_size = optional_nullable_i64(root, "caption_vocab_size", 0);
    config.caption_dim = optional_nullable_i64(root, "caption_dim", 0);
    config.caption_layers = optional_nullable_i64(root, "caption_layers", 0);
    config.caption_heads = optional_nullable_i64(root, "caption_heads", 0);
    config.caption_mlp_ratio = optional_nullable_f32(root, "caption_mlp_ratio", 0.0F);
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
        optional_nullable_string(root, "duration_caption_fusion", config.duration_caption_fusion);
    config.duration_caption_pooling =
        optional_nullable_string(root, "duration_caption_pooling", config.duration_caption_pooling);
    config.max_text_len = json::optional_i64(root, "max_text_len", config.max_text_len);
    config.max_caption_len = json::optional_i64(root, "max_caption_len", config.max_text_len);

    require_positive(config.latent_dim, "latent_dim");
    require_positive(config.latent_patch_size, "latent_patch_size");
    require_positive(config.model_dim, "model_dim");
    require_positive(config.num_layers, "num_layers");
    require_positive(config.num_heads, "num_heads");
    require_positive_float(config.mlp_ratio, "mlp_ratio");
    require_positive(config.text_vocab_size, "text_vocab_size");
    require_positive(config.text_dim, "text_dim");
    require_positive(config.text_layers, "text_layers");
    require_positive(config.text_heads, "text_heads");
    require_positive(config.timestep_embed_dim, "timestep_embed_dim");
    require_positive(config.adaln_rank, "adaln_rank");
    require_positive_float(config.norm_eps, "norm_eps");
    require_divisible(config.model_dim, config.num_heads, "model_dim / num_heads");
    require_divisible(config.text_dim, config.text_heads, "text_dim / text_heads");
    if (config.use_speaker_condition) {
        require_positive(config.speaker_dim, "speaker_dim");
        require_positive(config.speaker_layers, "speaker_layers");
        require_positive(config.speaker_heads, "speaker_heads");
        require_positive(config.speaker_patch_size, "speaker_patch_size");
        require_divisible(config.speaker_dim, config.speaker_heads, "speaker_dim / speaker_heads");
    }
    if (config.use_caption_condition) {
        require_positive(config.caption_vocab_size_resolved(), "caption_vocab_size");
        require_positive(config.caption_dim_resolved(), "caption_dim");
        require_positive(config.caption_layers_resolved(), "caption_layers");
        require_positive(config.caption_heads_resolved(), "caption_heads");
        require_divisible(config.caption_dim_resolved(), config.caption_heads_resolved(), "caption_dim / caption_heads");
    }
    if (config.use_duration_predictor) {
        require_positive(config.duration_aux_dim, "duration_aux_dim");
        require_positive(config.duration_hidden_dim, "duration_hidden_dim");
        require_positive(config.duration_layers, "duration_layers");
    }
    return config;
}

void require_tensor_metadata(
    const assets::TensorSource & source,
    const std::string & name,
    const std::vector<int64_t> & expected_shape) {
    const auto metadata = source.require_metadata(name);
    if (metadata.shape != expected_shape) {
        throw std::runtime_error("Irodori-TTS tensor shape mismatch for " + name);
    }
}

void validate_model_weights(const IrodoriModelConfig & config, const assets::TensorSource & source) {
    require_tensor_metadata(source, "text_encoder.text_embedding.weight", {config.text_vocab_size, config.text_dim});
    require_tensor_metadata(source, "text_norm.weight", {config.text_dim});
    require_tensor_metadata(source, "in_proj.weight", {config.model_dim, config.patched_latent_dim()});
    require_tensor_metadata(source, "in_proj.bias", {config.model_dim});
    require_tensor_metadata(source, "out_norm.weight", {config.model_dim});
    require_tensor_metadata(source, "out_proj.weight", {config.patched_latent_dim(), config.model_dim});
    require_tensor_metadata(source, "out_proj.bias", {config.patched_latent_dim()});
    require_tensor_metadata(source, "blocks.0.attention.wq.weight", {config.model_dim, config.model_dim});
    require_tensor_metadata(source, "blocks.0.attention.wk_text.weight", {config.model_dim, config.text_dim});
    if (config.use_speaker_condition) {
        require_tensor_metadata(source, "speaker_encoder.in_proj.weight", {config.speaker_dim, config.speaker_patched_latent_dim()});
        require_tensor_metadata(source, "speaker_norm.weight", {config.speaker_dim});
        require_tensor_metadata(source, "blocks.0.attention.wk_speaker.weight", {config.model_dim, config.speaker_dim});
    }
    if (config.use_caption_condition) {
        require_tensor_metadata(source, "caption_encoder.text_embedding.weight", {config.caption_vocab_size_resolved(), config.caption_dim_resolved()});
        require_tensor_metadata(source, "caption_norm.weight", {config.caption_dim_resolved()});
        require_tensor_metadata(source, "blocks.0.attention.wk_caption.weight", {config.model_dim, config.caption_dim_resolved()});
        require_tensor_metadata(source, "blocks.0.attention.wv_caption.weight", {config.model_dim, config.caption_dim_resolved()});
        require_tensor_metadata(source, "duration_predictor.null_caption", {config.caption_dim_resolved()});
        require_tensor_metadata(
            source,
            "duration_predictor.token_blocks.0.caption_modulation.weight",
            {3 * config.duration_hidden_dim, config.caption_dim_resolved()});
    }
    if (config.use_duration_predictor) {
        require_tensor_metadata(source, "duration_predictor.token_input_proj.weight", {config.duration_hidden_dim, config.text_dim});
        require_tensor_metadata(source, "duration_predictor.token_out_proj.weight", {1, config.duration_hidden_dim});
        require_tensor_metadata(source, "duration_predictor.token_out_proj.bias", {1});
    }
}

void validate_codec_weights(const IrodoriCodecConfig & config, const assets::TensorSource & source) {
    require_tensor_metadata(source, "quantizer.out_proj.weight_v", {config.latent_dim, config.codebook_dim, 1});
    require_tensor_metadata(source, "quantizer.out_proj.weight_g", {config.latent_dim, 1, 1});
    require_tensor_metadata(source, "quantizer.out_proj.bias", {config.latent_dim});
    require_tensor_metadata(source, "decoder.model.0.weight_v", {config.decoder_dim, config.latent_dim, 7});
    require_tensor_metadata(source, "decoder.model.0.weight_g", {config.decoder_dim, 1, 1});
    require_tensor_metadata(source, "decoder.model.0.bias", {config.decoder_dim});
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

IrodoriAssetPaths resolve_irodori_assets(const std::filesystem::path & model_path) {
    IrodoriAssetPaths paths;
    paths.model_root = resolve_model_root(model_path);
    paths.model_safetensors_path = require_file(paths.model_root / "model.safetensors", "model.safetensors");

    const auto repo_root = repo_root_from_model_root(paths.model_root);
    const auto tokenizer_root = require_dir(repo_root / "models" / "llm-jp-3-150m", "llm-jp tokenizer directory");
    paths.tokenizer_json_path = require_file(tokenizer_root / "tokenizer.json", "llm-jp tokenizer.json");

    paths.codec_root = require_dir(repo_root / "models" / "Semantic-DACVAE-Japanese-32dim", "DACVAE codec directory");
    paths.codec_safetensors_path = require_file(paths.codec_root / "weights.safetensors", "DACVAE weights.safetensors");
    return paths;
}

std::shared_ptr<const IrodoriAssets> load_irodori_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<IrodoriAssets>();
    assets->paths = resolve_irodori_assets(model_path);

    const auto index = engine::io::load_safetensors_index(assets->paths.model_safetensors_path);
    const auto metadata_it = index.metadata.find("config_json");
    if (metadata_it == index.metadata.end()) {
        throw std::runtime_error("Irodori-TTS model.safetensors is missing config_json metadata");
    }
    assets->config = parse_model_config_json(metadata_it->second);
    assets->model_weights = engine::assets::open_tensor_source(assets->paths.model_safetensors_path);
    assets->codec_weights = engine::assets::open_tensor_source(assets->paths.codec_safetensors_path);
    validate_model_weights(assets->config, *assets->model_weights);
    validate_codec_weights(assets->codec, *assets->codec_weights);
    return assets;
}

}  // namespace engine::models::irodori_tts
