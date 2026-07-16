#include "engine/models/moss/moss_tts_local/assets.h"

#include "engine/framework/assets/model_package.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::moss_tts_local {
namespace json = engine::io::json;
namespace {

MossBackboneConfig parse_backbone_config(const json::Value & value) {
    MossBackboneConfig config;
    config.hidden_size = json::require_i64(value, "hidden_size");
    config.intermediate_size = json::require_i64(value, "intermediate_size");
    config.num_hidden_layers = json::require_i64(value, "num_hidden_layers");
    config.num_attention_heads = json::require_i64(value, "num_attention_heads");
    config.num_key_value_heads = json::require_i64(value, "num_key_value_heads");
    engine::io::require_positive(config.num_attention_heads, "backbone num_attention_heads");
    config.head_dim =
        json::optional_i64(value, "head_dim", config.hidden_size / config.num_attention_heads);
    config.max_position_embeddings = json::require_i64(value, "max_position_embeddings");
    config.vocab_size = json::require_i64(value, "vocab_size");
    config.rms_norm_eps = json::optional_f32(value, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = json::optional_f32(value, "rope_theta", config.rope_theta);
    config.tie_word_embeddings = json::optional_bool(value, "tie_word_embeddings", config.tie_word_embeddings);
    engine::io::require_positive(config.hidden_size, "backbone hidden_size");
    engine::io::require_positive(config.intermediate_size, "backbone intermediate_size");
    engine::io::require_positive(config.num_hidden_layers, "backbone num_hidden_layers");
    engine::io::require_positive(config.num_key_value_heads, "backbone num_key_value_heads");
    engine::io::require_positive(config.head_dim, "backbone head_dim");
    engine::io::require_positive(config.vocab_size, "backbone vocab_size");
    return config;
}

MossLocalTransformerConfig parse_local_config(const json::Value & value) {
    MossLocalTransformerConfig config;
    config.hidden_size = json::require_i64(value, "n_embd");
    config.intermediate_size = json::optional_i64(value, "n_inner", config.hidden_size * 4);
    config.num_layers = json::require_i64(value, "n_layer");
    config.num_heads = json::require_i64(value, "n_head");
    config.max_positions = json::optional_i64(value, "n_positions", config.max_positions);
    config.layer_norm_eps = json::optional_f32(value, "layer_norm_epsilon", config.layer_norm_eps);
    config.rope_base = json::optional_f32(value, "rope_base", config.rope_base);
    engine::io::require_positive(config.hidden_size, "local n_embd");
    engine::io::require_positive(config.num_layers, "local n_layer");
    engine::io::require_positive(config.num_heads, "local n_head");
    return config;
}

}  // namespace

MossTTSLocalConfig parse_model_config(const json::Value & root) {
    const auto model_type = json::optional_string(root, "model_type", "");
    if (model_type != "moss_tts_local") {
        throw std::runtime_error(
            "MOSS-TTS-Local config model_type mismatch: expected moss_tts_local, got " + model_type);
    }
    MossTTSLocalConfig config;
    config.backbone = parse_backbone_config(root.require("qwen3_config"));
    config.local = parse_local_config(root.require("gpt2_config"));
    config.num_codebooks = json::require_i64(root, "n_vq");
    config.audio_vocab_size = json::optional_i64(root, "audio_vocab_size", 0);
    config.audio_codebook_sizes = json::optional_i64_array(root, "audio_codebook_sizes");
    config.audio_pad_token_id = json::optional_i64(root, "audio_pad_token_id", 0);
    config.pad_token_id = json::optional_i64(root, "pad_token_id", 0);
    config.im_start_token_id = json::optional_i64(root, "im_start_token_id", 0);
    config.im_end_token_id = json::optional_i64(root, "im_end_token_id", 0);
    config.audio_start_token_id = json::optional_i64(root, "audio_start_token_id", 0);
    config.audio_end_token_id = json::optional_i64(root, "audio_end_token_id", 0);
    config.audio_user_slot_token_id = json::optional_i64(root, "audio_user_slot_token_id", 0);
    config.audio_assistant_slot_token_id = json::optional_i64(root, "audio_assistant_slot_token_id", 0);
    config.sampling_rate = json::optional_i64(root, "sampling_rate", 0);
    config.local_text_head_mode = json::optional_string(root, "local_text_head_mode", "");
    engine::io::require_positive(config.num_codebooks, "n_vq");
    if (!config.audio_codebook_sizes.empty() &&
        static_cast<int64_t>(config.audio_codebook_sizes.size()) != config.num_codebooks) {
        throw std::runtime_error("MOSS-TTS-Local audio_codebook_sizes length does not match n_vq");
    }
    return config;
}

MossTTSLocalConfig parse_config(const assets::ResourceBundle & resources) {
    return parse_model_config(resources.parse_json("config"));
}

std::shared_ptr<const MossTTSLocalAssets> load_moss_tts_local_assets(const std::filesystem::path & model_path) {
    MossTTSLocalAssets assets;
    assets.resources = assets::load_resource_bundle_from_package_spec(
        model_path,
        assets::default_model_package_spec_path("moss_tts_local"));
    assets.config = parse_config(assets.resources);
    assets.model_weights = assets.resources.open_tensor_source("model_weights");
    assets.audio_tokenizer_weights = assets.resources.open_tensor_source("audio_tokenizer_weights");
    return std::make_shared<MossTTSLocalAssets>(std::move(assets));
}

}  // namespace engine::models::moss_tts_local
