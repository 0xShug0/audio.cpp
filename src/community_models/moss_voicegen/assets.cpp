#include "engine/community_models/moss_voicegen/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::moss_voicegen {
namespace json = engine::io::json;
namespace {

MossVoiceGenBackboneConfig parse_backbone_config(const json::Value & value) {
    MossVoiceGenBackboneConfig config;
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

}  // namespace

MossVoiceGenConfig parse_model_config(const json::Value & root) {
    const auto model_type = json::optional_string(root, "model_type", "");
    if (model_type != "moss_tts_delay") {
        throw std::runtime_error(
            "MOSS-VoiceGenerator config model_type mismatch: expected moss_tts_delay, got " + model_type);
    }
    MossVoiceGenConfig config;
    config.backbone = parse_backbone_config(root.require("language_config"));
    config.num_codebooks = json::require_i64(root, "n_vq");
    config.audio_vocab_size = json::require_i64(root, "audio_vocab_size");
    config.audio_pad_code = json::require_i64(root, "audio_pad_code");
    config.pad_token_id = json::optional_i64(root, "pad_token_id", config.pad_token_id);
    config.im_start_token_id = json::optional_i64(root, "im_start_token_id", config.im_start_token_id);
    config.im_end_token_id = json::optional_i64(root, "im_end_token_id", config.im_end_token_id);
    config.audio_start_token_id = json::require_i64(root, "audio_start_token_id");
    config.audio_end_token_id = json::require_i64(root, "audio_end_token_id");
    config.audio_user_slot_token_id = json::require_i64(root, "audio_user_slot_token_id");
    config.audio_assistant_gen_slot_token_id = json::require_i64(root, "audio_assistant_gen_slot_token_id");
    config.audio_assistant_delay_slot_token_id = json::require_i64(root, "audio_assistant_delay_slot_token_id");
    config.sampling_rate = json::optional_i64(root, "sampling_rate", 24000);
    engine::io::require_positive(config.num_codebooks, "n_vq");
    engine::io::require_positive(config.audio_vocab_size, "audio_vocab_size");
    // The checkpoint carries one embedding table and one head per codebook. A config that
    // claims the family default of 32 against a 16-codebook checkpoint would leave the
    // generator without a usable length bound, so refuse it rather than half-load.
    if (config.audio_pad_code != config.audio_vocab_size) {
        throw std::runtime_error(
            "MOSS-VoiceGenerator expects audio_pad_code to be the code past audio_vocab_size");
    }
    return config;
}

MossVoiceGenConfig parse_config(const assets::ResourceBundle & resources) {
    return parse_model_config(resources.parse_json("config"));
}

std::shared_ptr<const MossVoiceGenAssets> load_moss_voicegen_assets(const std::filesystem::path & model_path) {
    MossVoiceGenAssets assets;
    assets.resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("moss_voicegen"));
    assets.config = parse_config(assets.resources);
    assets.model_weights = assets.resources.open_tensor_source("model_weights");
    assets.audio_tokenizer_weights = assets.resources.open_tensor_source("audio_tokenizer_weights");
    return std::make_shared<MossVoiceGenAssets>(std::move(assets));
}

}  // namespace engine::models::moss_voicegen
