#include "engine/framework/decoders/moss_tts_delay/config.h"

#include "engine/framework/io/config.h"

#include <stdexcept>
#include <string>

namespace engine::decoders {
namespace json = engine::io::json;
namespace {

MossTtsDelayBackboneConfig parse_backbone_config(const json::Value & value) {
    MossTtsDelayBackboneConfig config;
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
    config.tie_word_embeddings =
        json::optional_bool(value, "tie_word_embeddings", config.tie_word_embeddings);
    engine::io::require_positive(config.hidden_size, "backbone hidden_size");
    engine::io::require_positive(config.intermediate_size, "backbone intermediate_size");
    engine::io::require_positive(config.num_hidden_layers, "backbone num_hidden_layers");
    engine::io::require_positive(config.num_key_value_heads, "backbone num_key_value_heads");
    engine::io::require_positive(config.head_dim, "backbone head_dim");
    engine::io::require_positive(config.vocab_size, "backbone vocab_size");
    return config;
}

}  // namespace

MossTtsDelayConfig parse_moss_tts_delay_config(const json::Value & root, std::string_view model_label) {
    const std::string label(model_label);
    const auto model_type = json::optional_string(root, "model_type", "");
    if (model_type != "moss_tts_delay") {
        throw std::runtime_error(
            label + " config model_type mismatch: expected moss_tts_delay, got " + model_type);
    }
    MossTtsDelayConfig config;
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
    config.audio_assistant_gen_slot_token_id =
        json::require_i64(root, "audio_assistant_gen_slot_token_id");
    config.audio_assistant_delay_slot_token_id =
        json::require_i64(root, "audio_assistant_delay_slot_token_id");
    config.sampling_rate = json::optional_i64(root, "sampling_rate", 24000);
    engine::io::require_positive(config.num_codebooks, "n_vq");
    engine::io::require_positive(config.audio_vocab_size, "audio_vocab_size");
    // The checkpoint carries one embedding table and one head per codebook, and the pad
    // code is the entry past the audio vocabulary. A config that disagrees would leave
    // the generator without a usable length bound, so refuse it rather than half-load.
    if (config.audio_pad_code != config.audio_vocab_size) {
        throw std::runtime_error(
            label + " expects audio_pad_code to be the code past audio_vocab_size");
    }
    return config;
}

}  // namespace engine::decoders
