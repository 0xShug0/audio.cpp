#include "engine/models/audio_flamingo/assets.h"

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <utility>

namespace engine::models::audio_flamingo {
namespace json = engine::io::json;
namespace {

AudioFlamingoAudioEncoderConfig parse_audio_config(const engine::io::json::Value & value) {
    AudioFlamingoAudioEncoderConfig config;
    config.num_mel_bins = value.require("num_mel_bins").as_i64();
    config.num_hidden_layers = value.require("num_hidden_layers").as_i64();
    config.num_attention_heads = value.require("num_attention_heads").as_i64();
    config.intermediate_size = value.require("intermediate_size").as_i64();
    config.hidden_size = value.require("hidden_size").as_i64();
    config.max_source_positions = value.require("max_source_positions").as_i64();
    config.activation_function = json::optional_string(value, "activation_function", config.activation_function);
    if (config.activation_function != "gelu") {
        throw std::runtime_error("Audio Flamingo audio encoder currently expects gelu activation");
    }
    return config;
}

AudioFlamingoTextDecoderConfig parse_text_config(
    const engine::io::json::Value & root,
    const engine::io::json::Value & text_config,
    const engine::io::json::Value & tokenizer_config,
    const engine::io::json::Value & generation_config) {
    AudioFlamingoTextDecoderConfig config;
    config.vocab_size = text_config.require("vocab_size").as_i64();
    config.hidden_size = text_config.require("hidden_size").as_i64();
    config.intermediate_size = text_config.require("intermediate_size").as_i64();
    config.num_hidden_layers = text_config.require("num_hidden_layers").as_i64();
    config.num_attention_heads = text_config.require("num_attention_heads").as_i64();
    config.num_key_value_heads = text_config.require("num_key_value_heads").as_i64();
    config.head_dim = json::optional_i64(text_config, "head_dim", config.hidden_size / config.num_attention_heads);
    config.max_position_embeddings = text_config.require("max_position_embeddings").as_i64();
    config.audio_token_id = root.require("audio_token_id").as_i64();
    config.audio_bos_token_id = json::optional_i64(root, "audio_bos_token_id", config.audio_bos_token_id);
    config.audio_eos_token_id = json::optional_i64(root, "audio_eos_token_id", config.audio_eos_token_id);
    config.rms_norm_eps = json::optional_f32(text_config, "rms_norm_eps", config.rms_norm_eps);
    if (const auto * rope = text_config.find("rope_parameters")) {
        config.rope_theta = json::optional_f32(*rope, "rope_theta", config.rope_theta);
    } else {
        config.rope_theta = json::optional_f32(text_config, "rope_theta", config.rope_theta);
    }
    config.pad_token_id = json::optional_i64(generation_config, "pad_token_id", config.pad_token_id);
    config.eos_token_ids = json::require_i64_array_or_scalar(generation_config, "eos_token_id");
    if (config.pad_token_id == 0) {
        const auto * pad_token = tokenizer_config.find("pad_token_id");
        if (pad_token != nullptr && pad_token->is_number()) {
            config.pad_token_id = pad_token->as_i64();
        }
    }
    return config;
}

AudioFlamingoFrontendConfig parse_frontend_config(
    const engine::io::json::Value & processor_config,
    const AudioFlamingoAudioEncoderConfig & audio_config,
    AudioFlamingoVariant variant) {
    AudioFlamingoFrontendConfig config;
    const int64_t default_max_audio_length = variant == AudioFlamingoVariant::Next ? 1800 : 600;
    config.max_audio_length_sec = json::optional_i64(processor_config, "max_audio_len", default_max_audio_length);
    const auto & frontend = processor_config.require("feature_extractor");
    config.sample_rate = static_cast<int>(frontend.require("sampling_rate").as_i64());
    config.feature_size = frontend.require("feature_size").as_i64();
    config.hop_length = frontend.require("hop_length").as_i64();
    config.n_fft = frontend.require("n_fft").as_i64();
    config.chunk_length_sec = frontend.require("chunk_length").as_i64();
    if (config.feature_size != audio_config.num_mel_bins) {
        throw std::runtime_error("Audio Flamingo processor and encoder mel dimensions differ");
    }
    return config;
}

AudioFlamingoRoTEConfig parse_rote_config(const engine::io::json::Value & root) {
    AudioFlamingoRoTEConfig config;
    config.max_position_embeddings = root.require("max_position_embeddings").as_i64();
    const auto & rope = root.require("rope_parameters");
    config.rope_theta = json::optional_f32(rope, "rope_theta", config.rope_theta);
    config.partial_rotary_factor = json::optional_f32(rope, "partial_rotary_factor", config.partial_rotary_factor);
    config.audio_frame_step = json::optional_f32(root, "audio_frame_step", config.audio_frame_step);
    return config;
}

AudioFlamingoConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    const auto tokenizer_config = resources.parse_json("tokenizer_config");
    const auto processor_config = resources.parse_json("processor_config");
    const auto generation_config = resources.parse_json("generation_config");

    AudioFlamingoConfig config;
    config.model_type = root.require("model_type").as_string();
    if (config.model_type == "audioflamingo3") {
        config.variant = AudioFlamingoVariant::V3;
    } else if (config.model_type == "musicflamingo") {
        config.variant = AudioFlamingoVariant::Next;
    } else {
        throw std::runtime_error("unsupported Audio Flamingo model_type: " + config.model_type);
    }
    config.projector_bias = json::optional_bool(root, "projector_bias", config.projector_bias);
    config.projector_hidden_act = json::optional_string(root, "projector_hidden_act", config.projector_hidden_act);
    if (config.projector_hidden_act != "gelu") {
        throw std::runtime_error("Audio Flamingo projector currently expects gelu activation");
    }
    config.audio_encoder = parse_audio_config(root.require("audio_config"));
    config.frontend = parse_frontend_config(processor_config, config.audio_encoder, config.variant);
    config.text_decoder = parse_text_config(root, root.require("text_config"), tokenizer_config, generation_config);
    if (config.variant == AudioFlamingoVariant::Next) {
        config.rote = parse_rote_config(root);
    }
    config.max_new_tokens = json::optional_i64(generation_config, "max_new_tokens", config.max_new_tokens);
    return config;
}

}  // namespace

std::shared_ptr<const AudioFlamingoAssets> load_audio_flamingo_assets(const std::filesystem::path & model_path) {
    auto resources = model_spec::load_resource_bundle_for_family(model_path, "audio_flamingo");
    AudioFlamingoAssets assets;
    assets.config = parse_config(resources);
    assets.model_weights = resources.open_tensor_source("weights");
    assets.resources = std::move(resources);
    return std::make_shared<AudioFlamingoAssets>(std::move(assets));
}

}  // namespace engine::models::audio_flamingo
