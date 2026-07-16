#include "engine/models/qwen3_tts/assets.h"

#include "engine/framework/assets/model_package.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <utility>

namespace engine::models::qwen3_tts {
namespace json = engine::io::json;
namespace {

Qwen3TTSVariant parse_variant(const std::string & value) {
    if (value == "base") {
        return Qwen3TTSVariant::Base;
    }
    if (value == "voice_design") {
        return Qwen3TTSVariant::VoiceDesign;
    }
    if (value == "custom_voice") {
        return Qwen3TTSVariant::CustomVoice;
    }
    throw std::runtime_error("Qwen3 TTS unsupported tts_model_type: " + value);
}

Qwen3TTSCodePredictorConfig parse_code_predictor_config(const json::Value & value) {
    Qwen3TTSCodePredictorConfig config;
    config.hidden_size = json::require_i64(value, "hidden_size");
    config.intermediate_size = json::require_i64(value, "intermediate_size");
    config.num_hidden_layers = json::require_i64(value, "num_hidden_layers");
    config.num_attention_heads = json::require_i64(value, "num_attention_heads");
    config.num_key_value_heads = json::require_i64(value, "num_key_value_heads");
    config.head_dim = json::optional_i64(value, "head_dim", config.hidden_size / config.num_attention_heads);
    config.vocab_size = json::require_i64(value, "vocab_size");
    config.rms_norm_eps = json::optional_f32(value, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = json::optional_f32(value, "rope_theta", config.rope_theta);
    return config;
}

Qwen3TTSTalkerConfig parse_talker_config(const json::Value & value) {
    Qwen3TTSTalkerConfig config;
    config.max_position_embeddings = json::optional_i64(value, "max_position_embeddings", config.max_position_embeddings);
    config.hidden_size = json::require_i64(value, "hidden_size");
    config.text_hidden_size = json::require_i64(value, "text_hidden_size");
    config.text_vocab_size = json::require_i64(value, "text_vocab_size");
    config.intermediate_size = json::require_i64(value, "intermediate_size");
    config.num_hidden_layers = json::require_i64(value, "num_hidden_layers");
    config.num_attention_heads = json::require_i64(value, "num_attention_heads");
    config.num_key_value_heads = json::require_i64(value, "num_key_value_heads");
    config.head_dim = json::optional_i64(value, "head_dim", config.hidden_size / config.num_attention_heads);
    config.num_code_groups = json::require_i64(value, "num_code_groups");
    config.vocab_size = json::require_i64(value, "vocab_size");
    config.codec_bos_id = json::require_i64(value, "codec_bos_id");
    config.codec_eos_token_id = json::require_i64(value, "codec_eos_token_id");
    config.codec_think_id = json::require_i64(value, "codec_think_id");
    config.codec_nothink_id = json::require_i64(value, "codec_nothink_id");
    config.codec_pad_id = json::require_i64(value, "codec_pad_id");
    config.codec_think_bos_id = json::require_i64(value, "codec_think_bos_id");
    config.codec_think_eos_id = json::require_i64(value, "codec_think_eos_id");
    const auto * language_ids = value.find("codec_language_id");
    if (language_ids != nullptr) {
        for (const auto & [name, id] : language_ids->as_object()) {
            config.codec_language_id.emplace(name, id.as_i64());
        }
    }
    const auto * speaker_ids = value.find("spk_id");
    if (speaker_ids != nullptr) {
        for (const auto & [name, id] : speaker_ids->as_object()) {
            config.speaker_id.emplace(name, id.as_i64());
        }
    }
    const auto * speaker_dialects = value.find("spk_is_dialect");
    if (speaker_dialects != nullptr) {
        for (const auto & [name, dialect] : speaker_dialects->as_object()) {
            if (dialect.is_bool()) {
                if (dialect.as_bool()) {
                    throw std::runtime_error("Qwen3 TTS speaker dialect must be false or a string for speaker: " + name);
                }
                config.speaker_dialect.emplace(name, std::nullopt);
            } else if (dialect.is_string()) {
                config.speaker_dialect.emplace(name, dialect.as_string());
            } else {
                throw std::runtime_error("Qwen3 TTS speaker dialect must be false or a string for speaker: " + name);
            }
        }
    }
    config.rms_norm_eps = json::optional_f32(value, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = json::optional_f32(value, "rope_theta", config.rope_theta);
    return config;
}

Qwen3TTSSpeechTokenizerConfig parse_speech_tokenizer_config(const json::Value & value) {
    Qwen3TTSSpeechTokenizerConfig config;
    config.model_type = json::require_string(value, "model_type");
    config.input_sample_rate = static_cast<int>(json::require_i64(value, "input_sample_rate"));
    config.output_sample_rate = static_cast<int>(json::require_i64(value, "output_sample_rate"));
    const auto & decoder = value.require("decoder_config");
    config.num_quantizers = json::require_i64(decoder, "num_quantizers");
    config.codebook_size = json::require_i64(decoder, "codebook_size");
    config.semantic_codebook_size = json::require_i64(decoder, "semantic_codebook_size");
    return config;
}

Qwen3TTSSpeakerEncoderConfig parse_speaker_encoder_config(const json::Value & value) {
    Qwen3TTSSpeakerEncoderConfig config;
    config.embedding_dim = json::optional_i64(value, "enc_dim", 1024);
    config.sample_rate = static_cast<int>(json::optional_i64(value, "sample_rate", 24000));
    return config;
}

int64_t parse_generation_max_new_tokens(const assets::ResourceBundle & resources) {
    const auto generation = resources.parse_json("generation_config");
    return json::optional_i64(generation, "max_new_tokens", 2048);
}

Qwen3TTSConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    Qwen3TTSConfig config;
    config.tts_model_type = json::require_string(root, "tts_model_type");
    config.variant = parse_variant(config.tts_model_type);
    config.tts_model_size = json::require_string(root, "tts_model_size");
    config.tokenizer_type = json::require_string(root, "tokenizer_type");
    config.max_new_tokens = parse_generation_max_new_tokens(resources);
    config.talker = parse_talker_config(root.require("talker_config"));
    config.code_predictor = parse_code_predictor_config(root.require("talker_config").require("code_predictor_config"));
    config.speech_tokenizer = parse_speech_tokenizer_config(resources.parse_json("speech_tokenizer_config"));
    config.tts_bos_token_id = json::require_i64(root, "tts_bos_token_id");
    config.tts_eos_token_id = json::require_i64(root, "tts_eos_token_id");
    config.tts_pad_token_id = json::require_i64(root, "tts_pad_token_id");
    config.has_speaker_encoder = root.find("speaker_encoder_config") != nullptr;
    if (config.has_speaker_encoder) {
        config.speaker_encoder = parse_speaker_encoder_config(root.require("speaker_encoder_config"));
    }
    return config;
}

void validate_config(const Qwen3TTSConfig & config) {
    if (config.tokenizer_type != "qwen3_tts_tokenizer_12hz") {
        throw std::runtime_error("Qwen3 TTS currently supports qwen3_tts_tokenizer_12hz");
    }
    if (config.variant == Qwen3TTSVariant::Base && !config.has_speaker_encoder) {
        throw std::runtime_error("Qwen3 base TTS model must provide speaker_encoder_config for voice clone");
    }
    if (config.variant == Qwen3TTSVariant::CustomVoice && config.talker.speaker_id.empty()) {
        throw std::runtime_error("Qwen3 custom voice model must provide speaker ids");
    }
    if (config.speech_tokenizer.model_type != "qwen3_tts_tokenizer_12hz") {
        throw std::runtime_error("Qwen3 TTS speech tokenizer must be qwen3_tts_tokenizer_12hz");
    }
}

}  // namespace

std::shared_ptr<const Qwen3TTSAssets> load_qwen3_tts_assets(const std::filesystem::path & model_path) {
    auto resources = assets::load_resource_bundle_from_package_spec(
        model_path,
        assets::default_model_package_spec_path("qwen3_tts"));
    auto assets = std::make_shared<Qwen3TTSAssets>();
    assets->config = parse_config(resources);
    validate_config(assets->config);
    assets->model_weights = resources.open_tensor_source("model_weights");
    assets->speech_tokenizer_weights = resources.open_tensor_source("speech_tokenizer_weights");
    assets->resources = std::move(resources);
    return assets;
}

}  // namespace engine::models::qwen3_tts
