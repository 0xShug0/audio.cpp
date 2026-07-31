#include "engine/models/qwen3_tts/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/json.h"

#include <cmath>
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

int64_t parse_head_dim(
    const json::Value & value,
    int64_t hidden_size,
    int64_t attention_heads,
    const char * component) {
    if (const auto * head_dim = value.find("head_dim")) {
        return head_dim->as_i64();
    }
    if (hidden_size <= 0 || attention_heads <= 0) {
        throw std::runtime_error(std::string("Qwen3 TTS ") + component +
            " hidden_size and num_attention_heads must be positive");
    }
    if (hidden_size % attention_heads != 0) {
        throw std::runtime_error(std::string("Qwen3 TTS ") + component +
            " hidden_size must be divisible by num_attention_heads when head_dim is omitted");
    }
    return hidden_size / attention_heads;
}

Qwen3TTSCodePredictorConfig parse_code_predictor_config(const json::Value & value) {
    Qwen3TTSCodePredictorConfig config;
    config.hidden_size = json::require_i64(value, "hidden_size");
    config.intermediate_size = json::require_i64(value, "intermediate_size");
    config.num_hidden_layers = json::require_i64(value, "num_hidden_layers");
    config.num_attention_heads = json::require_i64(value, "num_attention_heads");
    config.num_key_value_heads = json::require_i64(value, "num_key_value_heads");
    config.head_dim = parse_head_dim(
        value,
        config.hidden_size,
        config.num_attention_heads,
        "code predictor");
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
    config.head_dim = parse_head_dim(
        value,
        config.hidden_size,
        config.num_attention_heads,
        "talker");
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

void validate_transformer_shape(
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t layers,
    int64_t attention_heads,
    int64_t key_value_heads,
    int64_t head_dim,
    const char * component) {
    if (hidden_size <= 0 || intermediate_size <= 0 || layers <= 0 ||
        attention_heads <= 0 || key_value_heads <= 0 || head_dim <= 0) {
        throw std::runtime_error(std::string("Qwen3 TTS ") + component +
            " transformer dimensions must be positive");
    }
    if (attention_heads % key_value_heads != 0) {
        throw std::runtime_error(std::string("Qwen3 TTS ") + component +
            " num_attention_heads must be divisible by num_key_value_heads");
    }
}

void validate_config_impl(const Qwen3TTSConfig & config) {
    if (config.tokenizer_type != "qwen3_tts_tokenizer_12hz") {
        throw std::runtime_error("Qwen3 TTS currently supports qwen3_tts_tokenizer_12hz");
    }
    if (config.max_new_tokens <= 0) {
        throw std::runtime_error("Qwen3 TTS max_new_tokens must be positive");
    }
    validate_transformer_shape(
        config.talker.hidden_size,
        config.talker.intermediate_size,
        config.talker.num_hidden_layers,
        config.talker.num_attention_heads,
        config.talker.num_key_value_heads,
        config.talker.head_dim,
        "talker");
    if (config.talker.max_position_embeddings <= 0 || config.talker.text_hidden_size <= 0 ||
        config.talker.text_vocab_size <= 0 || config.talker.num_code_groups <= 1 ||
        config.talker.vocab_size <= 0) {
        throw std::runtime_error("Qwen3 TTS talker sizes must be positive and include multiple code groups");
    }
    if (!std::isfinite(config.talker.rms_norm_eps) || config.talker.rms_norm_eps <= 0.0F ||
        !std::isfinite(config.talker.rope_theta) || config.talker.rope_theta <= 0.0F) {
        throw std::runtime_error("Qwen3 TTS talker norm and RoPE parameters must be finite and positive");
    }
    validate_transformer_shape(
        config.code_predictor.hidden_size,
        config.code_predictor.intermediate_size,
        config.code_predictor.num_hidden_layers,
        config.code_predictor.num_attention_heads,
        config.code_predictor.num_key_value_heads,
        config.code_predictor.head_dim,
        "code predictor");
    if (config.code_predictor.vocab_size <= 0 ||
        !std::isfinite(config.code_predictor.rms_norm_eps) || config.code_predictor.rms_norm_eps <= 0.0F ||
        !std::isfinite(config.code_predictor.rope_theta) || config.code_predictor.rope_theta <= 0.0F) {
        throw std::runtime_error("Qwen3 TTS code predictor vocabulary, norm, and RoPE parameters must be positive");
    }
    if (config.speech_tokenizer.model_type != "qwen3_tts_tokenizer_12hz") {
        throw std::runtime_error("Qwen3 TTS speech tokenizer must be qwen3_tts_tokenizer_12hz");
    }
    if (config.speech_tokenizer.input_sample_rate <= 0 ||
        config.speech_tokenizer.output_sample_rate <= 0 ||
        config.speech_tokenizer.num_quantizers <= 0 ||
        config.speech_tokenizer.codebook_size <= 0 ||
        config.speech_tokenizer.semantic_codebook_size <= 0) {
        throw std::runtime_error("Qwen3 TTS speech tokenizer sizes and sample rates must be positive");
    }
    if (config.talker.num_code_groups != config.speech_tokenizer.num_quantizers) {
        throw std::runtime_error("Qwen3 TTS talker code groups must match speech tokenizer quantizers");
    }
    if (config.variant == Qwen3TTSVariant::Base && !config.has_speaker_encoder) {
        throw std::runtime_error("Qwen3 base TTS model must provide speaker_encoder_config for voice clone");
    }
    if (config.has_speaker_encoder &&
        (config.speaker_encoder.embedding_dim <= 0 || config.speaker_encoder.sample_rate <= 0)) {
        throw std::runtime_error("Qwen3 TTS speaker encoder dimensions and sample rate must be positive");
    }
    if (config.has_speaker_encoder && config.speaker_encoder.embedding_dim != config.talker.hidden_size) {
        throw std::runtime_error("Qwen3 TTS speaker encoder embedding_dim must match talker hidden_size");
    }
    if (config.variant == Qwen3TTSVariant::CustomVoice && config.talker.speaker_id.empty()) {
        throw std::runtime_error("Qwen3 custom voice model must provide speaker ids");
    }
}

}  // namespace

void validate_qwen3_tts_config(const Qwen3TTSConfig & config) {
    validate_config_impl(config);
}

std::shared_ptr<const Qwen3TTSAssets> load_qwen3_tts_assets(const std::filesystem::path & model_path) {
    auto resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("qwen3_tts"));
    auto assets = std::make_shared<Qwen3TTSAssets>();
    assets->config = parse_config(resources);
    validate_qwen3_tts_config(assets->config);
    assets->model_weights = resources.open_tensor_source("model_weights");
    assets->speech_tokenizer_weights = resources.open_tensor_source("speech_tokenizer_weights");
    assets->resources = std::move(resources);
    return assets;
}

}  // namespace engine::models::qwen3_tts
