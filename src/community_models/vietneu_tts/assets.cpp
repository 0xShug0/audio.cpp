#include "engine/community_models/vietneu_tts/assets.h"

#include "engine/framework/assets/model_package.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <utility>

namespace engine::models::vietneu_tts {
namespace json = engine::io::json;
namespace {

int64_t parse_generation_max_new_tokens(const assets::ResourceBundle & resources) {
    if (!resources.has_file("generation_config")) {
        return 2048;
    }
    const auto generation = resources.parse_json("generation_config");
    return json::optional_i64(generation, "max_new_tokens", 2048);
}

VietneuTTSConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    VietneuTTSConfig config;

    config.is_vieneu = true;
    config.tts_model_type = "base";
    config.variant = VietneuTTSVariant::Base;
    config.tts_model_size = json::optional_string(root, "tts_model_size", "1.7B");
    config.tokenizer_type = "vietneu_tts_tokenizer_12hz";
    config.max_new_tokens = parse_generation_max_new_tokens(resources);
    config.tts_bos_token_id = json::optional_i64(root, "bos_token_id", 1);
    config.tts_eos_token_id = json::optional_i64(root, "eos_token_id", 2);
    config.tts_pad_token_id = json::optional_i64(root, "pad_token_id", 0);
    config.text_prompt_start_token_id = json::optional_i64(root, "text_prompt_start_token_id", 3);
    config.text_prompt_end_token_id = json::optional_i64(root, "text_prompt_end_token_id", 4);
    config.audio_ref_slot_token_id = json::optional_i64(root, "audio_ref_slot_token_id", 7);
    config.audio_pad_token_id = json::optional_i64(root, "audio_pad_token_id", 1024);
    config.speech_generation_start_token_id = json::optional_i64(root, "speech_generation_start_token_id", 5);
    config.local_num_hidden_layers = json::optional_i64(root, "local_num_hidden_layers", 1);
    config.local_num_attention_heads = json::optional_i64(root, "local_num_attention_heads", 8);
    config.local_intermediate_size = json::optional_i64(root, "local_intermediate_size", 2048);
    config.use_speaker_embedding = json::optional_bool(root, "use_speaker_embedding", false);
    config.speaker_embedding_dim = json::optional_i64(root, "speaker_embedding_dim", 192);

    // Talker
    config.talker.hidden_size = json::require_i64(root, "hidden_size");
    config.talker.text_hidden_size = config.talker.hidden_size;
    config.talker.text_vocab_size = json::require_i64(root, "text_vocab_size");
    config.talker.intermediate_size = json::require_i64(root, "intermediate_size");
    config.talker.num_hidden_layers = json::require_i64(root, "num_hidden_layers");
    config.talker.num_attention_heads = json::require_i64(root, "num_attention_heads");
    config.talker.num_key_value_heads = json::require_i64(root, "num_key_value_heads");
    config.talker.head_dim = json::optional_i64(root, "head_dim", config.talker.hidden_size / config.talker.num_attention_heads);
    config.talker.vocab_size = json::require_i64(root, "audio_vocab_size");
    config.talker.num_code_groups = json::require_i64(root, "n_vq");
    config.talker.codec_bos_id = 0;
    config.talker.codec_eos_token_id = json::optional_i64(root, "speech_generation_end_token_id", 6);
    config.talker.codec_think_id = 0;
    config.talker.codec_nothink_id = 0;
    config.talker.codec_pad_id = json::optional_i64(root, "audio_pad_token_id", 1024);
    config.talker.codec_think_bos_id = 0;
    config.talker.codec_think_eos_id = 0;
    config.talker.rope_theta = json::optional_f32(root, "rope_theta", config.talker.rope_theta);
    config.talker.rms_norm_eps = json::optional_f32(root, "rms_norm_eps", config.talker.rms_norm_eps);

    // Code predictor / Acoustic decoder
    config.code_predictor.hidden_size = config.talker.hidden_size;
    config.code_predictor.intermediate_size = config.local_intermediate_size;
    config.code_predictor.num_hidden_layers = config.local_num_hidden_layers;
    config.code_predictor.num_attention_heads = config.local_num_attention_heads;
    config.code_predictor.num_key_value_heads = config.local_num_attention_heads;
    config.code_predictor.head_dim = config.code_predictor.hidden_size / config.code_predictor.num_attention_heads;
    config.code_predictor.vocab_size = config.talker.vocab_size;
    config.code_predictor.rope_theta = json::optional_f32(root, "rope_theta", config.code_predictor.rope_theta);
    config.code_predictor.rms_norm_eps = json::optional_f32(root, "rms_norm_eps", config.code_predictor.rms_norm_eps);

    // Speech tokenizer
    config.speech_tokenizer.model_type = "vietneu_tts_tokenizer_12hz";
    config.speech_tokenizer.input_sample_rate = 48000;
    config.speech_tokenizer.output_sample_rate = 48000;
    config.speech_tokenizer.num_quantizers = config.talker.num_code_groups;
    config.speech_tokenizer.codebook_size = config.talker.vocab_size;

    config.has_speaker_encoder = false; // Bypassed for VieNeu-TTS v3 Turbo
    return config;
}

void validate_config(const VietneuTTSConfig & config) {
    if (config.tokenizer_type != "vietneu_tts_tokenizer_12hz") {
        throw std::runtime_error("Vietneu TTS currently supports vietneu_tts_tokenizer_12hz");
    }
}

}  // namespace

std::shared_ptr<const VietneuTTSAssets> load_vietneu_tts_assets(const std::filesystem::path & model_path) {
    auto resources = assets::load_resource_bundle_from_package_spec(
        model_path,
        assets::default_model_package_spec_path("vietneu_tts"));
    auto assets = std::make_shared<VietneuTTSAssets>();
    assets->config = parse_config(resources);
    validate_config(assets->config);
    assets->model_weights = resources.open_tensor_source("model_weights");
    assets->speech_tokenizer_weights = resources.open_tensor_source("speech_tokenizer_weights");
    assets->resources = std::move(resources);
    return assets;
}

}  // namespace engine::models::vietneu_tts
