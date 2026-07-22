#include "engine/models/higgs_audio_tts/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/config.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <string>

namespace engine::models::higgs_audio_tts {
namespace json = engine::io::json;
namespace {

constexpr const char * kExpectedArchitecture = "HiggsMultimodalQwen3ForConditionalGeneration";

float parse_rope_theta(const engine::io::json::Value & text_config) {
    const auto * rope_parameters = text_config.find("rope_parameters");
    if (rope_parameters != nullptr && rope_parameters->is_object()) {
        return json::optional_f32(*rope_parameters, "rope_theta", 1000000.0F);
    }
    const auto * rope_theta = text_config.find("rope_theta");
    if (rope_theta != nullptr && rope_theta->is_number()) {
        return rope_theta->as_f32();
    }
    return 1000000.0F;
}

std::string parse_architecture(const engine::io::json::Value & root) {
    const auto * architectures = root.find("architectures");
    if (architectures == nullptr || !architectures->is_array() || architectures->as_array().empty()) {
        throw std::runtime_error("Higgs TTS config must provide architectures[0]");
    }
    return architectures->as_array().front().as_string();
}

HiggsTextConfig parse_text_config(const engine::io::json::Value & root) {
    HiggsTextConfig config;
    config.model_type = json::require_string(root, "model_type");
    if (config.model_type != "qwen3") {
        throw std::runtime_error("Higgs TTS text_config.model_type mismatch");
    }
    config.vocab_size = json::require_i64(root, "vocab_size");
    config.hidden_size = json::require_i64(root, "hidden_size");
    config.intermediate_size = json::require_i64(root, "intermediate_size");
    config.num_hidden_layers = json::require_i64(root, "num_hidden_layers");
    config.num_attention_heads = json::require_i64(root, "num_attention_heads");
    config.num_key_value_heads = json::require_i64(root, "num_key_value_heads");
    config.head_dim = json::optional_i64(root, "head_dim", config.hidden_size / config.num_attention_heads);
    config.max_position_embeddings = json::require_i64(root, "max_position_embeddings");
    config.bos_token_id = json::require_i64(root, "bos_token_id");
    config.eos_token_id = json::require_i64(root, "eos_token_id");
    config.pad_token_id = json::optional_nullable_i64(root, "pad_token_id", -1);
    config.rms_norm_eps = json::optional_f32(root, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = parse_rope_theta(root);
    config.tie_word_embeddings = json::optional_bool(root, "tie_word_embeddings", config.tie_word_embeddings);

    engine::io::require_positive(config.vocab_size, "text vocab_size");
    engine::io::require_positive(config.hidden_size, "text hidden_size");
    engine::io::require_positive(config.intermediate_size, "text intermediate_size");
    engine::io::require_positive(config.num_hidden_layers, "text num_hidden_layers");
    engine::io::require_positive(config.num_attention_heads, "text num_attention_heads");
    engine::io::require_positive(config.num_key_value_heads, "text num_key_value_heads");
    engine::io::require_positive(config.head_dim, "text head_dim");
    engine::io::require_positive(config.max_position_embeddings, "text max_position_embeddings");
    engine::io::require_divisible(config.num_attention_heads, config.num_key_value_heads, "text grouped-query attention");
    return config;
}

HiggsAudioEncoderConfig parse_audio_config(const engine::io::json::Value & root) {
    HiggsAudioEncoderConfig config;
    config.model_type = json::require_string(root, "model_type");
    if (config.model_type != "higgs_audio_encoder") {
        throw std::runtime_error("Higgs TTS audio_encoder_config.model_type mismatch");
    }
    config.encoder_type = json::require_string(root, "encoder_type");
    if (config.encoder_type != "discrete") {
        throw std::runtime_error("Higgs TTS audio_encoder_config.encoder_type mismatch");
    }
    config.num_codebooks = json::require_i64(root, "num_codebooks");
    config.vocab_size = json::require_i64(root, "vocab_size");
    config.out_dim = json::require_i64(root, "out_dim");
    config.mel_per_sample = json::require_i64(root, "mel_per_sample");
    config.max_chunk_size = json::require_i64(root, "max_chunk_size");
    config.tie_word_embeddings = json::optional_bool(root, "tie_word_embeddings", config.tie_word_embeddings);
    config.use_delay_pattern = json::optional_bool(root, "use_delay_pattern", config.use_delay_pattern);

    engine::io::require_positive(config.num_codebooks, "audio num_codebooks");
    engine::io::require_positive(config.vocab_size, "audio vocab_size");
    engine::io::require_positive(config.out_dim, "audio out_dim");
    engine::io::require_positive(config.mel_per_sample, "audio mel_per_sample");
    engine::io::require_positive(config.max_chunk_size, "audio max_chunk_size");
    if (!config.tie_word_embeddings) {
        throw std::runtime_error("Higgs TTS currently expects tied modality embedding/head weights");
    }
    if (!config.use_delay_pattern) {
        throw std::runtime_error("Higgs TTS v3 requires delay-pattern audio codebooks");
    }
    return config;
}

HiggsConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    HiggsConfig config;
    config.model_type = json::require_string(root, "model_type");
    if (config.model_type != "higgs_multimodal_qwen3") {
        throw std::runtime_error("Higgs TTS model_type mismatch");
    }
    config.architecture = parse_architecture(root);
    if (config.architecture != kExpectedArchitecture) {
        throw std::runtime_error("Higgs TTS architecture mismatch");
    }
    config.hidden_size = json::require_i64(root, "_hidden_size");
    config.vocab_size = json::require_i64(root, "_vocab_size");
    config.audio_token_id = json::optional_i64(root, "audio_token_id", config.audio_token_id);
    config.ignore_index = json::optional_i64(root, "ignore_index", config.ignore_index);
    config.text = parse_text_config(root.require("text_config"));
    config.audio = parse_audio_config(root.require("audio_encoder_config"));

    if (config.hidden_size != config.text.hidden_size) {
        throw std::runtime_error("Higgs TTS _hidden_size must match text hidden_size");
    }
    if (config.vocab_size != config.text.vocab_size) {
        throw std::runtime_error("Higgs TTS _vocab_size must match text vocab_size");
    }
    if (config.audio.out_dim != config.text.hidden_size) {
        throw std::runtime_error("Higgs TTS audio out_dim must match text hidden_size");
    }
    if (config.audio_token_id != -100) {
        throw std::runtime_error("Higgs TTS audio_token_id mismatch");
    }
    return config;
}

void validate_weight_anchors(const HiggsAssets & assets) {
    const auto & config = assets.config;
    const auto & weights = *assets.weights;
    const int64_t hidden = config.text.hidden_size;
    const int64_t audio_fused_vocab = config.audio.num_codebooks * config.audio.vocab_size;
    assets::require_tensor_shape(weights, "tied.embedding.text_embedding.weight", {config.text.vocab_size, hidden});
    assets::require_tensor_shape(weights, "tied.embedding.modality_embeddings.0.embedding.weight", {audio_fused_vocab, hidden});
    assets::require_tensor_shape(weights, "body.norm.weight", {hidden});
    assets::require_tensor_shape(weights, "body.layers.0.input_layernorm.weight", {hidden});
    assets::require_tensor_shape(weights, "body.layers.0.post_attention_layernorm.weight", {hidden});
    assets::require_tensor_shape(weights, "body.layers.0.self_attn.q_proj.weight", {config.text.num_attention_heads * config.text.head_dim, hidden});
    assets::require_tensor_shape(weights, "body.layers.0.self_attn.k_proj.weight", {config.text.num_key_value_heads * config.text.head_dim, hidden});
    assets::require_tensor_shape(weights, "body.layers.0.self_attn.v_proj.weight", {config.text.num_key_value_heads * config.text.head_dim, hidden});
    assets::require_tensor_shape(weights, "body.layers.0.self_attn.o_proj.weight", {hidden, config.text.num_attention_heads * config.text.head_dim});
    assets::require_tensor_shape(weights, "body.layers.0.self_attn.q_norm.weight", {config.text.head_dim});
    assets::require_tensor_shape(weights, "body.layers.0.self_attn.k_norm.weight", {config.text.head_dim});
    assets::require_tensor_shape(weights, "body.layers.0.mlp.gate_proj.weight", {config.text.intermediate_size, hidden});
    assets::require_tensor_shape(weights, "body.layers.0.mlp.up_proj.weight", {config.text.intermediate_size, hidden});
    assets::require_tensor_shape(weights, "body.layers.0.mlp.down_proj.weight", {hidden, config.text.intermediate_size});
    assets::require_tensor_shape(weights, "tied.embedding.modality_embeddings.0.model.acoustic_encoder.conv1.weight", {64, 1, 7});
    assets::require_tensor_shape(weights, "tied.embedding.modality_embeddings.0.model.acoustic_decoder.conv2.weight", {1, 32, 7});
    assets::require_tensor_shape(weights, "tied.embedding.modality_embeddings.0.model.quantizer.quantizers.0.codebook.embed", {1024, 64});
}

}  // namespace

std::shared_ptr<const HiggsAssets> load_higgs_assets(const std::filesystem::path & model_path) {
    HiggsAssets assets;
    assets.resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("higgs_audio_tts"));
    assets.config = parse_config(assets.resources);
    assets.weights = assets.resources.open_tensor_source("weights");
    validate_weight_anchors(assets);
    return std::make_shared<HiggsAssets>(std::move(assets));
}

}  // namespace engine::models::higgs_audio_tts
