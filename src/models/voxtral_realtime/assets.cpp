#include "engine/models/voxtral_realtime/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <utility>

namespace engine::models::voxtral_realtime {
namespace json = engine::io::json;
namespace {

VoxtralRealtimeAudioConfig parse_audio_config(const engine::io::json::Value & value) {
    VoxtralRealtimeAudioConfig config;
    config.model_type = json::require_string(value, "model_type");
    config.activation_function = json::require_string(value, "activation_function");
    config.hidden_act = json::require_string(value, "hidden_act");
    config.hidden_size = json::require_i64(value, "hidden_size");
    config.intermediate_size = json::require_i64(value, "intermediate_size");
    config.num_hidden_layers = json::require_i64(value, "num_hidden_layers");
    config.num_attention_heads = json::require_i64(value, "num_attention_heads");
    config.num_key_value_heads = json::require_i64(value, "num_key_value_heads");
    config.head_dim = json::require_i64(value, "head_dim");
    config.num_mel_bins = json::require_i64(value, "num_mel_bins");
    config.max_position_embeddings = json::require_i64(value, "max_position_embeddings");
    config.sliding_window = json::require_i64(value, "sliding_window");
    config.vocab_size = json::require_i64(value, "vocab_size");
    config.rms_norm_eps = json::optional_f32(value, "rms_norm_eps", config.rms_norm_eps);
    if (const auto * parameters = value.find("rope_parameters"); parameters != nullptr && parameters->is_object()) {
        config.rope_theta = json::optional_f32(*parameters, "rope_theta", config.rope_theta);
    }
    if (config.model_type != "voxtral_realtime_encoder") {
        throw std::runtime_error("VoxTral realtime unsupported audio_config.model_type: " + config.model_type);
    }
    if (config.activation_function != "gelu") {
        throw std::runtime_error(
            "VoxTral realtime unsupported audio_config.activation_function: " + config.activation_function);
    }
    if (config.hidden_act != "silu") {
        throw std::runtime_error("VoxTral realtime unsupported audio_config.hidden_act: " + config.hidden_act);
    }
    return config;
}

VoxtralRealtimeTextConfig parse_text_config(const engine::io::json::Value & value,
                                            const engine::io::json::Value & generation) {
    VoxtralRealtimeTextConfig config;
    config.model_type = json::require_string(value, "model_type");
    config.hidden_act = json::require_string(value, "hidden_act");
    config.vocab_size = json::require_i64(value, "vocab_size");
    config.hidden_size = json::require_i64(value, "hidden_size");
    config.intermediate_size = json::require_i64(value, "intermediate_size");
    config.num_hidden_layers = json::require_i64(value, "num_hidden_layers");
    config.num_attention_heads = json::require_i64(value, "num_attention_heads");
    config.num_key_value_heads = json::require_i64(value, "num_key_value_heads");
    config.head_dim = json::require_i64(value, "head_dim");
    config.max_position_embeddings = json::require_i64(value, "max_position_embeddings");
    config.sliding_window = json::require_i64(value, "sliding_window");
    config.bos_token_id = json::optional_i64(value, "bos_token_id", config.bos_token_id);
    config.eos_token_id = json::optional_i64(value, "eos_token_id", config.eos_token_id);
    config.pad_token_id = json::optional_i64(value, "pad_token_id", config.pad_token_id);
    config.pad_token_id = json::optional_i64(generation, "pad_token_id", config.pad_token_id);
    config.rms_norm_eps = json::optional_f32(value, "rms_norm_eps", config.rms_norm_eps);
    if (const auto * parameters = value.find("rope_parameters"); parameters != nullptr && parameters->is_object()) {
        config.rope_theta = json::optional_f32(*parameters, "rope_theta", config.rope_theta);
    }
    config.tie_word_embeddings = json::optional_bool(value, "tie_word_embeddings", config.tie_word_embeddings);
    config.use_cache = json::optional_bool(value, "use_cache", config.use_cache);
    config.use_cache = json::optional_bool(generation, "use_cache", config.use_cache);
    if (config.model_type != "voxtral_realtime_text") {
        throw std::runtime_error("VoxTral realtime unsupported text_config.model_type: " + config.model_type);
    }
    if (config.hidden_act != "silu") {
        throw std::runtime_error("VoxTral realtime unsupported text_config.hidden_act: " + config.hidden_act);
    }
    return config;
}

VoxtralRealtimeFrontendConfig parse_frontend_config(const engine::io::json::Value & processor) {
    const auto processor_class = json::require_string(processor, "processor_class");
    if (processor_class != "VoxtralRealtimeProcessor") {
        throw std::runtime_error("VoxTral realtime unsupported processor_class: " + processor_class);
    }
    const auto & feature = processor.require("feature_extractor");
    const auto feature_type = json::require_string(feature, "feature_extractor_type");
    if (feature_type != "VoxtralRealtimeFeatureExtractor") {
        throw std::runtime_error("VoxTral realtime unsupported feature_extractor.feature_extractor_type: " +
                                 feature_type);
    }
    VoxtralRealtimeFrontendConfig config;
    config.sample_rate = json::require_i64(feature, "sampling_rate");
    config.feature_size = json::require_i64(feature, "feature_size");
    config.n_fft = json::require_i64(feature, "n_fft");
    config.win_length = json::require_i64(feature, "win_length");
    config.hop_length = json::require_i64(feature, "hop_length");
    config.global_log_mel_max = json::optional_f32(feature, "global_log_mel_max", config.global_log_mel_max);
    config.padding_value = json::optional_f32(feature, "padding_value", config.padding_value);
    config.return_attention_mask = json::optional_bool(feature, "return_attention_mask", config.return_attention_mask);
    config.padding_side = json::optional_string(feature, "padding_side", config.padding_side);
    if (config.padding_side != "right") {
        throw std::runtime_error("VoxTral realtime unsupported feature_extractor.padding_side: " +
                                 config.padding_side);
    }
    return config;
}

VoxtralRealtimeConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    const auto generation = resources.parse_json("generation_config");
    const auto processor = resources.parse_json("processor_config");

    VoxtralRealtimeConfig config;
    config.model_type = json::require_string(root, "model_type");
    config.dtype = json::require_string(root, "dtype");
    config.projector_hidden_act = json::require_string(root, "projector_hidden_act");
    config.hidden_size = json::require_i64(root, "hidden_size");
    config.audio_length_per_tok = json::require_i64(root, "audio_length_per_tok");
    config.default_num_delay_tokens = json::require_i64(root, "default_num_delay_tokens");
    config.downsample_factor = json::require_i64(root, "downsample_factor");
    config.audio = parse_audio_config(root.require("audio_config"));
    config.text = parse_text_config(root.require("text_config"), generation);
    config.frontend = parse_frontend_config(processor);
    config.supported_languages = {"Auto"};

    if (config.model_type != "voxtral_realtime") {
        throw std::runtime_error("VoxTral realtime unsupported model_type: " + config.model_type);
    }
    if (config.dtype != "bfloat16") {
        throw std::runtime_error("VoxTral realtime unsupported dtype: " + config.dtype);
    }
    if (config.projector_hidden_act != "gelu") {
        throw std::runtime_error("VoxTral realtime unsupported projector_hidden_act: " + config.projector_hidden_act);
    }
    if (config.frontend.feature_size != config.audio.num_mel_bins) {
        throw std::runtime_error("VoxTral realtime frontend feature size does not "
                                 "match audio_config.num_mel_bins");
    }
    if (config.hidden_size != config.text.hidden_size) {
        throw std::runtime_error("VoxTral realtime hidden_size does not match text_config.hidden_size");
    }
    return config;
}

}  // namespace

std::shared_ptr<const VoxtralRealtimeAssets> load_voxtral_realtime_assets(const std::filesystem::path & model_path) {
    VoxtralRealtimeAssets assets;
    assets.resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("voxtral_realtime"));
    assets.config = parse_config(assets.resources);
    assets.model_weights = assets.resources.open_tensor_source("weights");
    assets::require_tensor_shape(*assets.model_weights, "audio_tower.embedder.conv1.weight",
                                 {assets.config.audio.hidden_size, assets.config.frontend.feature_size, 3});
    assets::require_tensor_shape(
        *assets.model_weights, "audio_tower.layers.0.self_attn.q_proj.weight",
        {assets.config.audio.num_attention_heads * assets.config.audio.head_dim, assets.config.audio.hidden_size});
    assets::require_tensor_shape(*assets.model_weights, "multi_modal_projector.linear_1.weight",
                                 {assets.config.hidden_size,
                                  assets.config.downsample_factor * assets.config.audio.hidden_size});
    assets::require_tensor_shape(*assets.model_weights, "multi_modal_projector.linear_2.weight",
                                 {assets.config.hidden_size, assets.config.hidden_size});
    assets::require_tensor_shape(*assets.model_weights, "language_model.model.embed_tokens.weight",
                                 {assets.config.text.vocab_size, assets.config.text.hidden_size});
    return std::make_shared<VoxtralRealtimeAssets>(std::move(assets));
}

}  // namespace engine::models::voxtral_realtime
