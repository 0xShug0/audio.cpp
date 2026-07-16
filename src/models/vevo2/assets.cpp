#include "engine/models/vevo2/assets.h"

#include "engine/framework/assets/model_package.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <string_view>
#include <utility>

namespace engine::models::vevo2 {
namespace json = engine::io::json;
namespace {

engine::modules::WhisperEmbeddingConfig parse_whisper_config(const json::Value & root) {
    const auto & audio = root.require("audio_encoder");
    engine::modules::WhisperEmbeddingConfig config;
    config.n_mels = json::require_i64(audio, "n_mels");
    config.n_audio_ctx = json::require_i64(audio, "n_audio_ctx");
    config.n_audio_state = json::require_i64(audio, "n_audio_state");
    config.n_audio_head = json::require_i64(audio, "n_audio_head");
    config.n_audio_layer = json::require_i64(audio, "n_audio_layer");
    if (config.n_mels <= 0 || config.n_audio_ctx <= 0 || config.n_audio_state <= 0 ||
        config.n_audio_head <= 0 || config.n_audio_layer <= 0) {
        throw std::runtime_error("Vevo2 Whisper config contains non-positive dimensions");
    }
    return config;
}

Vevo2ARConfig parse_ar_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("ar_config");
    Vevo2ARConfig config;
    config.model_type = json::require_string(root, "model_type");
    config.vocab_size = json::require_i64(root, "vocab_size");
    config.hidden_size = json::require_i64(root, "hidden_size");
    config.intermediate_size = json::require_i64(root, "intermediate_size");
    config.num_hidden_layers = json::require_i64(root, "num_hidden_layers");
    config.num_attention_heads = json::require_i64(root, "num_attention_heads");
    config.num_key_value_heads = json::require_i64(root, "num_key_value_heads");
    config.max_position_embeddings = json::require_i64(root, "max_position_embeddings");
    config.max_window_layers = json::optional_i64(root, "max_window_layers", config.num_hidden_layers);
    config.bos_token_id = json::optional_i64(root, "bos_token_id", config.bos_token_id);
    config.eos_token_id = json::optional_i64(root, "eos_token_id", config.eos_token_id);
    config.rms_norm_eps = json::optional_f32(root, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = json::optional_f32(root, "rope_theta", config.rope_theta);
    config.tie_word_embeddings = json::optional_bool(root, "tie_word_embeddings", config.tie_word_embeddings);

    const auto generation = resources.parse_json("ar_generation_config");
    config.pad_token_id = json::optional_i64(generation, "pad_token_id", config.pad_token_id);
    config.generation_eos_token_ids =
        json::optional_i64_array_or_scalar(generation, "eos_token_id", {config.eos_token_id});
    config.generation_top_k = json::optional_i64(generation, "top_k", config.generation_top_k);
    config.generation_top_p = json::optional_f32(generation, "top_p", config.generation_top_p);
    config.generation_temperature = json::optional_f32(generation, "temperature", config.generation_temperature);
    config.generation_repetition_penalty =
        json::optional_f32(generation, "repetition_penalty", config.generation_repetition_penalty);
    config.generation_do_sample = json::optional_bool(generation, "do_sample", config.generation_do_sample);

    if (config.model_type != "qwen2") {
        throw std::runtime_error("Vevo2 AR currently expects qwen2 model_type");
    }
    if (config.hidden_size <= 0 || config.intermediate_size <= 0 || config.num_hidden_layers <= 0 ||
        config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 || config.vocab_size <= 0) {
        throw std::runtime_error("Vevo2 AR config contains non-positive dimensions");
    }
    if (config.hidden_size % config.num_attention_heads != 0) {
        throw std::runtime_error("Vevo2 AR hidden_size must be divisible by num_attention_heads");
    }
    if (config.num_attention_heads % config.num_key_value_heads != 0) {
        throw std::runtime_error("Vevo2 AR attention heads must be divisible by key-value heads");
    }
    if (config.generation_eos_token_ids.empty()) {
        throw std::runtime_error("Vevo2 AR generation config requires at least one eos token id");
    }
    return config;
}

Vevo2CocoTokenizerConfig parse_coco_config(const engine::io::json::Value & value, const std::string & expected_type) {
    Vevo2CocoTokenizerConfig config;
    config.coco_type = json::require_string(value, "coco_type");
    config.downsample_rate = json::require_i64(value, "downsample_rate");
    config.codebook_size = json::require_i64(value, "codebook_size");
    config.hidden_size = json::require_i64(value, "hidden_size");
    config.codebook_dim = json::require_i64(value, "codebook_dim");
    const auto & encoder = value.require("encoder");
    config.vocos_dim = json::require_i64(encoder, "vocos_dim");
    config.vocos_intermediate_dim = json::require_i64(encoder, "vocos_intermediate_dim");
    config.vocos_num_layers = json::require_i64(encoder, "vocos_num_layers");
    config.use_normed_whisper = json::optional_bool(value, "use_normed_whisper", config.use_normed_whisper);
    config.whisper_dim = json::require_i64(value, "whisper_dim");
    config.chromagram_dim = json::require_i64(value, "chromagram_dim");
    if (config.coco_type != expected_type) {
        throw std::runtime_error("Vevo2 Coco tokenizer type mismatch: expected " + expected_type);
    }
    if (config.downsample_rate <= 0 || config.codebook_size <= 0 || config.hidden_size <= 0 ||
        config.codebook_dim <= 0 || config.vocos_dim <= 0 || config.vocos_intermediate_dim <= 0 ||
        config.vocos_num_layers <= 0 || config.whisper_dim <= 0 || config.chromagram_dim <= 0) {
        throw std::runtime_error("Vevo2 Coco tokenizer config contains non-positive dimensions");
    }
    return config;
}

Vevo2AcousticPreprocessConfig parse_acoustic_preprocess_config(const engine::io::json::Value & value) {
    Vevo2AcousticPreprocessConfig config;
    config.sample_rate = json::require_i64(value, "sample_rate");
    config.hop_size = json::require_i64(value, "hop_size");
    config.n_fft = json::require_i64(value, "n_fft");
    config.num_mels = json::require_i64(value, "num_mels");
    config.win_size = json::require_i64(value, "win_size");
    config.fmin = json::require_f32(value, "fmin");
    config.fmax = json::require_f32(value, "fmax");
    config.mel_var = json::require_f32(value, "mel_var");
    config.mel_mean = json::require_f32(value, "mel_mean");
    if (config.sample_rate <= 0 || config.hop_size <= 0 || config.n_fft <= 0 || config.num_mels <= 0 ||
        config.win_size <= 0) {
        throw std::runtime_error("Vevo2 acoustic preprocess config contains non-positive dimensions");
    }
    return config;
}

std::pair<Vevo2CocoTokenizerConfig, Vevo2CocoTokenizerConfig> parse_ar_coco_configs(
    const assets::ResourceBundle & resources) {
    const auto root = resources.parse_jsonc("ar_amphion_config");
    const auto & model = root.require("model");
    return {
        parse_coco_config(model.require("coco_style"), "style"),
        parse_coco_config(model.require("coco_content_style"), "content_style"),
    };
}

Vevo2FMConfig parse_fm_config(
    const assets::ResourceBundle & resources,
    std::string_view id,
    bool expect_text_condition) {
    const auto root = resources.parse_jsonc(id);
    Vevo2FMConfig config;
    if (json::require_string(root, "model_type") != "FlowMatchingTransformer") {
        throw std::runtime_error("Vevo2 FM config model_type mismatch");
    }
    config.preprocess = parse_acoustic_preprocess_config(root.require("preprocess"));
    const auto & model = root.require("model");
    const auto & fmt = model.require("flow_matching_transformer");
    config.mel_dim = json::require_i64(fmt, "mel_dim");
    config.hidden_size = json::require_i64(fmt, "hidden_size");
    config.num_layers = json::require_i64(fmt, "num_layers");
    config.num_heads = json::require_i64(fmt, "num_heads");
    config.cfg_scale = json::require_f32(fmt, "cfg_scale");
    config.use_cond_code = json::optional_bool(fmt, "use_cond_code", config.use_cond_code);
    config.cond_codebook_size = json::require_i64(fmt, "cond_codebook_size");
    config.cond_scale_factor = json::require_i64(fmt, "cond_scale_factor");
    config.sigma = json::require_f32(fmt, "sigma");
    config.time_scheduler = json::require_string(fmt, "time_scheduler");
    config.cond_sample_rate = json::require_i64(model, "cond_sample_rate");
    config.coco_content_style = parse_coco_config(model.require("coco"), "content_style");
    config.use_text_as_condition = json::optional_bool(fmt, "use_text_as_condition", false);
    config.text_cond_codebook_size = json::optional_i64(fmt, "text_cond_codebook_size", 0);
    if (config.use_text_as_condition != expect_text_condition) {
        throw std::runtime_error("Vevo2 FM text-condition config mismatch");
    }
    if (config.mel_dim <= 0 || config.hidden_size <= 0 || config.num_layers <= 0 || config.num_heads <= 0 ||
        config.cond_codebook_size <= 0 || config.cond_scale_factor <= 0 || config.cond_sample_rate <= 0) {
        throw std::runtime_error("Vevo2 FM config contains non-positive dimensions");
    }
    if (config.time_scheduler != "cos") {
        throw std::runtime_error("Vevo2 FM currently expects cos time scheduler");
    }
    if (config.mel_dim != config.preprocess.num_mels) {
        throw std::runtime_error("Vevo2 FM mel_dim does not match preprocess num_mels");
    }
    return config;
}

Vevo2VocoderConfig parse_vocoder_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_jsonc("vocoder_config");
    Vevo2VocoderConfig config;
    if (json::require_string(root, "model_type") != "Vocoder") {
        throw std::runtime_error("Vevo2 vocoder config model_type mismatch");
    }
    config.preprocess = parse_acoustic_preprocess_config(root.require("preprocess"));
    const auto & vocos = root.require("model").require("vocos");
    config.input_channels = json::require_i64(vocos, "input_channels");
    config.dim = json::require_i64(vocos, "dim");
    config.intermediate_dim = json::require_i64(vocos, "intermediate_dim");
    config.num_layers = json::require_i64(vocos, "num_layers");
    config.n_fft = json::require_i64(vocos, "n_fft");
    config.hop_size = json::require_i64(vocos, "hop_size");
    config.padding = json::require_string(vocos, "padding");
    if (config.input_channels <= 0 || config.dim <= 0 || config.intermediate_dim <= 0 ||
        config.num_layers <= 0 || config.n_fft <= 0 || config.hop_size <= 0) {
        throw std::runtime_error("Vevo2 vocoder config contains non-positive dimensions");
    }
    if (config.input_channels != config.preprocess.num_mels) {
        throw std::runtime_error("Vevo2 vocoder input channels do not match preprocess num_mels");
    }
    return config;
}

Vevo2Config parse_config(const assets::ResourceBundle & resources) {
    Vevo2Config config;
    config.ar = parse_ar_config(resources);
    const auto coco_configs = parse_ar_coco_configs(resources);
    config.prosody_tokenizer = coco_configs.first;
    config.content_style_tokenizer = coco_configs.second;
    config.fm = parse_fm_config(resources, "fm_config", false);
    config.fm_text = parse_fm_config(resources, "fm_text_config", true);
    config.vocoder = parse_vocoder_config(resources);
    config.whisper = parse_whisper_config(resources.parse_json("whisper_config"));
    return config;
}

}  // namespace

std::shared_ptr<const Vevo2Assets> load_vevo2_assets(
    const std::filesystem::path & model_path,
    const std::optional<std::filesystem::path> & whisper_model_path) {
    Vevo2Assets assets;
    assets.resources = assets::load_resource_bundle_from_package_spec(
        model_path,
        assets::default_model_package_spec_path("vevo2"));
    const auto whisper_root = whisper_model_path.value_or(assets.resources.model_root().parent_path() / "whisper-medium");
    assets.resources.add_file("whisper_config", whisper_root / "config.json");
    assets.resources.add_tensor_source("whisper_weights", whisper_root / "model.safetensors");
    assets.config = parse_config(assets.resources);
    assets.content_style_tokenizer_weights = assets.resources.open_tensor_source("content_style_tokenizer_weights");
    assets.prosody_tokenizer_weights = assets.resources.open_tensor_source("prosody_tokenizer_weights");
    assets.ar_weights = assets.resources.open_tensor_source("ar_weights");
    assets.fm_weights = assets.resources.open_tensor_source("fm_weights");
    assets.fm_whisper_stats = assets.resources.open_tensor_source("fm_whisper_stats");
    assets.vocoder_weights_0 = assets.resources.open_tensor_source("vocoder_weights_0");
    assets.whisper_weights = assets.resources.open_tensor_source("whisper_weights");
    return std::make_shared<Vevo2Assets>(std::move(assets));
}

}  // namespace engine::models::vevo2
