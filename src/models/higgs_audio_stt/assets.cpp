#include "engine/models/higgs_audio_stt/assets.h"

#include "engine/framework/assets/model_package.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::higgs_audio_stt {
namespace json = engine::io::json;
namespace {

HiggsAudioSTTAudioEncoderConfig parse_audio_encoder_config(const json::Value & value) {
    HiggsAudioSTTAudioEncoderConfig config;
    config.num_mel_bins = json::require_i64(value, "num_mel_bins");
    config.encoder_layers = json::require_i64(value, "encoder_layers");
    config.encoder_attention_heads = json::require_i64(value, "encoder_attention_heads");
    config.encoder_ffn_dim = json::require_i64(value, "encoder_ffn_dim");
    config.d_model = json::require_i64(value, "d_model");
    config.max_source_positions = json::require_i64(value, "max_source_positions");
    config.activation_function = json::require_string(value, "activation_function");
    if (config.activation_function != "gelu") {
        throw std::runtime_error("Higgs Audio STT currently supports gelu audio activation");
    }
    return config;
}

HiggsAudioSTTTextDecoderConfig parse_text_decoder_config(
    const json::Value & root,
    const json::Value & text_config) {
    HiggsAudioSTTTextDecoderConfig config;
    config.vocab_size = json::require_i64(text_config, "vocab_size");
    config.output_size = config.vocab_size;
    config.hidden_size = json::require_i64(text_config, "hidden_size");
    config.intermediate_size = json::require_i64(text_config, "intermediate_size");
    config.num_hidden_layers = json::require_i64(text_config, "num_hidden_layers");
    config.num_attention_heads = json::require_i64(text_config, "num_attention_heads");
    config.num_key_value_heads = json::require_i64(text_config, "num_key_value_heads");
    config.head_dim = json::optional_i64(text_config, "head_dim", config.hidden_size / config.num_attention_heads);
    config.max_position_embeddings = json::optional_i64(text_config, "max_position_embeddings", 40960);
    config.audio_in_token_id = json::require_i64(root, "audio_in_token_idx");
    config.audio_out_token_id = json::require_i64(root, "audio_out_token_idx");
    config.audio_bos_token_id = json::optional_i64(root, "audio_bos_token_id", 151669);
    config.audio_eos_token_id = json::require_i64(root, "audio_eos_token_id");
    config.audio_out_bos_token_id = json::require_i64(root, "audio_out_bos_token_id");
    config.pad_token_id = json::require_i64(root, "pad_token_id");
    config.rms_norm_eps = json::optional_f32(text_config, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = json::optional_f32(text_config, "rope_theta", config.rope_theta);
    return config;
}

HiggsAudioSTTConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    const auto & audio_config = root.require("audio_encoder_config");
    const auto & text_config = root.require("text_config");

    HiggsAudioSTTConfig config;
    config.model_type = json::require_string(root, "model_type");
    config.text_decoder_model_type = json::require_string(text_config, "model_type");
    config.model_size = json::optional_string(root, "_name_or_path", config.model_type);
    config.audio_encoder = parse_audio_encoder_config(audio_config);
    config.text_decoder = parse_text_decoder_config(root, text_config);
    config.projector_temporal_downsample = json::require_i64(root, "projector_temporal_downsample");
    if (const auto * chunk_size = root.find("chunk_size_seconds")) {
        if (chunk_size->is_number()) {
            config.frontend.chunk_size_seconds = chunk_size->as_number();
        }
    }
    if (json::optional_string(root, "projector_type", "mlp") != "mlp") {
        throw std::runtime_error("Higgs Audio STT currently supports the mlp audio projector");
    }

    const auto generation = resources.parse_json("generation_config");
    config.text_decoder.eos_token_ids = json::require_i64_array_or_scalar(generation, "eos_token_id");
    if (std::find(config.text_decoder.eos_token_ids.begin(), config.text_decoder.eos_token_ids.end(), 151645) ==
        config.text_decoder.eos_token_ids.end()) {
        config.text_decoder.eos_token_ids.push_back(151645);
    }
    config.text_decoder.pad_token_id = json::optional_i64(generation, "pad_token_id", config.text_decoder.pad_token_id);

    const auto preprocessor = resources.parse_json("preprocessor_config");
    config.frontend.sample_rate = static_cast<int>(json::optional_i64(preprocessor, "sampling_rate", config.frontend.sample_rate));
    config.frontend.feature_size = json::require_i64(preprocessor, "feature_size");
    config.frontend.hop_length = json::require_i64(preprocessor, "hop_length");
    config.frontend.n_fft = json::require_i64(preprocessor, "n_fft");
    config.sample_rate = config.frontend.sample_rate;
    if (config.frontend.feature_size != config.audio_encoder.num_mel_bins) {
        throw std::runtime_error("Higgs Audio STT frontend feature size does not match audio encoder config");
    }

    config.supported_languages = {"en"};
    return config;
}

}  // namespace

std::shared_ptr<const HiggsAudioSTTAssets> load_higgs_audio_stt_assets(const std::filesystem::path & model_path) {
    auto resources = assets::load_resource_bundle_from_package_spec(
        model_path,
        assets::default_model_package_spec_path("higgs_audio_stt"));
    auto assets = std::make_shared<HiggsAudioSTTAssets>();
    assets->config = parse_config(resources);
    assets->model_weights = resources.open_tensor_source("weights");
    assets->resources = std::move(resources);
    return assets;
}

}  // namespace engine::models::higgs_audio_stt
