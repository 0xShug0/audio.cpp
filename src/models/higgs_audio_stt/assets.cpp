#include "engine/models/higgs_audio_stt/assets.h"

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::higgs_audio_stt {
namespace json = engine::io::json;
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("Higgs Audio STT model path does not exist: " + model_path.string());
}

HiggsAudioSTTAudioEncoderConfig parse_audio_encoder_config(const engine::io::json::Value & value) {
    HiggsAudioSTTAudioEncoderConfig config;
    config.num_mel_bins = value.require("num_mel_bins").as_i64();
    config.encoder_layers = value.require("encoder_layers").as_i64();
    config.encoder_attention_heads = value.require("encoder_attention_heads").as_i64();
    config.encoder_ffn_dim = value.require("encoder_ffn_dim").as_i64();
    config.d_model = value.require("d_model").as_i64();
    config.max_source_positions = value.require("max_source_positions").as_i64();
    config.activation_function = value.require("activation_function").as_string();
    if (config.activation_function != "gelu") {
        throw std::runtime_error("Higgs Audio STT currently supports gelu audio activation");
    }
    return config;
}

HiggsAudioSTTTextDecoderConfig parse_text_decoder_config(
    const engine::io::json::Value & root,
    const engine::io::json::Value & text_config) {
    HiggsAudioSTTTextDecoderConfig config;
    config.vocab_size = text_config.require("vocab_size").as_i64();
    config.output_size = config.vocab_size;
    config.hidden_size = text_config.require("hidden_size").as_i64();
    config.intermediate_size = text_config.require("intermediate_size").as_i64();
    config.num_hidden_layers = text_config.require("num_hidden_layers").as_i64();
    config.num_attention_heads = text_config.require("num_attention_heads").as_i64();
    config.num_key_value_heads = text_config.require("num_key_value_heads").as_i64();
    config.head_dim = json::optional_i64(text_config, "head_dim", config.hidden_size / config.num_attention_heads);
    config.max_position_embeddings = json::optional_i64(text_config, "max_position_embeddings", 40960);
    config.audio_in_token_id = root.require("audio_in_token_idx").as_i64();
    config.audio_out_token_id = root.require("audio_out_token_idx").as_i64();
    config.audio_bos_token_id = json::optional_i64(root, "audio_bos_token_id", 151669);
    config.audio_eos_token_id = root.require("audio_eos_token_id").as_i64();
    config.audio_out_bos_token_id = root.require("audio_out_bos_token_id").as_i64();
    config.pad_token_id = root.require("pad_token_id").as_i64();
    config.rms_norm_eps = json::optional_f32(text_config, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = json::optional_f32(text_config, "rope_theta", config.rope_theta);
    return config;
}

HiggsAudioSTTConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    const auto & audio_config = root.require("audio_encoder_config");
    const auto & text_config = root.require("text_config");

    HiggsAudioSTTConfig config;
    config.model_type = root.require("model_type").as_string();
    config.text_decoder_model_type = text_config.require("model_type").as_string();
    config.model_size = json::optional_string(root, "_name_or_path", config.model_type);
    config.audio_encoder = parse_audio_encoder_config(audio_config);
    config.text_decoder = parse_text_decoder_config(root, text_config);
    config.projector_temporal_downsample = root.require("projector_temporal_downsample").as_i64();
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
    config.frontend.feature_size = preprocessor.require("feature_size").as_i64();
    config.frontend.hop_length = preprocessor.require("hop_length").as_i64();
    config.frontend.n_fft = preprocessor.require("n_fft").as_i64();
    config.sample_rate = config.frontend.sample_rate;
    if (config.frontend.feature_size != config.audio_encoder.num_mel_bins) {
        throw std::runtime_error("Higgs Audio STT frontend feature size does not match audio encoder config");
    }

    config.supported_languages = {"en"};
    return config;
}

assets::ResourceBundle make_resource_bundle(const std::filesystem::path & model_path) {
    assets::ResourceBundle resources(resolve_model_root(model_path));
    resources.add_model_files({
        {"config", "config.json", true},
        {"generation_config", "generation_config.json", true},
        {"preprocessor_config", "../whisper-large-v3/preprocessor_config.json", true},
        {"weights_index", "model.safetensors.index.json", true},
        {"tokenizer_config", "tokenizer_config.json", true},
        {"vocab", "vocab.json", true},
        {"merges", "merges.txt", true},
    });
    return resources;
}

void fill_paths(
    HiggsAudioSTTAssetPaths & paths,
    const assets::ResourceBundle & resources) {
    paths.model_root = resources.model_root();
    paths.config_path = resources.require_file("config");
    paths.generation_config_path = resources.require_file("generation_config");
    paths.preprocessor_config_path = resources.require_file("preprocessor_config");
    paths.model_index_path = resources.require_file("weights_index");
    paths.model_shard_paths = engine::assets::indexed_tensor_source_shard_paths(
        paths.model_index_path,
        paths.model_index_path.parent_path());
    paths.tokenizer_config_path = resources.require_file("tokenizer_config");
    paths.tokenizer_vocab_path = resources.require_file("vocab");
    paths.tokenizer_merges_path = resources.require_file("merges");
}

}  // namespace

HiggsAudioSTTAssetPaths resolve_higgs_audio_stt_assets(const std::filesystem::path & model_path) {
    auto resources = make_resource_bundle(model_path);
    HiggsAudioSTTAssetPaths paths;
    fill_paths(paths, resources);
    return paths;
}

std::shared_ptr<const HiggsAudioSTTAssets> load_higgs_audio_stt_assets(const std::filesystem::path & model_path) {
    auto resources = make_resource_bundle(model_path);
    auto assets = std::make_shared<HiggsAudioSTTAssets>();
    fill_paths(assets->paths, resources);
    assets->config = parse_config(resources);
    assets->model_weights = engine::assets::open_indexed_tensor_source(
        assets->paths.model_index_path,
        assets->paths.model_index_path.parent_path());
    return assets;
}

}  // namespace engine::models::higgs_audio_stt
