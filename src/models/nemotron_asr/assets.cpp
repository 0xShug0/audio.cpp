#include "engine/models/nemotron_asr/assets.h"

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::nemotron_asr {
namespace {

std::filesystem::path require_file(const std::filesystem::path & path, const char * role) {
    if (!engine::io::is_existing_file(path)) {
        throw std::runtime_error(std::string("Nemotron ASR missing ") + role + ": " + path.string());
    }
    return std::filesystem::weakly_canonical(path);
}

std::filesystem::path resolve_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("Nemotron ASR model path does not exist: " + model_path.string());
}

std::vector<int64_t> parse_i64_array(const engine::io::json::Value & value) {
    std::vector<int64_t> out;
    for (const auto & item : value.as_array()) {
        out.push_back(item.as_i64());
    }
    return out;
}

std::unordered_map<std::string, int64_t> parse_prompt_dictionary(const engine::io::json::Value & value) {
    std::unordered_map<std::string, int64_t> out;
    for (const auto & [key, item] : value.as_object()) {
        out.emplace(key, item.as_i64());
    }
    return out;
}

void validate_config(const NemotronConfig & config) {
    if (config.model_type != "nemotron3_5_asr") {
        throw std::runtime_error("Nemotron ASR expects model_type=nemotron3_5_asr");
    }
    if (config.vocab_size <= 0 || config.blank_token_id < 0 || config.blank_token_id >= config.vocab_size) {
        throw std::runtime_error("Nemotron ASR invalid vocab metadata");
    }
    if (config.decoder_hidden_size <= 0 || config.decoder_layers != 2) {
        throw std::runtime_error("Nemotron ASR currently expects the 2-layer RNNT decoder from the checkpoint");
    }
    if (config.encoder.hidden_size <= 0 || config.encoder.layers <= 0 || config.encoder.heads <= 0 ||
        config.encoder.hidden_size % config.encoder.heads != 0) {
        throw std::runtime_error("Nemotron ASR invalid encoder metadata");
    }
    if (config.encoder.subsampling_factor != 8 || config.encoder.subsampling_kernel != 3 ||
        config.encoder.subsampling_stride != 2) {
        throw std::runtime_error("Nemotron ASR expects the official factor-8 causal subsampling config");
    }
    if (config.frontend.sample_rate != 16000 || config.frontend.feature_size <= 0 ||
        config.frontend.n_fft <= 0 || config.frontend.win_length <= 0 || config.frontend.hop_length <= 0) {
        throw std::runtime_error("Nemotron ASR invalid frontend metadata");
    }
    if (config.prompt_dictionary.empty()) {
        throw std::runtime_error("Nemotron ASR processor_config prompt dictionary is empty");
    }
}

std::vector<uint8_t> parse_special_token_ids(const std::filesystem::path & tokenizer_json, int64_t vocab_size) {
    std::vector<uint8_t> special(static_cast<size_t>(vocab_size), 0);
    const auto root = engine::io::json::parse_file(tokenizer_json);
    if (const auto * added = root.find("added_tokens"); added != nullptr && added->is_array()) {
        for (const auto & item : added->as_array()) {
            if (!engine::io::json::optional_bool(item, "special", false)) {
                continue;
            }
            const int64_t id = engine::io::json::require_i64(item, "id");
            if (id >= 0 && id < vocab_size) {
                special[static_cast<size_t>(id)] = 1;
            }
        }
    }
    return special;
}

void parse_decoder_metadata(
    const std::filesystem::path & tokenizer_json,
    std::string & metaspace_replacement,
    bool & trim_leading_space) {
    const auto root = engine::io::json::parse_file(tokenizer_json);
    metaspace_replacement.clear();
    trim_leading_space = false;
    if (const auto * decoder = root.find("decoder"); decoder != nullptr
        && decoder->is_object()
        && decoder->find("type") != nullptr
        && decoder->require("type").as_string() == "Metaspace") {
        metaspace_replacement = decoder->require("replacement").as_string();
        trim_leading_space = true;
    }
}

NemotronConfig parse_config(
    const engine::io::json::Value & config_root,
    const engine::io::json::Value & processor_root) {
    NemotronConfig config;
    config.model_type = engine::io::json::require_string(config_root, "model_type");
    config.vocab_size = engine::io::json::require_i64(config_root, "vocab_size");
    config.blank_token_id = engine::io::json::require_i64(config_root, "blank_token_id");
    config.pad_token_id = engine::io::json::require_i64(config_root, "pad_token_id");
    config.default_prompt_id = engine::io::json::require_i64(config_root, "default_prompt_id");
    config.decoder_hidden_size = engine::io::json::require_i64(config_root, "decoder_hidden_size");
    config.decoder_layers = engine::io::json::require_i64(config_root, "num_decoder_layers");
    config.max_symbols_per_step = engine::io::json::require_i64(config_root, "max_symbols_per_step");
    config.num_prompts = engine::io::json::require_i64(config_root, "num_prompts");
    config.prompt_intermediate_size = engine::io::json::require_i64(config_root, "prompt_intermediate_size");

    const auto & encoder = config_root.require("encoder_config");
    config.encoder.hidden_size = engine::io::json::require_i64(encoder, "hidden_size");
    config.encoder.intermediate_size = engine::io::json::require_i64(encoder, "intermediate_size");
    config.encoder.layers = engine::io::json::require_i64(encoder, "num_hidden_layers");
    config.encoder.heads = engine::io::json::require_i64(encoder, "num_attention_heads");
    config.encoder.kv_heads = engine::io::json::require_i64(encoder, "num_key_value_heads");
    config.encoder.conv_kernel = engine::io::json::require_i64(encoder, "conv_kernel_size");
    config.encoder.sliding_window = engine::io::json::require_i64(encoder, "sliding_window");
    config.encoder.subsampling_factor = engine::io::json::require_i64(encoder, "subsampling_factor");
    config.encoder.subsampling_channels = engine::io::json::require_i64(encoder, "subsampling_conv_channels");
    config.encoder.subsampling_kernel = engine::io::json::require_i64(encoder, "subsampling_conv_kernel_size");
    config.encoder.subsampling_stride = engine::io::json::require_i64(encoder, "subsampling_conv_stride");
    config.encoder.max_position_embeddings = engine::io::json::require_i64(encoder, "max_position_embeddings");
    config.encoder.default_lookahead_tokens =
        engine::io::json::require_i64(encoder, "default_num_lookahead_tokens");
    config.encoder.supported_lookahead_tokens =
        parse_i64_array(encoder.require("supported_num_lookahead_tokens"));

    const auto & feature = processor_root.require("feature_extractor");
    config.frontend.sample_rate = engine::io::json::require_i64(feature, "sampling_rate");
    config.frontend.feature_size = engine::io::json::require_i64(feature, "feature_size");
    config.frontend.n_fft = engine::io::json::require_i64(feature, "n_fft");
    config.frontend.win_length = engine::io::json::require_i64(feature, "win_length");
    config.frontend.hop_length = engine::io::json::require_i64(feature, "hop_length");
    config.frontend.preemphasis = engine::io::json::require_f32(feature, "preemphasis");
    config.encoder.default_lookahead_tokens =
        engine::io::json::require_i64(processor_root, "default_num_lookahead_tokens");
    config.encoder.supported_lookahead_tokens =
        parse_i64_array(processor_root.require("supported_num_lookahead_tokens"));
    config.prompt_dictionary = parse_prompt_dictionary(processor_root.require("prompt_dictionary"));
    validate_config(config);
    return config;
}

}  // namespace

NemotronAssetPaths resolve_nemotron_asr_assets(const std::filesystem::path & model_path) {
    const auto root = resolve_root(model_path);
    NemotronAssetPaths paths;
    paths.model_root = root;
    paths.config_path = require_file(root / "config.json", "config.json");
    paths.processor_config_path = require_file(root / "processor_config.json", "processor_config.json");
    paths.tokenizer_json_path = require_file(root / "tokenizer.json", "tokenizer.json");
    paths.weight_path = require_file(root / "model.safetensors", "model.safetensors");
    return paths;
}

std::shared_ptr<const NemotronAssets> load_nemotron_asr_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<NemotronAssets>();
    assets->paths = resolve_nemotron_asr_assets(model_path);
    engine::assets::ResourceBundle resources(assets->paths.model_root);
    resources.add_file("weights", assets->paths.weight_path);
    assets->source = resources.open_tensor_source("weights");
    assets->config = parse_config(
        engine::io::json::parse_file(assets->paths.config_path),
        engine::io::json::parse_file(assets->paths.processor_config_path));
    assets->tokenizer = engine::tokenizers::load_huggingface_tokenizer_json(assets->paths.tokenizer_json_path);
    assets->special_token_ids =
        parse_special_token_ids(assets->paths.tokenizer_json_path, assets->config.vocab_size);
    parse_decoder_metadata(
        assets->paths.tokenizer_json_path,
        assets->metaspace_replacement,
        assets->trim_leading_space);
    return assets;
}

}  // namespace engine::models::nemotron_asr
