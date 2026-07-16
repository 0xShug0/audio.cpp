#include "engine/models/nemotron_asr/assets.h"

#include "engine/framework/assets/model_package.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::nemotron_asr {
namespace json = engine::io::json;
namespace {

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
    const auto root = json::parse_file(tokenizer_json);
    if (const auto * added = root.find("added_tokens"); added != nullptr && added->is_array()) {
        for (const auto & item : added->as_array()) {
            if (!json::optional_bool(item, "special", false)) {
                continue;
            }
            const int64_t id = json::require_i64(item, "id");
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
    const auto root = json::parse_file(tokenizer_json);
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

NemotronConfig parse_config(const assets::ResourceBundle & resources) {
    const auto config_root = resources.parse_json("config");
    const auto processor_root = resources.parse_json("processor_config");

    NemotronConfig config;
    config.model_type = json::require_string(config_root, "model_type");
    config.vocab_size = json::require_i64(config_root, "vocab_size");
    config.blank_token_id = json::require_i64(config_root, "blank_token_id");
    config.pad_token_id = json::require_i64(config_root, "pad_token_id");
    config.default_prompt_id = json::require_i64(config_root, "default_prompt_id");
    config.decoder_hidden_size = json::require_i64(config_root, "decoder_hidden_size");
    config.decoder_layers = json::require_i64(config_root, "num_decoder_layers");
    config.max_symbols_per_step = json::require_i64(config_root, "max_symbols_per_step");
    config.num_prompts = json::require_i64(config_root, "num_prompts");
    config.prompt_intermediate_size = json::require_i64(config_root, "prompt_intermediate_size");

    const auto & encoder = config_root.require("encoder_config");
    config.encoder.hidden_size = json::require_i64(encoder, "hidden_size");
    config.encoder.intermediate_size = json::require_i64(encoder, "intermediate_size");
    config.encoder.layers = json::require_i64(encoder, "num_hidden_layers");
    config.encoder.heads = json::require_i64(encoder, "num_attention_heads");
    config.encoder.kv_heads = json::require_i64(encoder, "num_key_value_heads");
    config.encoder.conv_kernel = json::require_i64(encoder, "conv_kernel_size");
    config.encoder.sliding_window = json::require_i64(encoder, "sliding_window");
    config.encoder.subsampling_factor = json::require_i64(encoder, "subsampling_factor");
    config.encoder.subsampling_channels = json::require_i64(encoder, "subsampling_conv_channels");
    config.encoder.subsampling_kernel = json::require_i64(encoder, "subsampling_conv_kernel_size");
    config.encoder.subsampling_stride = json::require_i64(encoder, "subsampling_conv_stride");
    config.encoder.max_position_embeddings = json::require_i64(encoder, "max_position_embeddings");
    config.encoder.default_lookahead_tokens =
        json::require_i64(encoder, "default_num_lookahead_tokens");
    config.encoder.supported_lookahead_tokens =
        json::require_i64_array(encoder, "supported_num_lookahead_tokens");

    const auto & feature = processor_root.require("feature_extractor");
    config.frontend.sample_rate = json::require_i64(feature, "sampling_rate");
    config.frontend.feature_size = json::require_i64(feature, "feature_size");
    config.frontend.n_fft = json::require_i64(feature, "n_fft");
    config.frontend.win_length = json::require_i64(feature, "win_length");
    config.frontend.hop_length = json::require_i64(feature, "hop_length");
    config.frontend.preemphasis = json::require_f32(feature, "preemphasis");
    config.encoder.default_lookahead_tokens =
        json::require_i64(processor_root, "default_num_lookahead_tokens");
    config.encoder.supported_lookahead_tokens =
        json::require_i64_array(processor_root, "supported_num_lookahead_tokens");
    config.prompt_dictionary = json::require_i64_object(processor_root, "prompt_dictionary");
    validate_config(config);
    return config;
}

}  // namespace

std::shared_ptr<const NemotronASRAssets> load_nemotron_asr_assets(const std::filesystem::path & model_path) {
    auto resources = engine::assets::load_resource_bundle_from_package_spec(
        model_path,
        engine::assets::default_model_package_spec_path("nemotron_asr"));
    auto assets = std::make_shared<NemotronASRAssets>();
    assets->resources = std::move(resources);
    assets->source = assets->resources.open_tensor_source("weights");
    assets->config = parse_config(assets->resources);
    const auto & tokenizer_json = assets->resources.require_file("tokenizer_json");
    assets->tokenizer = engine::tokenizers::load_huggingface_tokenizer_json(tokenizer_json);
    assets->special_token_ids =
        parse_special_token_ids(tokenizer_json, assets->config.vocab_size);
    parse_decoder_metadata(
        tokenizer_json,
        assets->metaspace_replacement,
        assets->trim_leading_space);
    return assets;
}

}  // namespace engine::models::nemotron_asr
