#include "engine/community_models/parakeet_tdt/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::community_models::parakeet_tdt {
namespace json = engine::io::json;
namespace {

void validate_config(const ParakeetConfig & config) {
    if (config.model_type != "parakeet_tdt") {
        throw std::runtime_error("Parakeet TDT expects model_type=parakeet_tdt");
    }
    if (config.vocab_size <= 0 || config.blank_token_id < 0 || config.blank_token_id >= config.vocab_size) {
        throw std::runtime_error("Parakeet TDT invalid vocab metadata");
    }
    if (config.decoder_hidden_size <= 0 || config.decoder_layers != 2) {
        throw std::runtime_error("Parakeet TDT expects 2-layer LSTM decoder");
    }
    if (config.encoder.hidden_size <= 0 || config.encoder.layers <= 0 || config.encoder.heads <= 0 ||
        config.encoder.hidden_size % config.encoder.heads != 0) {
        throw std::runtime_error("Parakeet TDT invalid encoder metadata");
    }
    if (config.encoder.subsampling_factor != 8 || config.encoder.subsampling_kernel != 3 ||
        config.encoder.subsampling_stride != 2) {
        throw std::runtime_error("Parakeet TDT expects factor-8 causal subsampling config");
    }
    if (config.frontend.sample_rate != 16000 || config.frontend.feature_size <= 0 ||
        config.frontend.n_fft <= 0 || config.frontend.win_length <= 0 || config.frontend.hop_length <= 0) {
        throw std::runtime_error("Parakeet TDT invalid frontend metadata");
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

ParakeetConfig parse_config(const assets::ResourceBundle & resources) {
    const auto config_root = resources.parse_json("config");

    ParakeetConfig config;
    config.model_type = json::require_string(config_root, "model_type");
    config.vocab_size = json::require_i64(config_root, "vocab_size");
    config.blank_token_id = json::require_i64(config_root, "blank_token_id");
    config.pad_token_id = json::require_i64(config_root, "pad_token_id");
    config.decoder_hidden_size = json::require_i64(config_root, "decoder_hidden_size");
    config.decoder_layers = json::require_i64(config_root, "num_decoder_layers");
    config.max_symbols_per_step = json::require_i64(config_root, "max_symbols_per_step");

    if (const auto * dur = config_root.find("durations"); dur != nullptr && dur->is_array()) {
        config.durations.clear();
        for (const auto & d : dur->as_array()) {
            if (!d.is_number()) continue;
            config.durations.push_back(static_cast<int32_t>(d.as_i64()));
        }
    }

    const auto & encoder = config_root.require("encoder_config");
    config.encoder.hidden_size = json::require_i64(encoder, "hidden_size");
    config.encoder.intermediate_size = json::require_i64(encoder, "intermediate_size");
    config.encoder.layers = json::require_i64(encoder, "num_hidden_layers");
    config.encoder.heads = json::require_i64(encoder, "num_attention_heads");
    config.encoder.conv_kernel = json::require_i64(encoder, "conv_kernel_size");
    config.encoder.subsampling_factor = json::require_i64(encoder, "subsampling_factor");
    config.encoder.subsampling_channels = json::require_i64(encoder, "subsampling_conv_channels");
    config.encoder.subsampling_kernel = json::require_i64(encoder, "subsampling_conv_kernel_size");
    config.encoder.subsampling_stride = json::require_i64(encoder, "subsampling_conv_stride");
    config.encoder.max_position_embeddings = json::require_i64(encoder, "max_position_embeddings");

    // processor_config.json is genuinely optional (not every model directory
    // layout provides it), so fall back to the ParakeetFrontendConfig
    // defaults when it's absent. But if the file IS present, any parse or
    // schema failure is a real problem worth failing loudly on: silently
    // keeping the defaults could load a variant model's weights against the
    // wrong frontend parameters (sample rate, FFT size, hop length, ...)
    // without any indication anything went wrong.
    if (resources.has_file("processor_config")) {
        const auto processor_root = resources.parse_json("processor_config");
        const auto & feature = processor_root.require("feature_extractor");
        config.frontend.sample_rate = json::require_i64(feature, "sampling_rate");
        config.frontend.feature_size = json::require_i64(feature, "feature_size");
        config.frontend.n_fft = json::require_i64(feature, "n_fft");
        config.frontend.win_length = json::require_i64(feature, "win_length");
        config.frontend.hop_length = json::require_i64(feature, "hop_length");
        if (const auto * p = feature.find("preemphasis"); p != nullptr) {
            config.frontend.preemphasis = json::require_f32(feature, "preemphasis");
        }
    }

    validate_config(config);
    return config;
}

}  // namespace

std::shared_ptr<const ParakeetTDTAssets> load_parakeet_assets(const std::filesystem::path & model_path) {
    auto resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("parakeet_tdt"));
    auto assets = std::make_shared<ParakeetTDTAssets>();
    assets->resources = std::move(resources);
    assets->source = assets->resources.open_tensor_source("weights");
    assets->config = parse_config(assets->resources);
    const auto & tokenizer_json = assets->resources.require_file("tokenizer_json");
    assets->tokenizer = engine::tokenizers::load_huggingface_tokenizer_json(tokenizer_json);
    assets->special_token_ids =
        parse_special_token_ids(tokenizer_json, assets->config.vocab_size);
    return assets;
}

}  // namespace engine::community_models::parakeet_tdt
