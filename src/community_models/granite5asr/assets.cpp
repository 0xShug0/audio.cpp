#include "engine/community_models/granite5asr/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::community_models::granite5asr {
namespace json = engine::io::json;
namespace {

void validate_config(const Granite5ASRConfig & config) {
    if (config.vocab_size <= 0) {
        throw std::runtime_error("Granite 5 ASR invalid vocab size");
    }
    if (config.encoder.hidden_size <= 0 || config.encoder.num_layers <= 0 ||
        config.encoder.num_attention_heads <= 0 ||
        config.encoder.hidden_size % config.encoder.num_attention_heads != 0) {
        throw std::runtime_error("Granite 5 ASR invalid encoder metadata");
    }
    if (config.frontend.sample_rate != 16000 || config.frontend.n_mels <= 0 ||
        config.frontend.n_fft <= 0 || config.frontend.win_length <= 0 ||
        config.frontend.hop_length <= 0) {
        throw std::runtime_error("Granite 5 ASR invalid frontend metadata");
    }
}

std::vector<uint8_t> parse_special_token_ids(
    const std::filesystem::path & tokenizer_json,
    int64_t vocab_size) {
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

Granite5ASRConfig parse_config(const assets::ResourceBundle & resources) {
    const auto config_root = resources.parse_json("config");

    Granite5ASRConfig config;
    config.model_type = json::optional_string(config_root, "model_type", "granite_speech5_ctc");
    config.vocab_size = json::optional_i64(config_root, "vocab_size", 16384);
    config.pad_token_id = json::optional_i64(config_root, "pad_token_id", 0);
    config.blank_token_id = json::optional_i64(config_root, "pad_token_id", 0);

    if (const auto * enc = config_root.find("encoder_config"); enc != nullptr && enc->is_object()) {
        config.encoder.hidden_size = json::optional_i64(*enc, "hidden_size", 1024);
        config.encoder.intermediate_size = json::optional_i64(*enc, "intermediate_size", 4096);
        config.encoder.num_layers = json::optional_i64(*enc, "num_hidden_layers", 16);
        config.encoder.num_attention_heads = json::optional_i64(*enc, "num_attention_heads", 8);
        config.encoder.num_key_value_heads = json::optional_i64(*enc, "num_key_value_heads", 8);
        config.encoder.head_dim = json::optional_i64(*enc, "head_dim", 128);
        config.encoder.context_size = json::optional_i64(*enc, "context_size", 128);
        config.encoder.conv_kernel_size = json::optional_i64(*enc, "conv_kernel_size", 7);
        config.encoder.conv_expansion_factor = json::optional_i64(*enc, "conv_expansion_factor", 2);
        config.encoder.max_position_embeddings = json::optional_i64(*enc, "max_position_embeddings", 512);
        config.encoder.num_mel_bins = json::optional_i64(*enc, "num_mel_bins", 80);
        config.encoder.vocab_size = json::optional_i64(*enc, "vocab_size", config.vocab_size);

        if (const auto * subsample = enc->find("subsample_layers"); subsample != nullptr && subsample->is_array()) {
            config.encoder.subsample_layers.clear();
            for (const auto & val : subsample->as_array()) {
                if (val.is_number()) {
                    config.encoder.subsample_layers.push_back(val.as_i64());
                }
            }
        }
    }

    if (resources.has_file("preprocessor_config")) {
        const auto preproc_root = resources.parse_json("preprocessor_config");
        config.frontend.sample_rate = json::optional_i64(preproc_root, "sample_rate", 16000);
        config.frontend.n_fft = json::optional_i64(preproc_root, "n_fft", 512);
        config.frontend.win_length = json::optional_i64(preproc_root, "win_length", 400);
        config.frontend.hop_length = json::optional_i64(preproc_root, "hop_length", 160);
        config.frontend.n_mels = json::optional_i64(preproc_root, "n_mels", 80);
        config.frontend.stack_factor = json::optional_i64(preproc_root, "stack_factor", 2);
        config.frontend.deltas = json::optional_bool(preproc_root, "deltas", true);
        config.frontend.delta_win_length = json::optional_i64(preproc_root, "delta_win_length", 3);
        config.frontend.logmel_floor_db = json::optional_f32(preproc_root, "logmel_floor_db", 8.0f);
    }

    config.encoder.input_features = config.frontend.n_mels * (config.frontend.deltas ? 2 : 1) * config.frontend.stack_factor;
    validate_config(config);
    return config;
}

}  // namespace

std::shared_ptr<const Granite5ASRAssets> load_granite5asr_assets(
    const std::filesystem::path & model_path) {
    auto resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("granite5asr"));
    auto assets = std::make_shared<Granite5ASRAssets>();
    assets->resources = std::move(resources);
    assets->source = assets->resources.open_tensor_source("weights");
    assets->config = parse_config(assets->resources);
    const auto & tokenizer_json = assets->resources.require_file("tokenizer_json");
    assets->tokenizer = engine::tokenizers::load_huggingface_tokenizer_json(tokenizer_json);
    assets->special_token_ids =
        parse_special_token_ids(tokenizer_json, assets->config.vocab_size);
    return assets;
}

}  // namespace engine::community_models::granite5asr
