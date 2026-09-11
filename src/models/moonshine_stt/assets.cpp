#include "engine/models/moonshine_stt/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::moonshine_stt {
namespace json = engine::io::json;
namespace {

std::vector<MoonshineSlidingWindow> parse_windows(const engine::io::json::Value & value) {
    std::vector<MoonshineSlidingWindow> windows;
    for (const auto & item : value.as_array()) {
        const auto & pair = item.as_array();
        if (pair.size() != 2) {
            throw std::runtime_error("Moonshine sliding window entries must have two values");
        }
        windows.push_back({pair[0].as_i64(), pair[1].as_i64()});
    }
    return windows;
}

MoonshineConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    MoonshineConfig config;
    config.model_type = json::require_string(root, "model_type");
    config.variant = json::optional_string(root, "_name_or_path", config.model_type);
    config.max_tokens_per_second = json::optional_f32(root, "max_tokens_per_second", config.max_tokens_per_second);
    if (config.model_type != "moonshine_streaming") {
        throw std::runtime_error("Moonshine STT unsupported model_type: " + config.model_type);
    }

    const auto & enc = root.require("encoder_config");
    config.encoder.model_type = json::optional_string(enc, "model_type", config.encoder.model_type);
    config.encoder.hidden_act = json::optional_string(enc, "hidden_act", config.encoder.hidden_act);
    config.encoder.sample_rate = json::require_i64(enc, "sample_rate");
    config.encoder.frame_ms = json::require_f32(enc, "frame_ms");
    config.encoder.frame_length = static_cast<int64_t>(
        std::llround(static_cast<double>(config.encoder.sample_rate) * config.encoder.frame_ms / 1000.0));
    config.encoder.hidden_size = json::require_i64(enc, "hidden_size");
    config.encoder.intermediate_size = json::require_i64(enc, "intermediate_size");
    config.encoder.layers = json::require_i64(enc, "num_hidden_layers");
    config.encoder.heads = json::require_i64(enc, "num_attention_heads");
    config.encoder.kv_heads = json::optional_i64(enc, "num_key_value_heads", config.encoder.heads);
    config.encoder.head_dim = json::require_i64(enc, "head_dim");
    config.encoder.max_position_embeddings = json::require_i64(enc, "max_position_embeddings");
    config.encoder.attention_bias = json::optional_bool(enc, "attention_bias", false);
    config.encoder.sliding_windows = parse_windows(enc.require("sliding_windows"));

    config.decoder.hidden_act = json::optional_string(root, "hidden_act", config.decoder.hidden_act);
    config.decoder.hidden_size = json::require_i64(root, "hidden_size");
    config.decoder.intermediate_size = json::require_i64(root, "intermediate_size");
    config.decoder.layers = json::require_i64(root, "num_hidden_layers");
    config.decoder.heads = json::require_i64(root, "num_attention_heads");
    config.decoder.kv_heads = json::optional_i64(root, "num_key_value_heads", config.decoder.heads);
    config.decoder.head_dim = json::require_i64(root, "head_dim");
    config.decoder.vocab_size = json::require_i64(root, "vocab_size");
    config.decoder.max_position_embeddings = json::require_i64(root, "max_position_embeddings");
    config.decoder.bos_token_id = json::optional_i64(root, "bos_token_id", config.decoder.bos_token_id);
    config.decoder.eos_token_id = json::optional_i64(root, "eos_token_id", config.decoder.eos_token_id);
    config.decoder.pad_token_id = json::optional_i64(root, "pad_token_id", config.decoder.pad_token_id);
    config.decoder.decoder_start_token_id =
        json::optional_i64(root, "decoder_start_token_id", config.decoder.bos_token_id);
    config.decoder.attention_bias = json::optional_bool(root, "attention_bias", false);
    config.decoder.tie_word_embeddings = json::optional_bool(root, "tie_word_embeddings", false);
    if (const auto * rope = root.find("rope_parameters"); rope != nullptr && rope->is_object()) {
        config.decoder.rope_theta = json::optional_f32(*rope, "rope_theta", config.decoder.rope_theta);
        config.decoder.partial_rotary_factor =
            json::optional_f32(*rope, "partial_rotary_factor", config.decoder.partial_rotary_factor);
    }
    config.decoder.rotary_dim =
        static_cast<int64_t>(std::floor(static_cast<double>(config.decoder.head_dim) *
                                        static_cast<double>(config.decoder.partial_rotary_factor)));
    if (config.encoder.hidden_act != "gelu") {
        throw std::runtime_error("Moonshine STT unsupported encoder hidden_act: " + config.encoder.hidden_act);
    }
    if (config.decoder.hidden_act != "silu") {
        throw std::runtime_error("Moonshine STT unsupported decoder hidden_act: " + config.decoder.hidden_act);
    }
    if (config.encoder.sliding_windows.size() != static_cast<size_t>(config.encoder.layers)) {
        throw std::runtime_error("Moonshine STT sliding window count must match encoder layers");
    }
    return config;
}

int64_t checked_token_id(const engine::io::json::Value & value) {
    const int64_t id = value.as_i64();
    if (id < 0 || id > (1 << 24)) {
        throw std::runtime_error("Moonshine tokenizer id out of range");
    }
    return id;
}

std::vector<MoonshineToken> load_tokens(const std::filesystem::path & tokenizer_path) {
    const auto root = json::parse_file(tokenizer_path);
    const auto & vocab = root.require("model").require("vocab").as_object();
    size_t max_id = 0;
    for (const auto & [_, value] : vocab) {
        max_id = std::max(max_id, static_cast<size_t>(checked_token_id(value)));
    }
    if (const auto * added = root.find("added_tokens"); added != nullptr && added->is_array()) {
        for (const auto & item : added->as_array()) {
            max_id = std::max(max_id, static_cast<size_t>(json::require_i64(item, "id")));
        }
    }

    std::vector<MoonshineToken> tokens(max_id + 1);
    for (const auto & [text, value] : vocab) {
        tokens[static_cast<size_t>(checked_token_id(value))].text = text;
    }
    if (const auto * added = root.find("added_tokens"); added != nullptr && added->is_array()) {
        for (const auto & item : added->as_array()) {
            const auto id = static_cast<size_t>(json::require_i64(item, "id"));
            if (id >= tokens.size()) {
                throw std::runtime_error("Moonshine tokenizer added token id out of bounds");
            }
            tokens[id].text = json::require_string(item, "content");
            tokens[id].special = json::optional_bool(item, "special", false);
        }
    }
    return tokens;
}

bool is_byte_fallback_token(const std::string & token, unsigned char & value) {
    if (token.size() != 6 || token[0] != '<' || token[1] != '0' || token[2] != 'x' || token[5] != '>') {
        return false;
    }
    char * end = nullptr;
    const long parsed = std::strtol(token.substr(3, 2).c_str(), &end, 16);
    if (end == nullptr || *end != '\0' || parsed < 0 || parsed > 255) {
        return false;
    }
    value = static_cast<unsigned char>(parsed);
    return true;
}

std::string replace_metaspace(std::string text) {
    const std::string needle = "\xE2\x96\x81";
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        text.replace(pos, needle.size(), " ");
        ++pos;
    }
    return text;
}

}  // namespace

std::shared_ptr<const MoonshineAssets> load_moonshine_stt_assets(const std::filesystem::path & model_path) {
    MoonshineAssets assets;
    assets.resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("moonshine_stt"));
    assets.config = parse_config(assets.resources);
    assets.source = assets.resources.open_tensor_source("weights");
    assets.tokens = load_tokens(assets.resources.require_file("tokenizer_json"));
    if (assets.tokens.size() < static_cast<size_t>(assets.config.decoder.vocab_size)) {
        throw std::runtime_error("Moonshine STT tokenizer vocab is smaller than model vocab");
    }
    assets::require_tensor_shape(
        *assets.source,
        "model.encoder.embedder.linear.weight",
        {assets.config.encoder.hidden_size, assets.config.encoder.frame_length});
    assets::require_tensor_shape(
        *assets.source,
        "model.decoder.embed_tokens.weight",
        {assets.config.decoder.vocab_size, assets.config.decoder.hidden_size});
    return std::make_shared<MoonshineAssets>(std::move(assets));
}

std::string decode_moonshine_tokens(
    const MoonshineAssets & assets,
    const std::vector<int32_t> & token_ids) {
    std::string text;
    for (const int32_t id : token_ids) {
        if (id < 0 || id >= static_cast<int32_t>(assets.tokens.size())) {
            continue;
        }
        const auto & token = assets.tokens[static_cast<size_t>(id)];
        if (token.special || token.text.empty()) {
            continue;
        }
        unsigned char byte = 0;
        if (is_byte_fallback_token(token.text, byte)) {
            text.push_back(static_cast<char>(byte));
        } else {
            text += token.text;
        }
    }
    text = replace_metaspace(std::move(text));
    while (!text.empty() && text.front() == ' ') {
        text.erase(text.begin());
    }
    return text;
}

}  // namespace engine::models::moonshine_stt
