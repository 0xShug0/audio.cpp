#include "engine/models/miotts/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/json.h"

#include <stdexcept>

namespace engine::models::miotts {
namespace json = engine::io::json;
namespace {

int64_t require_added_token_id(
    const json::Value & tokenizer_config,
    const std::string & content) {
    const auto & tokens = tokenizer_config.require("added_tokens_decoder").as_object();
    for (const auto & [key, value] : tokens) {
        if (json::require_string(value, "content") == content) {
            return std::stoll(key);
        }
    }
    throw std::runtime_error("MioTTS tokenizer_config is missing added token: " + content);
}

MioTTSConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    const auto generation = resources.parse_json("generation_config");
    const auto tokenizer_config = resources.parse_json("tokenizer_config");

    MioTTSConfig config;
    config.model_type = json::require_string(root, "model_type");
    config.vocab_size = json::require_i64(root, "vocab_size");
    config.hidden_size = json::require_i64(root, "hidden_size");
    config.intermediate_size = json::require_i64(root, "intermediate_size");
    config.num_hidden_layers = json::require_i64(root, "num_hidden_layers");
    config.num_attention_heads = json::require_i64(root, "num_attention_heads");
    config.num_key_value_heads = json::require_i64(root, "num_key_value_heads");
    config.head_dim = json::optional_i64(root, "head_dim", config.hidden_size / config.num_attention_heads);
    config.max_position_embeddings = json::require_i64(root, "max_position_embeddings");
    config.rms_norm_eps = json::optional_f32(root, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = json::optional_f32(root, "rope_theta", config.rope_theta);
    config.tie_word_embeddings = json::optional_bool(root, "tie_word_embeddings", config.tie_word_embeddings);
    config.eos_token_id = json::optional_i64(generation, "eos_token_id", json::optional_i64(root, "eos_token_id", config.eos_token_id));
    config.pad_token_id = json::optional_i64(generation, "pad_token_id", json::optional_i64(root, "pad_token_id", config.pad_token_id));
    config.max_tokens = 700;
    config.do_sample = json::optional_bool(generation, "do_sample", config.do_sample);
    config.top_k = static_cast<int>(json::optional_i64(generation, "top_k", config.top_k));
    config.top_p = json::optional_f32(generation, "top_p", config.top_p);
    config.temperature = json::optional_f32(generation, "temperature", config.temperature);
    config.repetition_penalty = json::optional_f32(generation, "repetition_penalty", config.repetition_penalty);
    config.speech_token_start_id = require_added_token_id(tokenizer_config, "<|s_0|>");
    if (config.model_type != "qwen3") {
        throw std::runtime_error("MioTTS expects a qwen3 causal LM config");
    }
    if (!config.tie_word_embeddings) {
        throw std::runtime_error("MioTTS currently expects tied input/output embeddings");
    }
    if (config.speech_token_start_id <= 0 || config.speech_token_start_id + config.speech_token_count > config.vocab_size) {
        throw std::runtime_error("MioTTS speech token range is outside the vocabulary");
    }
    return config;
}

}  // namespace

std::shared_ptr<const MioTTSAssets> load_miotts_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<MioTTSAssets>();
    assets->resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("miotts"));
    assets->config = parse_config(assets->resources);
    assets->model_weights = assets->resources.open_tensor_source("weights");
    return assets;
}

}  // namespace engine::models::miotts
