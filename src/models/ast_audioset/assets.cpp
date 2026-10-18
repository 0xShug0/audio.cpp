#include "engine/models/ast_audioset/assets.h"

#include "engine/framework/model_spec/package.h"

#include <algorithm>
#include <stdexcept>

namespace engine::models::ast_audioset {
namespace {

Config parse_config(const engine::io::json::Value & model, const engine::io::json::Value & processor) {
    Config config;
    config.hidden_size = model.require("hidden_size").as_i64();
    config.layers = model.require("num_hidden_layers").as_i64();
    config.heads = model.require("num_attention_heads").as_i64();
    config.intermediate_size = model.require("intermediate_size").as_i64();
    config.patch_size = model.require("patch_size").as_i64();
    config.frequency_stride = model.require("frequency_stride").as_i64();
    config.time_stride = model.require("time_stride").as_i64();
    config.max_length = model.require("max_length").as_i64();
    config.num_mel_bins = model.require("num_mel_bins").as_i64();
    config.layer_norm_eps = model.require("layer_norm_eps").as_f32();
    config.sample_rate = static_cast<int>(processor.require("sampling_rate").as_i64());
    config.feature_mean = processor.require("mean").as_f32();
    config.feature_std = processor.require("std").as_f32();

    const auto & labels = model.require("id2label").as_object();
    config.labels.resize(labels.size());
    for (const auto & [key, value] : labels) {
        const size_t index = static_cast<size_t>(std::stoull(key));
        if (index >= config.labels.size()) {
            throw std::runtime_error("AST id2label indices must be dense");
        }
        config.labels[index] = value.as_string();
    }
    if (config.hidden_size != 768 || config.layers != 12 || config.heads != 12 ||
        config.intermediate_size != 3072 || config.patch_size != 16 ||
        config.frequency_stride != 10 || config.time_stride != 10 ||
        config.max_length != 1024 || config.num_mel_bins != 128 ||
        config.sample_rate != 16000 || config.labels.size() != 527 ||
        config.feature_std <= 0.0f) {
        throw std::runtime_error("unsupported AST AudioSet configuration");
    }
    if (std::any_of(config.labels.begin(), config.labels.end(), [](const std::string & label) { return label.empty(); })) {
        throw std::runtime_error("AST id2label contains an empty entry");
    }
    return config;
}

}  // namespace

std::shared_ptr<Assets> load_assets(const std::filesystem::path & model_path) {
    auto result = std::make_shared<Assets>();
    result->resources = model_spec::load_resource_bundle_for_family(model_path, "ast_audioset");
    result->config = parse_config(result->resources.parse_json("config"), result->resources.parse_json("processor"));
    return result;
}

}  // namespace engine::models::ast_audioset
