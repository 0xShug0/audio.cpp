#include "engine/models/muscriptor/assets.h"

#include "engine/framework/io/config.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::models::muscriptor {
namespace json = engine::io::json;
namespace {

MuScriptorConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    MuScriptorConfig config;
    config.model_type = json::require_string(root, "model_type");
    if (config.model_type != "muscriptor") {
        throw std::runtime_error("MuScriptor config model_type mismatch: " + config.model_type);
    }
    config.variant = json::require_string(root, "variant");
    config.dim = json::require_i64(root, "dim");
    config.num_heads = json::require_i64(root, "num_heads");
    config.num_layers = json::require_i64(root, "num_layers");
    config.card = json::require_i64(root, "card");
    engine::io::require_positive(config.dim, "dim");
    engine::io::require_positive(config.num_heads, "num_heads");
    engine::io::require_positive(config.num_layers, "num_layers");
    engine::io::require_positive(config.card, "card");
    engine::io::require_positive(config.sample_rate, "sample_rate");
    engine::io::require_positive(config.segment_seconds, "segment_seconds");
    engine::io::require_positive(config.n_fft, "n_fft");
    engine::io::require_positive(config.hop_length, "hop_length");
    engine::io::require_positive(config.n_mels, "n_mels");
    engine::io::require_divisible(config.dim, config.num_heads, "dim / num_heads");
    return config;
}

void validate_weight_anchors(const MuScriptorAssets & assets) {
    const auto & config = assets.config;
    const auto & weights = *assets.model_weights;
    assets::require_tensor_shape(
        weights,
        "condition_provider.conditioners.self_wav.output_proj.weight",
        {config.dim, config.n_mels});
    assets::require_tensor_shape(weights, "condition_provider.conditioners.self_wav.output_proj.bias", {config.dim});
    assets::require_tensor_shape(weights, "condition_provider.conditioners.instrument_group.embed.weight", {1001, config.dim});
    assets::require_tensor_shape(weights, "condition_provider.conditioners.dataset_name.embed.weight", {5, config.dim});
    assets::require_tensor_shape(weights, "emb.0.weight", {config.card + 1, config.dim});
    assets::require_tensor_shape(weights, "transformer.layers.0.self_attn.in_proj_weight", {config.dim * 3, config.dim});
    assets::require_tensor_shape(weights, "transformer.layers.0.self_attn.out_proj.weight", {config.dim, config.dim});
    assets::require_tensor_shape(weights, "transformer.layers.0.linear1.weight", {config.dim * 4, config.dim});
    assets::require_tensor_shape(weights, "transformer.layers.0.linear2.weight", {config.dim, config.dim * 4});
    assets::require_tensor_shape(weights, "out_norm.weight", {config.dim});
    assets::require_tensor_shape(weights, "linears.0.weight", {config.card, config.dim});
}

}  // namespace

std::shared_ptr<const MuScriptorAssets> load_muscriptor_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<MuScriptorAssets>();
    assets->resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("muscriptor"));
    assets->config = parse_config(assets->resources);
    assets->model_weights = assets->resources.open_tensor_source("weights");
    validate_weight_anchors(*assets);
    return assets;
}

}  // namespace engine::models::muscriptor
