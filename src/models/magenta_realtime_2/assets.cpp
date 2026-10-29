#include "engine/models/magenta_realtime_2/assets.h"

#include "engine/framework/io/config.h"
#include "engine/framework/model_spec/package.h"

namespace engine::models::magenta_realtime_2 {

std::shared_ptr<const MagentaAssets> load_magenta_realtime_2_assets(const std::filesystem::path & path) {
    auto result = std::make_shared<MagentaAssets>();
    result->resources = model_spec::load_resource_bundle_for_family(path, "magenta_realtime_2");
    const auto config = result->resources.parse_json("config");
    auto & c = result->config;
    c.variant = io::json::require_string(config, "variant");
    c.temporal_layers = io::json::require_i64(config, "temporal_layers");
    c.temporal_dim = io::json::require_i64(config, "temporal_dim");
    c.temporal_heads = io::json::require_i64(config, "temporal_heads");
    c.temporal_ffn_dim = io::json::require_i64(config, "temporal_ffn_dim");
    c.condition_dim = io::json::require_i64(config, "condition_dim");
    c.window = io::json::require_i64(config, "window");
    c.depth_layers = io::json::require_i64(config, "depth_layers");
    c.depth_dim = io::json::require_i64(config, "depth_dim");
    c.depth_heads = io::json::require_i64(config, "depth_heads");
    c.depth_ffn_dim = io::json::require_i64(config, "depth_ffn_dim");
    for (auto value : {c.temporal_layers, c.temporal_dim, c.temporal_heads, c.temporal_ffn_dim,
                       c.condition_dim, c.window, c.depth_layers, c.depth_dim, c.depth_heads, c.depth_ffn_dim})
        io::require_positive(value, "Magenta RealTime 2 configuration dimension");
    if (c.variant != "small" && c.variant != "base") throw std::runtime_error("Magenta RealTime 2 variant must be small or base");
    result->weights = result->resources.open_tensor_source("weights");
    engine::assets::require_tensor_shape(*result->weights, "token_embedding.weight", {12294, c.temporal_dim});
    engine::assets::require_tensor_shape(*result->weights, "depth.to_logits.weight", {12294, c.depth_dim});
    engine::assets::require_tensor_shape(*result->weights, "codec.quantizer.weight", {12, 1024, 256});
    return result;
}

}  // namespace engine::models::magenta_realtime_2
