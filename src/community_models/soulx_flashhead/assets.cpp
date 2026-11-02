#include "engine/community_models/soulx_flashhead/assets.h"
#include <stdexcept>

namespace engine::community_models::soulx_flashhead {
Config Config::from_json(const io::json::Value & json, const std::string & variant) {
    Config c;
    c.variant = variant;
    c.dim = io::json::require_i64(json, "dim");
    c.ffn_dim = io::json::require_i64(json, "ffn_dim");
    c.freq_dim = io::json::require_i64(json, "freq_dim");
    c.heads = io::json::require_i64(json, "num_heads");
    c.layers = io::json::require_i64(json, "num_layers");
    c.in_channels = io::json::require_i64(json, "in_dim");
    c.latent_channels = io::json::require_i64(json, "out_dim");
    c.eps = io::json::require_f32(json, "eps");
    if (c.variant != "lite" && c.variant != "pro" && c.variant != "pretrained")
        throw std::runtime_error("Unsupported SoulX-FlashHead variant: " + c.variant);
    if (c.dim <= 0 || c.ffn_dim <= 0 || c.heads <= 0 || c.layers <= 0 || c.freq_dim <= 0 || c.freq_dim % 2 ||
        c.dim % c.heads || c.eps <= 0)
        throw std::runtime_error("Invalid SoulX-FlashHead dimensions");
    for (const auto * key : {"vae_stride", "patch_size"}) {
        const auto & values = json.require(key).as_array();
        if (values.size() != 3)
            throw std::runtime_error("FlashHead stride/patch must have three axes");
        auto & out = std::string(key) == "vae_stride" ? c.stride : c.patch;
        for (size_t i = 0; i < 3; ++i) {
            out[i] = values[i].as_i64();
            if (out[i] <= 0)
                throw std::runtime_error("FlashHead stride/patch must be positive");
        }
    }
    return c;
}
std::shared_ptr<const Assets> load_assets(const std::filesystem::path & path) {
    auto result = std::make_shared<Assets>();
    result->resources = model_spec::load_resource_bundle_for_family(path, "soulx_flashhead");
    const auto json = result->resources.parse_json("config");
    result->config = Config::from_json(json, io::json::require_string(json, "variant"));
    result->dit = result->resources.open_tensor_source("dit");
    result->vae = result->resources.open_tensor_source("vae");
    result->audio = result->resources.open_tensor_source("audio");
    return result;
}
} // namespace engine::community_models::soulx_flashhead
