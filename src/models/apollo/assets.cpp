#include "engine/models/apollo/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::models::apollo {

std::shared_ptr<const ApolloAssets> load_apollo_assets(const std::filesystem::path & path) {
    auto result = std::make_shared<ApolloAssets>();
    result->resources = model_spec::load_resource_bundle(path, model_spec::default_spec_path("apollo"));
    result->tensors = result->resources.open_tensor_source("weights");
    const auto json = result->resources.parse_json("config");
    if (io::json::require_string(json, "model_type") != "apollo") {
        throw std::runtime_error("Apollo config model_type mismatch");
    }
    auto & config = result->config;
    config.sample_rate = io::json::require_i32(json, "sr");
    config.n_fft = config.sample_rate * io::json::require_i32(json, "win") / 1000;
    config.hop_length = config.n_fft / 2;
    config.dim = io::json::require_i32(json, "feature_dim");
    config.layers = io::json::require_i32(json, "layer");
    if (config.sample_rate != 44100 || config.n_fft != 882 || config.dim != 256 || config.layers != 6) {
        throw std::runtime_error("Apollo requires the official 44.1 kHz, 256-channel, six-layer architecture");
    }
    config.band_widths.assign(79, config.n_fft / 160);
    config.band_widths.push_back(config.n_fft / 2 + 1 - 79 * (config.n_fft / 160));
    result->window = result->resources.open_tensor_source("frontend")->require_f32("window", {config.n_fft});
    return result;
}

}  // namespace engine::models::apollo
