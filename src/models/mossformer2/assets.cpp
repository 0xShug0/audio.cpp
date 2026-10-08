#include "engine/models/mossformer2/assets.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

namespace engine::models::mossformer2 {

std::shared_ptr<const MossFormer2Assets> load_mossformer2_assets(const std::filesystem::path & path) {
    auto assets = std::make_shared<MossFormer2Assets>();
    assets->resources = model_spec::load_resource_bundle(path, model_spec::default_spec_path("mossformer2"));
    assets->tensors = assets->resources.open_tensor_source("weights");
    const auto json = assets->resources.parse_json("config");
    if (io::json::require_string(json, "model_type") != "mossformer2") {
        throw std::runtime_error("MossFormer2 config model_type mismatch");
    }
    auto & c = assets->config;
    c.sample_rate = io::json::require_i32(json, "sampling_rate");
    c.speakers = io::json::require_i32(json, "num_spks");
    c.kernel = io::json::require_i32(json, "encoder_kernel_size");
    c.channels = io::json::require_i32(json, "encoder_embedding_dim");
    c.hidden = io::json::require_i32(json, "mossformer_sequence_dim");
    c.layers = io::json::require_i32(json, "num_mossformer_layer");
    if (c.sample_rate <= 0 || c.speakers <= 0 || c.kernel < 2 || c.kernel % 2 ||
        c.channels <= 0 || c.hidden <= 0 || c.hidden % 2 || c.layers <= 0) {
        throw std::runtime_error("MossFormer2 invalid architecture dimensions");
    }
    return assets;
}

}  // namespace engine::models::mossformer2
