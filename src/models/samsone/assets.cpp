#include "engine/models/samsone/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::models::samsone {
namespace {

SamsoneConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    if (io::json::require_string(root, "model_type") != "samsone") {
        throw std::runtime_error("SAMSONE model_type mismatch");
    }
    SamsoneConfig out;
    out.hidden_size = io::json::require_i64(root, "hidden_size");
    out.intermediate_size = io::json::require_i64(root, "intermediate_size");
    out.num_attention_heads = io::json::require_i64(root, "num_attention_heads");
    out.num_key_value_heads = io::json::require_i64(root, "num_key_value_heads");
    out.num_hidden_layers = io::json::require_i64(root, "num_hidden_layers");
    out.vocab_size = io::json::require_i64(root, "vocab_size");
    out.max_position_embeddings = io::json::require_i64(root, "max_position_embeddings");
    out.rms_norm_eps = io::json::require_f32(root, "rms_norm_eps");
    out.rope_theta = io::json::require_f32(root, "rope_theta");
    out.eos_token_id = static_cast<int32_t>(io::json::require_i64(root, "eos_token_id"));
    if (out.hidden_size <= 0 || out.intermediate_size <= 0 || out.num_attention_heads <= 0 ||
        out.num_key_value_heads <= 0 || out.num_hidden_layers <= 0 || out.vocab_size <= 0 ||
        out.max_position_embeddings <= 0 || out.hidden_size % out.num_attention_heads != 0 ||
        out.num_attention_heads % out.num_key_value_heads != 0) {
        throw std::runtime_error("SAMSONE config contains an unsupported decoder shape");
    }
    return out;
}

}  // namespace

std::shared_ptr<const SamsoneAssets> load_samsone_assets(const std::filesystem::path & model_path) {
    auto resources = model_spec::load_resource_bundle_for_family(model_path, "samsone");
    auto out = std::make_shared<SamsoneAssets>();
    out->config = parse_config(resources);
    out->weights = resources.open_tensor_source("weights");
    if (out->weights->tensors().empty()) {
        throw std::runtime_error("SAMSONE weights are empty");
    }
    out->resources = std::move(resources);
    return out;
}

}  // namespace engine::models::samsone
