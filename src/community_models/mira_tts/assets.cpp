#include "engine/community_models/mira_tts/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::community_models::mira_tts {
namespace {

namespace json = engine::io::json;
constexpr const char * kFamily = "mira_tts";

MiraTTSConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    if (json::require_string(root, "model_type") != "qwen2") {
        throw std::runtime_error("MiraTTS config must use model_type qwen2");
    }
    MiraTTSConfig out;
    out.hidden_size = json::require_i64(root, "hidden_size");
    out.intermediate_size = json::require_i64(root, "intermediate_size");
    out.layers = json::require_i64(root, "num_hidden_layers");
    out.attention_heads = json::require_i64(root, "num_attention_heads");
    out.kv_heads = json::require_i64(root, "num_key_value_heads");
    out.head_dim = out.hidden_size / out.attention_heads;
    out.vocab_size = json::require_i64(root, "vocab_size");
    out.max_position_embeddings = json::optional_i64(
        root, "max_position_embeddings", out.max_position_embeddings);
    out.rms_norm_eps = json::optional_f32(root, "rms_norm_eps", out.rms_norm_eps);
    out.rope_theta = json::optional_f32(root, "rope_theta", out.rope_theta);
    out.bos_token_id = static_cast<int32_t>(json::optional_i64(
        root, "bos_token_id", out.bos_token_id));
    out.eos_token_id = static_cast<int32_t>(json::optional_i64(
        root, "eos_token_id", out.eos_token_id));
    return out;
}

}  // namespace

std::shared_ptr<const MiraTTSAssets> load_mira_tts_assets(
    const std::filesystem::path & model_path) {
    auto out = std::make_shared<MiraTTSAssets>();
    out->resources = engine::model_spec::load_resource_bundle(
        model_path, engine::model_spec::default_spec_path(kFamily));
    out->config = parse_config(out->resources);
    out->language_model_weights = out->resources.open_tensor_source("language_model");
    out->speaker_encoder_weights = out->resources.open_tensor_source("speaker_encoder");
    out->processor_weights = out->resources.open_tensor_source("processor");
    out->decoder_weights = out->resources.open_tensor_source("decoder");
    out->upsampler_weights = out->resources.open_tensor_source("upsampler");
    return out;
}

}  // namespace engine::community_models::mira_tts
