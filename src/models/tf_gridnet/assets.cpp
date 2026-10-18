#include "engine/models/tf_gridnet/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::models::tf_gridnet {

std::shared_ptr<const TFGridNetAssets> load_tf_gridnet_assets(const std::filesystem::path & path) {
    auto assets = std::make_shared<TFGridNetAssets>();
    assets->resources = model_spec::load_resource_bundle(path, model_spec::default_spec_path("tf_gridnet"));
    assets->tensors = assets->resources.open_tensor_source("weights");
    const auto json = assets->resources.parse_json("config");
    if (io::json::require_string(json, "model_type") != "tf_gridnet") {
        throw std::runtime_error("TF-GridNet config model_type mismatch");
    }
    auto & config = assets->config;
    config.sample_rate = io::json::require_i32(json, "sample_rate");
    config.speakers = io::json::require_i32(json, "n_srcs");
    config.microphones = io::json::require_i32(json, "n_imics");
    config.n_fft = io::json::require_i32(json, "n_fft");
    config.hop_length = io::json::require_i32(json, "stride");
    config.layers = io::json::require_i32(json, "n_layers");
    config.embedding_dim = io::json::require_i32(json, "emb_dim");
    config.embedding_kernel = io::json::require_i32(json, "emb_ks");
    config.embedding_stride = io::json::require_i32(json, "emb_hs");
    config.lstm_hidden = io::json::require_i32(json, "lstm_hidden_units");
    config.attention_heads = io::json::require_i32(json, "attn_n_head");
    config.attention_qk_dim = io::json::require_i32(json, "attn_approx_qk_dim");
    config.epsilon = io::json::require_f32(json, "eps");
    if (config.sample_rate <= 0 || config.speakers <= 0 || config.microphones <= 0 ||
        config.n_fft <= 0 || config.n_fft % 2 || config.hop_length <= 0 ||
        config.layers <= 0 || config.embedding_dim <= 0 || config.embedding_kernel <= 0 ||
        config.embedding_stride <= 0 || config.lstm_hidden <= 0 || config.attention_heads <= 0 ||
        config.embedding_dim % config.attention_heads || config.attention_qk_dim <= 0 || config.epsilon <= 0) {
        throw std::runtime_error("TF-GridNet configuration has invalid architecture dimensions");
    }
    if (io::json::require_string(json, "activation") != "prelu") {
        throw std::runtime_error("TF-GridNet requires PReLU activation");
    }
    assets->window = assets->resources.open_tensor_source("frontend")->require_f32("window", {config.n_fft});
    return assets;
}

}  // namespace engine::models::tf_gridnet
