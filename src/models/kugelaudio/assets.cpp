#include "engine/models/kugelaudio/assets.h"

#include "engine/framework/io/json.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace engine::models::kugelaudio {
namespace {

ModelConfig parse_model_config(const io::json::Value & root) {
    namespace json = io::json;
    if (json::require_string(root, "model_type") != "kugelaudio") {
        throw std::runtime_error("KugelAudio requires a kugelaudio configuration");
    }
    ModelConfig result;
    const auto & decoder = root.require("decoder_config");
    auto & ar = result.ar;
    ar.hidden_size = json::require_i64(decoder, "hidden_size");
    ar.intermediate_size = json::require_i64(decoder, "intermediate_size");
    ar.attention_heads = json::require_i64(decoder, "num_attention_heads");
    ar.kv_heads = json::require_i64(decoder, "num_key_value_heads");
    ar.layers = json::require_i64(decoder, "num_hidden_layers");
    ar.vocab_size = json::require_i64(decoder, "vocab_size");
    ar.max_position_embeddings = json::require_i64(decoder, "max_position_embeddings");
    ar.norm_eps = json::require_f32(decoder, "rms_norm_eps");
    ar.rope_theta = json::require_f32(decoder, "rope_theta");
    if (ar.attention_heads <= 0 || ar.hidden_size % ar.attention_heads != 0) {
        throw std::runtime_error("KugelAudio hidden size must be divisible by attention heads");
    }
    ar.head_dim = ar.hidden_size / ar.attention_heads;

    const auto & head = root.require("diffusion_head_config");
    auto & diffusion = result.diffusion;
    diffusion.hidden_size = json::require_i64(head, "hidden_size");
    diffusion.latent_size = json::require_i64(head, "latent_size");
    diffusion.layers = json::require_i64(head, "head_layers");
    diffusion.intermediate_size = static_cast<int64_t>(
        diffusion.hidden_size * json::require_f32(head, "head_ffn_ratio"));
    diffusion.norm_eps = json::require_f32(head, "rms_norm_eps");
    result.inference_steps = json::require_i32(head, "ddpm_num_inference_steps");

    const auto & tokenizer = root.require("acoustic_tokenizer_config");
    auto & codec = result.codec;
    codec.latent_size = json::require_i64(tokenizer, "vae_dim");
    codec.filters = json::require_i64(tokenizer, "decoder_n_filters");
    codec.ratios = json::number_array_as<int>(tokenizer.require("decoder_ratios"));
    codec.norm_eps = json::require_f32(tokenizer, "layernorm_eps");
    const auto * depths = tokenizer.find("decoder_depths");
    const bool reverse = depths == nullptr || depths->is_null();
    const auto & depth_value = reverse ? tokenizer.require("encoder_depths") : *depths;
    codec.depths.clear();
    if (depth_value.is_string()) {
        std::istringstream stream(depth_value.as_string());
        std::string depth;
        while (std::getline(stream, depth, '-')) {
            codec.depths.push_back(std::stoi(depth));
        }
    } else {
        codec.depths = json::number_array_as<int>(depth_value);
    }
    if (reverse) {
        std::reverse(codec.depths.begin(), codec.depths.end());
    }
    if (codec.depths.size() != codec.ratios.size() + 1 ||
        codec.latent_size != diffusion.latent_size || diffusion.hidden_size != ar.hidden_size) {
        throw std::runtime_error("KugelAudio component dimensions are inconsistent");
    }
    return result;
}

}  // namespace

std::shared_ptr<const ModelAssets> load_assets(const std::filesystem::path & path) {
    auto result = std::make_shared<ModelAssets>();
    result->resources = model_spec::load_resource_bundle_for_family(path, "kugelaudio");
    result->config = parse_model_config(result->resources.parse_json("config"));
    result->weights = result->resources.open_tensor_source("model_weights");
    for (const auto * name : {"default", "clear", "english_female", "english_male"}) {
        const auto prefix = std::string("voices.") + name + ".";
        VoiceFeatures voice;
        voice.mean = result->weights->require_f32(prefix + "acoustic_mean");
        voice.std = result->weights->require_f32(prefix + "acoustic_std", {1}).front();
        result->voices.emplace(name, std::move(voice));
    }
    return result;
}

}  // namespace engine::models::kugelaudio
