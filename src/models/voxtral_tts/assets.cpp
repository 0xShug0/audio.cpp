#include "engine/models/voxtral_tts/assets.h"

#include "engine/framework/io/json.h"

#include <sstream>

namespace engine::models::voxtral_tts {
namespace {

TransformerConfig transformer_config(const io::json::Value & value) {
    TransformerConfig config;
    config.hidden_size = io::json::require_i64(value, "dim");
    config.intermediate_size = io::json::require_i64(value, "hidden_dim");
    config.attention_heads = io::json::require_i64(value, "n_heads");
    config.kv_heads = io::json::require_i64(value, "n_kv_heads");
    config.head_dim = io::json::require_i64(value, "head_dim");
    if (const auto * layers = value.find("n_layers")) {
        config.layers = layers->as_i64();
    }
    if (const auto * eps = value.find("norm_eps")) {
        config.norm_eps = eps->as_f32();
    }
    if (const auto * theta = value.find("rope_theta")) {
        config.rope_theta = theta->as_f32();
    }
    return config;
}

std::vector<int> config_list(const io::json::Value & value, const char * key) {
    std::istringstream stream(io::json::require_string(value, key));
    std::vector<int> result;
    std::string item;
    while (std::getline(stream, item, ',')) {
        result.push_back(std::stoi(item));
    }
    return result;
}

}  // namespace

std::shared_ptr<const VoxtralTtsAssets> load_assets(const std::filesystem::path & path) {
    auto assets = std::make_shared<VoxtralTtsAssets>();
    assets->resources = model_spec::load_resource_bundle_for_family(path, "voxtral_tts");
    assets->weights = assets->resources.open_tensor_source("model_weights");
    const auto root = assets->resources.parse_json("config");
    if (io::json::require_string(root, "model_type") != "voxtral_tts") {
        throw std::runtime_error("Voxtral TTS requires a voxtral_tts checkpoint");
    }
    auto & config = assets->config;
    config.ar = transformer_config(root);
    config.vocab_size = io::json::require_i64(root, "vocab_size");
    const auto & multimodal = root.require("multimodal");
    const auto & generation = multimodal.require("audio_model_args");
    config.acoustic = transformer_config(generation.require("acoustic_transformer_args"));
    config.semantic_vocab_size = io::json::require_i64(generation, "semantic_codebook_size");
    config.acoustic_dim = io::json::require_i64(generation, "n_acoustic_codebook");
    config.acoustic_levels = io::json::require_i64(generation, "acoustic_codebook_size");
    const auto & codec = multimodal.require("audio_tokenizer_args");
    config.codec = transformer_config(codec);
    config.semantic_dim = io::json::require_i64(codec, "semantic_dim");
    config.sample_rate = io::json::require_i64(codec, "sampling_rate");
    config.patch_size = io::json::require_i64(codec, "pretransform_patch_size");
    config.patch_kernel_size = io::json::require_i64(codec, "patch_proj_kernel_size");
    config.codec_qk_norm_eps = io::json::require_f32(codec, "qk_norm_eps");
    config.codec_layers = config_list(codec, "decoder_transformer_lengths_str");
    config.codec_kernels = config_list(codec, "decoder_convs_kernels_str");
    config.codec_strides = config_list(codec, "decoder_convs_strides_str");
    int window = static_cast<int>(io::json::require_i64(codec, "attn_sliding_window_size"));
    for (const auto stride : config.codec_strides) {
        if (stride > 1) {
            window /= stride;
        }
    }
    for (const auto stride : config.codec_strides) {
        window *= stride;
        config.codec_windows.push_back(window);
    }
    for (const auto & [voice, id] : codec.require("voice").as_object()) {
        (void)id;
        const auto name = "voices." + voice + ".prompt_embedding";
        const auto shape = assets->weights->require_metadata(name).shape;
        if (shape.size() != 2 || shape[1] != config.ar.hidden_size) {
            throw std::runtime_error("Voxtral TTS voice embedding shape is invalid: " + voice);
        }
        assets->voices.emplace(voice, assets->weights->require_f32(name));
    }
    return assets;
}

}  // namespace engine::models::voxtral_tts
