#include "engine/models/bark_tts/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::models::bark_tts {
namespace {

BarkTransformerConfig transformer_config(const engine::io::json::Value & value) {
    BarkTransformerConfig out;
    out.hidden = engine::io::json::require_i64(value, "hidden_size");
    out.layers = engine::io::json::require_i64(value, "num_layers");
    out.heads = engine::io::json::require_i64(value, "num_heads");
    out.block_size = engine::io::json::require_i64(value, "block_size");
    out.input_vocab = engine::io::json::require_i64(value, "input_vocab_size");
    out.output_vocab = engine::io::json::require_i64(value, "output_vocab_size");
    out.bias = engine::io::json::optional_bool(value, "bias", false);
    return out;
}

std::vector<std::vector<int32_t>> matrix_i32(const engine::io::json::Value & value) {
    std::vector<std::vector<int32_t>> out;
    for (const auto & row : value.as_array()) out.push_back(engine::io::json::number_array_as<int32_t>(row));
    return out;
}

void validate(const BarkAssets & assets) {
    if (assets.config.semantic.hidden != 768 || assets.config.coarse.hidden != 768 ||
        assets.config.fine.hidden != 768 || assets.config.semantic.heads != 12 ||
        assets.config.sample_rate != 24000 || assets.config.codebook_size != 1024) {
        throw std::runtime_error("unsupported Bark architecture; this runtime currently targets suno/bark-small");
    }
    engine::assets::require_tensor_shape(*assets.weights, "semantic.input_embeds_layer.weight",
        {assets.config.semantic.input_vocab, assets.config.semantic.hidden});
    engine::assets::require_tensor_shape(*assets.weights, "coarse_acoustics.input_embeds_layer.weight",
        {assets.config.coarse.input_vocab, assets.config.coarse.hidden});
    engine::assets::require_tensor_shape(*assets.weights, "fine_acoustics.input_embeds_layers.0.weight",
        {assets.config.fine.input_vocab, assets.config.fine.hidden});
    engine::assets::require_tensor_shape(*assets.weights, "codec_model.quantizer.layers.0.codebook.embed",
        {assets.config.codebook_size, assets.config.codebook_dim});
}

}  // namespace

std::shared_ptr<const BarkAssets> load_bark_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<BarkAssets>();
    assets->resources = engine::model_spec::load_resource_bundle_for_family(model_path, "bark_tts");
    const auto config = assets->resources.parse_json("config_json");
    assets->config.semantic = transformer_config(config.require("semantic_config"));
    assets->config.coarse = transformer_config(config.require("coarse_acoustics_config"));
    assets->config.fine = transformer_config(config.require("fine_acoustics_config"));
    const auto & codec = config.require("codec_config");
    assets->config.codebook_size = engine::io::json::require_i64(codec, "codebook_size");
    assets->config.codebook_dim = engine::io::json::require_i64(codec, "codebook_dim");
    assets->config.sample_rate = engine::io::json::require_i64(codec, "sampling_rate");
    assets->weights = assets->resources.open_tensor_source("weights");
    const auto preset_root = assets->resources.parse_json("speaker_presets_json").require("presets");
    for (const auto & [name, value] : preset_root.as_object()) {
        BarkSpeakerPreset preset;
        preset.semantic = engine::io::json::number_array_as<int32_t>(value.require("semantic"));
        preset.coarse = matrix_i32(value.require("coarse"));
        preset.fine = matrix_i32(value.require("fine"));
        assets->presets.emplace(name, std::move(preset));
    }
    validate(*assets);
    return assets;
}

}  // namespace engine::models::bark_tts
