#include "engine/models/personaplex/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <utility>

namespace engine::models::personaplex {
namespace json = engine::io::json;
namespace {

PersonaPlexConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    PersonaPlexConfig config;
    config.model_type = json::optional_string(root, "model_type", config.model_type);
    config.version = json::optional_string(root, "version", config.version);
    if (config.model_type != "personaplex") {
        throw std::runtime_error("PersonaPlex config model_type must be personaplex");
    }
    return config;
}

void add_voice_prompt_path(
    std::unordered_map<std::string, std::filesystem::path> & prompts,
    const assets::ResourceBundle & resources,
    const std::string & id) {
    const auto * path = resources.find_file("voice_" + id);
    if (path != nullptr) {
        prompts.emplace(id, *path);
    }
}

}  // namespace

std::shared_ptr<const PersonaPlexAssets> load_personaplex_assets(const std::filesystem::path & model_path) {
    auto resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("personaplex"));

    auto assets = std::make_shared<PersonaPlexAssets>();
    assets->config = parse_config(resources);
    assets->tokenizer_pieces = engine::tokenizers::load_sentencepiece_model(resources.require_file("tokenizer_model"));
    assets->lm_weights = resources.open_tensor_source("lm_weights");
    assets->mimi_weights = resources.open_tensor_source("mimi_weights");
    for (const std::string id : {
             "NATF0", "NATF1", "NATF2", "NATF3",
             "NATM0", "NATM1", "NATM2", "NATM3",
             "VARF0", "VARF1", "VARF2", "VARF3", "VARF4",
             "VARM0", "VARM1", "VARM2", "VARM3", "VARM4",
         }) {
        add_voice_prompt_path(assets->voice_prompt_paths, resources, id);
    }
    assets->resources = std::move(resources);
    return assets;
}

}  // namespace engine::models::personaplex
