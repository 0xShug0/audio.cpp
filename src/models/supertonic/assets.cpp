#include "engine/models/supertonic/assets.h"

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/io/filesystem.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::supertonic {
namespace json = engine::io::json;
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("Supertonic model path does not exist: " + model_path.string());
}

assets::ResourceBundle make_resource_bundle(const std::filesystem::path & model_path) {
    assets::ResourceBundle resources(resolve_model_root(model_path));
    resources.add_model_files({
        {"tts_config", "config/tts.json", true},
        {"unicode_indexer", "config/unicode_indexer.json", true},
        {"weights", "ggml/supertonic.safetensors", true},
    });
    return resources;
}

void require_positive(int64_t value, const char * label) {
    if (value <= 0) {
        throw std::runtime_error(std::string("Supertonic config contains non-positive ") + label);
    }
}

SupertonicConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("tts_config");
    SupertonicConfig config;
    const auto & ae = root.require("ae");
    const auto & ttl = root.require("ttl");
    config.sample_rate = static_cast<int>(json::require_i64(ae, "sample_rate"));
    config.base_chunk_size = json::require_i64(ae, "base_chunk_size");
    config.chunk_compress_factor = json::require_i64(ttl, "chunk_compress_factor");
    config.latent_dim = json::require_i64(ttl, "latent_dim");
    require_positive(config.sample_rate, "sample_rate");
    require_positive(config.base_chunk_size, "base_chunk_size");
    require_positive(config.chunk_compress_factor, "chunk_compress_factor");
    require_positive(config.latent_dim, "latent_dim");
    return config;
}

std::unordered_map<uint32_t, int64_t> parse_unicode_indexer(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("unicode_indexer");
    std::unordered_map<uint32_t, int64_t> out;
    if (root.is_array()) {
        const auto & values = root.as_array();
        out.reserve(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            const int64_t token = values[i].as_i64();
            if (token >= 0) {
                out.emplace(static_cast<uint32_t>(i), token);
            }
        }
    } else if (root.is_object()) {
        out.reserve(root.as_object().size());
        for (const auto & [key, value] : root.as_object()) {
            size_t parsed = 0;
            const auto codepoint = static_cast<uint32_t>(std::stoul(key, &parsed, 10));
            if (parsed != key.size()) {
                throw std::runtime_error("Supertonic unicode indexer contains a non-numeric codepoint key: " + key);
            }
            const int64_t token = value.as_i64();
            if (token >= 0) {
                out.emplace(codepoint, token);
            }
        }
    } else {
        throw std::runtime_error("Supertonic unicode indexer must be an array or object");
    }
    if (out.empty()) {
        throw std::runtime_error("Supertonic unicode indexer is empty");
    }
    return out;
}

void fill_paths(SupertonicAssetPaths & paths, assets::ResourceBundle & resources) {
    paths.model_root = resources.model_root();
    paths.tts_config_path = resources.require_file("tts_config");
    paths.unicode_indexer_path = resources.require_file("unicode_indexer");
    paths.weights_path = resources.require_file("weights");
    paths.voice_styles_dir = paths.model_root / "voice_styles";
    if (!engine::io::is_existing_directory(paths.voice_styles_dir)) {
        throw std::runtime_error("missing Supertonic voice_styles directory: " + paths.voice_styles_dir.string());
    }
}

}  // namespace

SupertonicAssetPaths resolve_supertonic_assets(const std::filesystem::path & model_path) {
    auto resources = make_resource_bundle(model_path);
    SupertonicAssetPaths paths;
    fill_paths(paths, resources);
    return paths;
}

std::shared_ptr<const SupertonicAssets> load_supertonic_assets(const std::filesystem::path & model_path) {
    auto resources = make_resource_bundle(model_path);
    auto assets = std::make_shared<SupertonicAssets>();
    fill_paths(assets->paths, resources);
    assets->config = parse_config(resources);
    assets->weights = resources.open_tensor_source("weights");
    assets->unicode_indexer = parse_unicode_indexer(resources);
    return assets;
}

}  // namespace engine::models::supertonic
