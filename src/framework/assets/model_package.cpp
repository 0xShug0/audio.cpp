#include "engine/framework/assets/model_package.h"

#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <optional>

namespace engine::assets {
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("model path does not exist: " + model_path.string());
}

std::string require_source_format(const engine::io::json::Value & source) {
    return engine::io::json::require_string(source, "format");
}

using ResourceRoots = std::unordered_map<std::string, std::filesystem::path>;

ResourceRoots resolve_source_roots(
    const std::filesystem::path & model_root,
    const engine::io::json::Value & source,
    const std::optional<std::filesystem::path> & standalone_gguf) {
    ResourceRoots roots;
    const auto & root_object = source.require("roots").as_object();
    for (const auto & [id, value] : root_object) {
        const auto root_value = value.as_string();
        if (root_value == "$gguf" && !standalone_gguf.has_value()) {
            throw std::runtime_error("model package source requires a GGUF root: " + id);
        }
        const auto root_path = root_value == "$gguf"
            ? *standalone_gguf
            : model_root / root_value;
        if (!engine::io::is_existing_directory(root_path) && !engine::io::is_existing_file(root_path)) {
            throw std::runtime_error("missing model package root: " + id + "=" + root_path.string());
        }
        roots.emplace(id, std::filesystem::weakly_canonical(root_path));
    }
    return roots;
}

std::filesystem::path resolve_resource_ref(
    const std::unordered_map<std::string, std::filesystem::path> & roots,
    const std::string & ref) {
    const auto split = ref.find(':');
    if (split == std::string::npos || split == 0) {
        throw std::runtime_error("invalid model package resource reference: " + ref);
    }
    const auto root_id = ref.substr(0, split);
    const auto relative_path = ref.substr(split + 1);
    const auto root_it = roots.find(root_id);
    if (root_it == roots.end()) {
        throw std::runtime_error("unknown model package resource root: " + root_id);
    }
    if (relative_path.empty()) {
        return root_it->second;
    }
    return root_it->second / relative_path;
}

void add_resource_map(
    ResourceBundle & bundle,
    const ResourceRoots & roots,
    const engine::io::json::Value * map_value) {
    if (map_value == nullptr || map_value->is_null()) {
        return;
    }
    for (const auto & [id, ref] : map_value->as_object()) {
        const auto path = resolve_resource_ref(roots, ref.as_string());
        if (!engine::io::is_existing_file(path)) {
            throw std::runtime_error("missing model package file '" + id + "': " + path.string());
        }
        bundle.add_file(id, path);
    }
}

std::filesystem::path resolve_tensor_source_ref(
    const ResourceRoots & roots,
    const engine::io::json::Value & value,
    std::string & prefix) {
    if (value.is_string()) {
        prefix.clear();
        return resolve_resource_ref(roots, value.as_string());
    }
    const auto & object = value.as_object();
    const auto source_it = object.find("source");
    if (source_it == object.end()) {
        throw std::runtime_error("model package tensor source object requires source");
    }
    const auto prefix_it = object.find("prefix");
    prefix = prefix_it == object.end() ? "" : prefix_it->second.as_string();
    return resolve_resource_ref(roots, source_it->second.as_string());
}

void add_tensor_map(
    ResourceBundle & bundle,
    const ResourceRoots & roots,
    const engine::io::json::Value * map_value) {
    if (map_value == nullptr || map_value->is_null()) {
        return;
    }
    for (const auto & [id, ref] : map_value->as_object()) {
        std::string prefix;
        const auto path = resolve_tensor_source_ref(roots, ref, prefix);
        if (!engine::io::is_existing_file(path)) {
            throw std::runtime_error("missing model package tensor source '" + id + "': " + path.string());
        }
        bundle.add_tensor_source(id, path, std::move(prefix));
    }
}

void add_optional_resource_map(
    ResourceBundle & bundle,
    const ResourceRoots & roots,
    const engine::io::json::Value * map_value) {
    if (map_value == nullptr || map_value->is_null()) {
        return;
    }
    for (const auto & [id, ref] : map_value->as_object()) {
        const auto path = resolve_resource_ref(roots, ref.as_string());
        if (engine::io::is_existing_file(path)) {
            bundle.add_file(id, path);
        }
    }
}

std::vector<ResourceFile> resources_from_resource_map(
    const ResourceRoots & roots,
    const engine::io::json::Value * map_value,
    bool required) {
    std::vector<ResourceFile> assets;
    if (map_value == nullptr || map_value->is_null()) {
        return assets;
    }
    const auto & map = map_value->as_object();
    assets.reserve(map.size());
    for (const auto & [id, ref] : map) {
        std::string prefix;
        const auto path = resolve_tensor_source_ref(roots, ref, prefix);
        if (required || engine::io::is_existing_file(path)) {
            assets.push_back({id, std::filesystem::weakly_canonical(path)});
        }
    }
    return assets;
}

ResourceBundle load_source(
    const std::filesystem::path & model_root,
    const engine::io::json::Value & source,
    const ResourceRoots & roots) {
    ResourceBundle bundle(model_root);
    add_resource_map(bundle, roots, source.find("files"));
    add_optional_resource_map(bundle, roots, source.find("optional_files"));
    add_tensor_map(bundle, roots, source.find("tensors"));
    return bundle;
}

std::vector<ResourceFile> discover_safetensors_source_resources(
    const engine::io::json::Value & source,
    ModelPackageResourceKind kind,
    const ResourceRoots & roots) {
    auto resources = resources_from_resource_map(
        roots,
        source.find(kind == ModelPackageResourceKind::Files ? "files" : "tensors"),
        true);
    if (kind == ModelPackageResourceKind::Files) {
        auto optional = resources_from_resource_map(roots, source.find("optional_files"), false);
        resources.insert(resources.end(), optional.begin(), optional.end());
    }
    return resources;
}

struct SelectedSource {
    std::filesystem::path model_root;
    engine::io::json::Value source;
    ResourceRoots roots;
};

SelectedSource select_source(
    const std::filesystem::path & model_path,
    const std::filesystem::path & model_root,
    const engine::io::json::Value & source) {
    const auto format = require_source_format(source);
    if (format == "gguf") {
        const auto prepared = prepare_model_directory(model_path);
        auto roots = resolve_source_roots(prepared.model_root, source, prepared.standalone_gguf);
        return SelectedSource{prepared.model_root, source, std::move(roots)};
    }
    if (format == "safetensors") {
        auto roots = resolve_source_roots(model_root, source, std::nullopt);
        return SelectedSource{model_root, source, std::move(roots)};
    }
    throw std::runtime_error("unsupported model package source format: " + format);
}

SelectedSource require_selected_source(
    const std::filesystem::path & model_path,
    const std::filesystem::path & spec_path) {
    const auto model_root = resolve_model_root(model_path);
    const auto spec = engine::io::json::parse_file(spec_path);
    const auto & sources = spec.require("sources").as_array();
    const auto prepared = prepare_model_directory(model_path);
    const bool explicit_gguf_path = engine::io::is_existing_file(model_path) &&
        model_path.extension() == ".gguf";
    const bool directory_has_gguf = engine::io::is_existing_directory(model_path) &&
        engine::io::is_existing_file(model_path / "model.gguf");
    const bool use_gguf = explicit_gguf_path || directory_has_gguf;
    if (use_gguf && !prepared.standalone_gguf.has_value()) {
        throw std::runtime_error("model package GGUF source requires embedded sidecars: " + model_path.string());
    }
    for (const auto & source : sources) {
        const auto format = require_source_format(source);
        if ((use_gguf && format == "gguf") || (!use_gguf && format == "safetensors")) {
            return select_source(model_path, model_root, source);
        }
    }
    throw std::runtime_error(
        std::string("no ") + (use_gguf ? "gguf" : "safetensors") +
        " model package source in " + spec_path.string());
}

}  // namespace

std::filesystem::path default_model_package_spec_path(std::string_view family) {
    const std::filesystem::path relative = std::filesystem::path("model_specs") / (std::string(family) + ".json");
    auto cursor = std::filesystem::current_path();
    while (true) {
        const auto candidate = cursor / relative;
        if (engine::io::is_existing_file(candidate)) {
            return std::filesystem::weakly_canonical(candidate);
        }
        const auto parent = cursor.parent_path();
        if (parent == cursor || parent.empty()) {
            break;
        }
        cursor = parent;
    }
    throw std::runtime_error("model package spec not found: " + relative.string());
}

ResourceBundle load_resource_bundle_from_package_spec(
    const std::filesystem::path & model_path,
    const std::filesystem::path & spec_path) {
    auto selected = require_selected_source(model_path, spec_path);
    return load_source(selected.model_root, selected.source, selected.roots);
}

std::vector<ResourceFile> discover_resources_from_package_spec(
    const std::filesystem::path & model_path,
    const std::filesystem::path & spec_path,
    ModelPackageResourceKind kind) {
    auto selected = require_selected_source(model_path, spec_path);
    return discover_safetensors_source_resources(selected.source, kind, selected.roots);
}

}  // namespace engine::assets
