#include "engine/framework/assets/model_package.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace engine::assets {
namespace {

thread_local std::optional<std::filesystem::path> active_model_spec_override;
thread_local std::optional<std::filesystem::path> active_model_path;
thread_local bool active_embedded_spec_checked = false;
thread_local std::optional<GgufEmbeddedModelSpec> active_embedded_spec;

const std::unordered_map<std::string, std::string_view> & builtin_model_specs() {
    static const std::unordered_map<std::string, std::string_view> specs = {
#include "model_package_specs.inc"
    };
    return specs;
}

std::optional<std::string_view> builtin_model_spec(const std::filesystem::path & spec_path) {
    if (spec_path.parent_path() != "@builtin")
        return std::nullopt;
    const auto it = builtin_model_specs().find(spec_path.stem().string());
    if (it == builtin_model_specs().end())
        return std::nullopt;
    return it->second;
}

std::string model_spec_description(const std::filesystem::path & spec_path) {
    if (spec_path.parent_path() == "@gguf") {
        return "embedded GGUF model package spec for family '" + spec_path.stem().string() + "'";
    }
    if (spec_path.parent_path() == "@builtin") {
        return "builtin model package spec for family '" + spec_path.stem().string() + "'";
    }
    return "model package spec '" + spec_path.string() + "'";
}

bool is_gguf_file(const std::filesystem::path & path) {
    if (!engine::io::is_existing_file(path))
        return false;
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension == ".gguf";
}

std::vector<std::string> directory_gguf_files(const std::filesystem::path & path) {
    std::vector<std::string> files;
    if (!engine::io::is_existing_directory(path)) {
        return files;
    }
    for (const auto & entry : std::filesystem::directory_iterator(path)) {
        const auto candidate = entry.path();
        if (is_gguf_file(candidate)) {
            files.push_back(candidate.filename().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::string directory_gguf_hint(std::string_view family) {
    if (!active_model_path.has_value()) {
        return {};
    }
    const auto files = directory_gguf_files(*active_model_path);
    if (files.empty()) {
        return {};
    }
    std::string message = "model directory has no default GGUF for family '" + std::string(family) +
                          "': " + active_model_path->string() + "; found: ";
    for (size_t i = 0; i < files.size(); ++i) {
        if (i != 0) {
            message += ", ";
        }
        message += files[i];
    }
    message += "; pass the GGUF file directly with --model, or rename it to model.gguf";
    return message;
}

std::optional<std::filesystem::path> active_gguf_path() {
    if (!active_model_path.has_value())
        return std::nullopt;
    const auto & path = *active_model_path;
    if (is_gguf_file(path)) {
        return std::filesystem::weakly_canonical(path);
    }
    if (engine::io::is_existing_directory(path) && engine::io::is_existing_file(path / "model.gguf")) {
        return std::filesystem::weakly_canonical(path / "model.gguf");
    }
    return std::nullopt;
}

const std::optional<GgufEmbeddedModelSpec> & embedded_model_spec() {
    if (!active_embedded_spec_checked) {
        active_embedded_spec_checked = true;
        if (const auto gguf = active_gguf_path()) {
            active_embedded_spec = read_gguf_embedded_model_spec(*gguf);
        }
    }
    return active_embedded_spec;
}

bool external_spec_matches_family(const std::filesystem::path & path, std::string_view family) {
    if (!engine::io::is_existing_file(path))
        return false;
    try {
        const auto root = engine::io::json::parse_file(path);
        return engine::io::json::require_string(root, "family") == family;
    } catch (const std::exception & error) {
        throw std::runtime_error("invalid candidate model package spec '" + path.string() +
                                 "' while resolving family '" + std::string(family) + "': " + error.what());
    }
}

std::optional<std::filesystem::path> discover_external_model_spec(std::string_view family) {
    std::vector<std::filesystem::path> candidates;
    if (active_model_path.has_value()) {
        const auto root = engine::io::is_existing_directory(*active_model_path) ? *active_model_path
                                                                                : active_model_path->parent_path();
        candidates.push_back(root / "model_specs" / (std::string(family) + ".json"));
        candidates.push_back(root / (std::string(family) + ".json"));
        candidates.push_back(root / "model_spec.json");
        candidates.push_back(root.parent_path() / "model_specs" / (std::string(family) + ".json"));
    }
    auto cursor = std::filesystem::current_path();
    while (true) {
        candidates.push_back(cursor / "model_specs" / (std::string(family) + ".json"));
        const auto parent = cursor.parent_path();
        if (parent == cursor || parent.empty())
            break;
        cursor = parent;
    }
    for (const auto & candidate : candidates) {
        if (external_spec_matches_family(candidate, family)) {
            return std::filesystem::weakly_canonical(candidate);
        }
    }
    return std::nullopt;
}

engine::io::json::Value parse_model_spec(const std::filesystem::path & spec_path) {
    if (spec_path.parent_path() == "@gguf") {
        const auto & spec = embedded_model_spec();
        if (!spec.has_value() || spec->family != spec_path.stem().string()) {
            throw std::runtime_error("embedded GGUF model package spec is not available for: " +
                                     spec_path.stem().string());
        }
        auto root = engine::io::json::parse(spec->json);
        if (engine::io::json::require_string(root, "family") != spec->family) {
            throw std::runtime_error("embedded GGUF model package spec family metadata does not match its JSON");
        }
        return root;
    }
    if (const auto text = builtin_model_spec(spec_path)) {
        return engine::io::json::parse(*text);
    }
    return engine::io::json::parse_file(spec_path);
}

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

ResourceRoots resolve_source_roots(const std::filesystem::path & model_root, const engine::io::json::Value & source,
    const std::optional<std::filesystem::path> & standalone_gguf) {
    ResourceRoots roots;
    const auto & root_object = source.require("roots").as_object();
    for (const auto & [id, value] : root_object) {
        const auto root_value = value.as_string();
        if (root_value == "$gguf" && !standalone_gguf.has_value()) {
            throw std::runtime_error("model package source requires a GGUF root: " + id);
        }
        const auto root_path = root_value == "$gguf" ? *standalone_gguf : model_root / root_value;
        if (!engine::io::is_existing_directory(root_path) && !engine::io::is_existing_file(root_path)) {
            throw std::runtime_error("missing model package root: " + id + "=" + root_path.string());
        }
        roots.emplace(id, std::filesystem::weakly_canonical(root_path));
    }
    return roots;
}

std::filesystem::path resolve_resource_ref(const std::unordered_map<std::string, std::filesystem::path> & roots,
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

void add_resource_map(ResourceBundle & bundle, const ResourceRoots & roots, const engine::io::json::Value * map_value) {
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

std::filesystem::path resolve_tensor_source_ref(const ResourceRoots & roots, const engine::io::json::Value & value,
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

void add_tensor_map(ResourceBundle & bundle, const ResourceRoots & roots, const engine::io::json::Value * map_value) {
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

void add_optional_resource_map(ResourceBundle & bundle, const ResourceRoots & roots,
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

std::vector<ResourceFile> resources_from_resource_map(const ResourceRoots & roots,
                                                      const engine::io::json::Value * map_value, bool required) {
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

ResourceBundle load_source(const std::filesystem::path & model_root, const engine::io::json::Value & source,
    const ResourceRoots & roots) {
    ResourceBundle bundle(model_root);
    add_resource_map(bundle, roots, source.find("files"));
    add_optional_resource_map(bundle, roots, source.find("optional_files"));
    add_tensor_map(bundle, roots, source.find("tensors"));
    return bundle;
}

std::vector<ResourceFile> discover_safetensors_source_resources(const engine::io::json::Value & source,
    ModelPackageResourceKind kind,
    const ResourceRoots & roots) {
    auto resources = resources_from_resource_map(
        roots, source.find(kind == ModelPackageResourceKind::Files ? "files" : "tensors"), true);
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
    std::string spec_description;
    std::string source_format;
};

SelectedSource select_source(const std::filesystem::path & model_path, const std::filesystem::path & model_root,
    const std::string & spec_description, const engine::io::json::Value & source) {
    const auto format = require_source_format(source);
    if (format == "gguf") {
        const auto prepared = prepare_model_directory(model_path);
        auto roots = resolve_source_roots(prepared.model_root, source, prepared.standalone_gguf);
        return SelectedSource{prepared.model_root, source, std::move(roots), spec_description, format};
    }
    if (format == "safetensors") {
        auto roots = resolve_source_roots(model_root, source, std::nullopt);
        return SelectedSource{model_root, source, std::move(roots), spec_description, format};
    }
    throw std::runtime_error("unsupported model package source format: " + format);
}

SelectedSource require_selected_source(const std::filesystem::path & model_path,
    const std::filesystem::path & spec_path) {
    const auto model_root = resolve_model_root(model_path);
    const auto spec_description = model_spec_description(spec_path);
    engine::io::json::Value spec;
    try {
        spec = parse_model_spec(spec_path);
    } catch (const std::exception & error) {
        throw std::runtime_error("failed to parse " + spec_description + ": " + error.what());
    }
    const auto & sources = spec.require("sources").as_array();
    const bool explicit_gguf_path = is_gguf_file(model_path);
    const bool directory_has_gguf =
        engine::io::is_existing_directory(model_path) && engine::io::is_existing_file(model_path / "model.gguf");
    const bool use_gguf = explicit_gguf_path || directory_has_gguf;
    for (const auto & source : sources) {
        const auto format = require_source_format(source);
        if ((use_gguf && format == "gguf") || (!use_gguf && format == "safetensors")) {
            try {
                return select_source(model_path, model_root, spec_description, source);
            } catch (const std::exception & error) {
                throw std::runtime_error("failed to select " + std::string(use_gguf ? "GGUF" : "safetensors") +
                                         " source from " + spec_description + ": " + error.what());
            }
        }
    }
    throw std::runtime_error(std::string("no ") + (use_gguf ? "gguf" : "safetensors") + " model package source in " +
                             spec_description);
}

}  // namespace

ScopedModelPackageSpecOverride::ScopedModelPackageSpecOverride(const std::optional<std::filesystem::path> & path,
                                                               const std::filesystem::path & model_path)
    : previous_(active_model_spec_override), previous_model_path_(active_model_path) {
    active_model_spec_override = path;
    active_model_path = model_path.empty() ? std::nullopt : std::make_optional(model_path);
    active_embedded_spec_checked = false;
    active_embedded_spec.reset();
}

ScopedModelPackageSpecOverride::~ScopedModelPackageSpecOverride() {
    active_model_spec_override = std::move(previous_);
    active_model_path = std::move(previous_model_path_);
    active_embedded_spec_checked = false;
    active_embedded_spec.reset();
}

std::filesystem::path default_model_package_spec_path(std::string_view family) {
    if (active_model_spec_override.has_value()) {
        auto path = *active_model_spec_override;
        if (engine::io::is_existing_directory(path)) {
            path /= std::string(family) + ".json";
        }
        if (!engine::io::is_existing_file(path)) {
            throw std::runtime_error("model package spec override not found: " + path.string());
        }
        return std::filesystem::weakly_canonical(path);
    }
    if (const auto & embedded = embedded_model_spec(); embedded.has_value()) {
        if (embedded->family != family) {
            throw std::runtime_error("GGUF embeds package spec for family '" + embedded->family + "', not '" +
                                     std::string(family) + "'");
        }
        return std::filesystem::path("@gguf") / (std::string(family) + ".json");
    }
    if (builtin_model_specs().find(std::string(family)) != builtin_model_specs().end()) {
        return std::filesystem::path("@builtin") / (std::string(family) + ".json");
    }
    if (const auto external = discover_external_model_spec(family)) {
        return *external;
    }
    if (const auto hint = directory_gguf_hint(family); !hint.empty()) {
        throw std::runtime_error(hint);
    }
    throw std::runtime_error("model package spec not found for family '" + std::string(family) +
                             "' (provide --model-spec-override, embed it in the GGUF, enable "
                             "AUDIOCPP_DEPLOYMENT_BUILD, or install model_specs/" +
                             std::string(family) + ".json)");
}

ResourceBundle load_resource_bundle_from_package_spec(
    const std::filesystem::path & model_path,
    const std::filesystem::path & spec_path) {
    auto selected = require_selected_source(model_path, spec_path);
    try {
        return load_source(selected.model_root, selected.source, selected.roots);
    } catch (const std::exception & error) {
        throw std::runtime_error("failed to load model package resources using " + selected.spec_description +
                                 " source '" + selected.source_format + "': " + error.what());
    }
}

std::vector<ResourceFile> discover_resources_from_package_spec(
    const std::filesystem::path & model_path,
    const std::filesystem::path & spec_path,
    ModelPackageResourceKind kind) {
    auto selected = require_selected_source(model_path, spec_path);
    try {
        return discover_safetensors_source_resources(selected.source, kind, selected.roots);
    } catch (const std::exception & error) {
        throw std::runtime_error("failed to discover model package resources using " + selected.spec_description +
                                 " source '" + selected.source_format + "': " + error.what());
    }
}

}  // namespace engine::assets
