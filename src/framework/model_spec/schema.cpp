#include "engine/framework/model_spec/schema.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>

namespace engine::model_spec {
namespace {

namespace json = engine::io::json;

[[noreturn]] void fail(std::string_view path, const std::string & message) {
    throw std::runtime_error(std::string(path) + ": " + message);
}

const json::Value::Object & require_spec_object(const json::Value & value, std::string_view path) {
    if (!value.is_object()) {
        fail(path, "expected object");
    }
    return value.as_object();
}

const json::Value::Array & require_spec_array(const json::Value & value, std::string_view path) {
    if (!value.is_array()) {
        fail(path, "expected array");
    }
    return value.as_array();
}

const json::Value & require_spec_field(const json::Value & object, std::string_view key, std::string_view path) {
    const std::string key_string(key);
    const auto * value = object.find(key_string);
    if (value == nullptr) {
        fail(path, "missing required field '" + key_string + "'");
    }
    return *value;
}

std::string require_spec_string(const json::Value & value, std::string_view path) {
    if (!value.is_string() || value.as_string().empty()) {
        fail(path, "expected non-empty string");
    }
    return value.as_string();
}

bool require_spec_bool(const json::Value & value, std::string_view path) {
    if (!value.is_bool()) {
        fail(path, "expected bool");
    }
    return value.as_bool();
}

void require_spec_number(const json::Value & value, std::string_view path) {
    if (!value.is_number()) {
        fail(path, "expected number");
    }
}

const std::unordered_set<std::string> & tasks() {
    static const std::unordered_set<std::string> values = {
        "vad", "asr", "diar", "sep", "music", "sfx", "edit", "tts", "clone", "vc",
        "s2s", "align", "design", "speaker", "svc", "codec", "dialogue",
    };
    return values;
}

const std::unordered_set<std::string> & modes() {
    static const std::unordered_set<std::string> values = {"offline", "streaming"};
    return values;
}

const std::unordered_set<std::string> & categories() {
    static const std::unordered_set<std::string> values = {
        "asr", "tts", "voice_conversion", "audio_generation", "audio_tools", "speech_analysis", "community",
    };
    return values;
}

const std::unordered_set<std::string> & statuses() {
    static const std::unordered_set<std::string> values = {"supported", "community", "experimental", "wip", "unsupported"};
    return values;
}

const std::unordered_set<std::string> & source_formats() {
    static const std::unordered_set<std::string> values = {"safetensors", "gguf", "onnx", "ggml", "pytorch"};
    return values;
}

const std::unordered_set<std::string> & precisions() {
    static const std::unordered_set<std::string> values = {"native", "orig", "f32", "f16", "bf16", "q8_0"};
    return values;
}

const std::unordered_set<std::string> & source_kinds() {
    static const std::unordered_set<std::string> values = {
        "huggingface_snapshot", "local_snapshot", "composite_snapshot", "converter", "unsupported",
    };
    return values;
}

const std::unordered_set<std::string> & companion_kinds() {
    static const std::unordered_set<std::string> values = {"model", "file", "directory", "external"};
    return values;
}

const std::unordered_set<std::string> & option_types() {
    static const std::unordered_set<std::string> values = {"bool", "int", "float", "string", "enum", "path", "audio_path"};
    return values;
}

const std::unordered_set<std::string> & runtime_tags() {
    static const std::unordered_set<std::string> values = {"gguf", "stream", "server", "cuda", "vulkan", "metal", "cpu"};
    return values;
}

const std::unordered_set<std::string> & ui_tags() {
    static const std::unordered_set<std::string> values = {
        "ASR", "TTS", "Clone", "VC", "Align", "VAD", "Diar", "Codec", "Sep", "Music", "SFX",
        "Edit", "Design", "Dialogue", "GGUF", "Stream",
    };
    return values;
}

const std::unordered_set<std::string> & common_options() {
    static const std::unordered_set<std::string> values = {
        "seed", "language", "voice_ref", "reference_text", "prompt_audio", "source_audio", "target_voice",
        "text_chunk_mode", "text_chunk_size", "max_new_tokens", "temperature", "top_p", "top_k",
        "repetition_penalty", "do_sample", "return_timestamps", "duration_scale", "min_seconds",
        "max_seconds", "speaking_rate", "pitch_shift", "energy_scale", "num_inference_steps",
    };
    return values;
}

void validate_enum(const std::string & value,
                   const std::unordered_set<std::string> & allowed,
                   std::string_view path,
                   std::string_view label) {
    if (allowed.find(value) == allowed.end()) {
        fail(path, "unknown " + std::string(label) + " '" + value + "'");
    }
}

void validate_string_array(const json::Value & value,
                           const std::unordered_set<std::string> * allowed,
                           std::string_view path,
                           std::string_view label) {
    const auto & array = require_spec_array(value, path);
    for (size_t index = 0; index < array.size(); ++index) {
        const auto item_path = std::string(path) + "[" + std::to_string(index) + "]";
        const auto value_string = require_spec_string(array[index], item_path);
        if (allowed != nullptr) {
            validate_enum(value_string, *allowed, item_path, label);
        }
    }
}

void validate_option_name(const std::string & name, const std::string & family, std::string_view path) {
    if (common_options().find(name) != common_options().end()) {
        return;
    }
    const std::string prefix = family + ".";
    if (name.rfind(prefix, 0) == 0 && name.size() > prefix.size()) {
        return;
    }
    fail(path, "option '" + name + "' must be canonical or namespaced as " + family + ".<name>");
}

void validate_option(const json::Value & value, const std::string & family, std::string_view path) {
    require_spec_object(value, path);
    const auto name = require_spec_string(require_spec_field(value, "name", path), std::string(path) + ".name");
    validate_option_name(name, family, std::string(path) + ".name");
    const auto type = require_spec_string(require_spec_field(value, "type", path), std::string(path) + ".type");
    validate_enum(type, option_types(), std::string(path) + ".type", "option type");
    const auto * values = value.find("values");
    if (type == "enum") {
        if (values == nullptr) {
            fail(std::string(path) + ".values", "enum option requires values");
        }
        validate_string_array(*values, nullptr, std::string(path) + ".values", "enum value");
        if (values->as_array().empty()) {
            fail(std::string(path) + ".values", "enum option values must not be empty");
        }
    } else if (values != nullptr) {
        fail(std::string(path) + ".values", "values are allowed only for enum options");
    }
    const auto * default_value = value.find("default");
    if (default_value != nullptr && default_value->is_null()) {
        fail(std::string(path) + ".default", "use absent default instead of null");
    }
    (void) require_spec_string(require_spec_field(value, "description", path), std::string(path) + ".description");
}

void validate_options(const json::Value & value, const std::string & family, std::string_view path) {
    require_spec_object(value, path);
    for (const std::string scope : {"request", "session", "load"}) {
        const auto child_path = std::string(path) + "." + scope;
        const auto & rows = require_spec_array(require_spec_field(value, scope, path), child_path);
        for (size_t index = 0; index < rows.size(); ++index) {
            validate_option(rows[index], family, child_path + "[" + std::to_string(index) + "]");
        }
    }
}

void validate_capabilities(const json::Value & value, std::string_view path) {
    require_spec_object(value, path);
    for (const std::string key : {"timestamps", "speaker_reference", "style_condition", "voice_design"}) {
        (void) require_spec_bool(require_spec_field(value, key, path), std::string(path) + "." + key);
    }
    validate_string_array(require_spec_field(value, "languages", path), nullptr, std::string(path) + ".languages", "language");
}

void validate_runtime(const json::Value & value, std::string_view path) {
    require_spec_object(value, path);
    validate_string_array(require_spec_field(value, "tags", path), &runtime_tags(), std::string(path) + ".tags", "runtime tag");
    if (const auto * default_format = value.find("default_format")) {
        validate_enum(require_spec_string(*default_format, std::string(path) + ".default_format"),
                      source_formats(),
                      std::string(path) + ".default_format",
                      "format");
    }
}

void validate_hf_snapshot_source(const json::Value & value, std::string_view path) {
    require_spec_object(value, path);
    (void) require_spec_string(require_spec_field(value, "repo", path), std::string(path) + ".repo");
    if (const auto * revision = value.find("revision")) {
        (void) require_spec_string(*revision, std::string(path) + ".revision");
    }
    for (const std::string key : {"include", "exclude"}) {
        if (const auto * array = value.find(key)) {
            validate_string_array(*array, nullptr, std::string(path) + "." + key, key);
        }
    }
    if (const auto * strip_prefix = value.find("strip_prefix")) {
        (void) require_spec_string(*strip_prefix, std::string(path) + ".strip_prefix");
    }
    if (const auto * gated = value.find("gated")) {
        (void) require_spec_bool(*gated, std::string(path) + ".gated");
    }
}

void validate_source(const json::Value & value, std::string_view path) {
    require_spec_object(value, path);
    const auto kind = require_spec_string(require_spec_field(value, "kind", path), std::string(path) + ".kind");
    validate_enum(kind, source_kinds(), std::string(path) + ".kind", "source kind");
    if (kind == "huggingface_snapshot") {
        validate_hf_snapshot_source(value, path);
    } else if (kind == "local_snapshot") {
        (void) require_spec_string(require_spec_field(value, "path", path), std::string(path) + ".path");
        if (const auto * array = value.find("include")) {
            validate_string_array(*array, nullptr, std::string(path) + ".include", "include");
        }
    } else if (kind == "composite_snapshot") {
        const auto placements_path = std::string(path) + ".placements";
        const auto & placements_field = require_spec_field(value, "placements", path);
        const auto & placements = require_spec_array(placements_field, placements_path);
        for (size_t index = 0; index < placements.size(); ++index) {
            const auto placement_path = std::string(path) + ".placements[" + std::to_string(index) + "]";
            const auto & placement = placements[index];
            require_spec_object(placement, placement_path);
            if (const auto * target = placement.find("target_subdir")) {
                (void) require_spec_string(*target, placement_path + ".target_subdir");
            }
            validate_hf_snapshot_source(require_spec_field(placement, "source", placement_path), placement_path + ".source");
            if (const auto * required = placement.find("required_files")) {
                validate_string_array(*required, nullptr, placement_path + ".required_files", "required file");
            }
        }
    } else if (kind == "converter") {
        (void) require_spec_string(require_spec_field(value, "converter", path), std::string(path) + ".converter");
        (void) require_spec_string(require_spec_field(value, "description", path), std::string(path) + ".description");
    } else if (kind == "unsupported") {
        (void) require_spec_string(require_spec_field(value, "reason", path), std::string(path) + ".reason");
    }
}

void validate_ref(const json::Value & value, const json::Value::Object & roots, std::string_view path) {
    std::string root;
    if (value.is_string()) {
        const auto ref = value.as_string();
        const auto split = ref.find(':');
        if (split == std::string::npos || split == 0) {
            fail(path, "invalid resource reference '" + ref + "'");
        }
        root = ref.substr(0, split);
    } else {
        require_spec_object(value, path);
        const auto source = require_spec_string(require_spec_field(value, "source", path), std::string(path) + ".source");
        const auto split = source.find(':');
        if (split == std::string::npos || split == 0) {
            fail(std::string(path) + ".source", "invalid resource reference '" + source + "'");
        }
        root = source.substr(0, split);
        if (const auto * prefix = value.find("prefix")) {
            (void) require_spec_string(*prefix, std::string(path) + ".prefix");
        }
    }
    if (roots.find(root) == roots.end()) {
        fail(path, "unknown root '" + root + "'");
    }
}

void validate_layout(const json::Value & value, std::string_view path) {
    require_spec_object(value, path);
    const auto format = require_spec_string(require_spec_field(value, "format", path), std::string(path) + ".format");
    validate_enum(format, source_formats(), std::string(path) + ".format", "format");
    const auto roots_path = std::string(path) + ".roots";
    const auto & roots_field = require_spec_field(value, "roots", path);
    const auto & roots = require_spec_object(roots_field, roots_path);
    for (const auto & [root_id, root_value] : roots) {
        if (root_id.empty()) {
            fail(std::string(path) + ".roots", "root id must not be empty");
        }
        (void) require_spec_string(root_value, std::string(path) + ".roots." + root_id);
    }
    for (const std::string map_name : {"files", "optional_files", "tensors"}) {
        const auto * map_value = value.find(map_name);
        if (map_value == nullptr) {
            continue;
        }
        const auto & map = require_spec_object(*map_value, std::string(path) + "." + map_name);
        for (const auto & [id, ref] : map) {
            if (id.empty()) {
                fail(std::string(path) + "." + map_name, "resource id must not be empty");
            }
            validate_ref(ref, roots, std::string(path) + "." + map_name + "." + id);
        }
    }
}

void validate_package(const json::Value & value, const json::Value::Object & layouts, std::string_view path) {
    require_spec_object(value, path);
    (void) require_spec_string(require_spec_field(value, "id", path), std::string(path) + ".id");
    (void) require_spec_string(require_spec_field(value, "display_name", path), std::string(path) + ".display_name");
    const auto format = require_spec_string(require_spec_field(value, "format", path), std::string(path) + ".format");
    validate_enum(format, source_formats(), std::string(path) + ".format", "format");
    const auto precision = require_spec_string(require_spec_field(value, "precision", path), std::string(path) + ".precision");
    validate_enum(precision, precisions(), std::string(path) + ".precision", "precision");
    (void) require_spec_string(require_spec_field(value, "target_directory", path), std::string(path) + ".target_directory");
    validate_string_array(require_spec_field(value, "required_files", path), nullptr, std::string(path) + ".required_files", "required file");
    const auto layout_id = require_spec_string(require_spec_field(value, "layout", path), std::string(path) + ".layout");
    const auto layout_it = layouts.find(layout_id);
    if (layout_it == layouts.end()) {
        fail(std::string(path) + ".layout", "unknown layout '" + layout_id + "'");
    }
    const auto layout_format =
        require_spec_string(require_spec_field(layout_it->second, "format", std::string("layouts.") + layout_id),
                       std::string("layouts.") + layout_id + ".format");
    if (layout_format != format) {
        fail(std::string(path) + ".layout",
             "package format '" + format + "' does not match layout format '" + layout_format + "'");
    }
    validate_source(require_spec_field(value, "source", path), std::string(path) + ".source");
    if (const auto * description = value.find("description")) {
        (void) require_spec_string(*description, std::string(path) + ".description");
    }
}

void validate_companions(const json::Value & value, const std::string & family, std::string_view path) {
    const auto & companions = require_spec_array(value, path);
    for (size_t index = 0; index < companions.size(); ++index) {
        const auto item_path = std::string(path) + "[" + std::to_string(index) + "]";
        const auto & companion = companions[index];
        require_spec_object(companion, item_path);
        (void) require_spec_string(require_spec_field(companion, "id", item_path), item_path + ".id");
        const auto kind = require_spec_string(require_spec_field(companion, "kind", item_path), item_path + ".kind");
        validate_enum(kind, companion_kinds(), item_path + ".kind", "companion kind");
        if (const auto * companion_family = companion.find("family")) {
            (void) require_spec_string(*companion_family, item_path + ".family");
        }
        if (const auto * option = companion.find("option")) {
            const auto option_name = require_spec_string(*option, item_path + ".option");
            validate_option_name(option_name, family, item_path + ".option");
        }
        (void) require_spec_bool(require_spec_field(companion, "required", item_path), item_path + ".required");
        validate_string_array(require_spec_field(companion, "required_for", item_path), nullptr, item_path + ".required_for", "requirement");
    }
}

void validate_ui(const json::Value & value, const std::unordered_set<std::string> & package_ids, std::string_view path) {
    require_spec_object(value, path);
    const auto recommended = require_spec_string(require_spec_field(value, "recommended_package", path),
                                            std::string(path) + ".recommended_package");
    if (package_ids.find(recommended) == package_ids.end()) {
        fail(std::string(path) + ".recommended_package", "unknown package '" + recommended + "'");
    }
    if (const auto * min_vram = value.find("min_vram_gb")) {
        require_spec_number(*min_vram, std::string(path) + ".min_vram_gb");
    }
    validate_string_array(require_spec_field(value, "tags", path), &ui_tags(), std::string(path) + ".tags", "UI tag");
    validate_string_array(require_spec_field(value, "docs", path), nullptr, std::string(path) + ".docs", "doc path");
    if (const auto * summary = value.find("summary")) {
        (void) require_spec_string(*summary, std::string(path) + ".summary");
    }
}

void validate_legacy_source(const json::Value & value, std::string_view path) {
    validate_layout(value, path);
}

void validate_v1(const json::Value & spec, std::string_view source_name) {
    const auto family = require_spec_string(require_spec_field(spec, "family", source_name), std::string(source_name) + ".family");
    (void) require_spec_string(require_spec_field(spec, "display_name", source_name), std::string(source_name) + ".display_name");
    validate_enum(require_spec_string(require_spec_field(spec, "category", source_name), std::string(source_name) + ".category"),
                  categories(), std::string(source_name) + ".category", "category");
    validate_enum(require_spec_string(require_spec_field(spec, "status", source_name), std::string(source_name) + ".status"),
                  statuses(), std::string(source_name) + ".status", "status");
    validate_string_array(require_spec_field(spec, "tasks", source_name), &tasks(), std::string(source_name) + ".tasks", "task");
    if (spec.require("tasks").as_array().empty()) {
        fail(std::string(source_name) + ".tasks", "tasks must not be empty");
    }
    validate_string_array(require_spec_field(spec, "modes", source_name), &modes(), std::string(source_name) + ".modes", "mode");
    if (spec.require("modes").as_array().empty()) {
        fail(std::string(source_name) + ".modes", "modes must not be empty");
    }
    validate_runtime(require_spec_field(spec, "runtime", source_name), std::string(source_name) + ".runtime");
    validate_capabilities(require_spec_field(spec, "capabilities", source_name), std::string(source_name) + ".capabilities");
    validate_options(require_spec_field(spec, "options", source_name), family, std::string(source_name) + ".options");

    const auto layouts_path = std::string(source_name) + ".layouts";
    const auto & layouts_field = require_spec_field(spec, "layouts", source_name);
    const auto & layouts = require_spec_object(layouts_field, layouts_path);
    if (layouts.empty()) {
        fail(std::string(source_name) + ".layouts", "layouts must not be empty");
    }
    for (const auto & [layout_id, layout] : layouts) {
        if (layout_id.empty()) {
            fail(std::string(source_name) + ".layouts", "layout id must not be empty");
        }
        validate_layout(layout, std::string(source_name) + ".layouts." + layout_id);
    }

    const auto packages_path = std::string(source_name) + ".packages";
    const auto & packages_field = require_spec_field(spec, "packages", source_name);
    const auto & packages = require_spec_array(packages_field, packages_path);
    if (packages.empty()) {
        fail(std::string(source_name) + ".packages", "packages must not be empty");
    }
    std::unordered_set<std::string> package_ids;
    for (size_t index = 0; index < packages.size(); ++index) {
        const auto path = std::string(source_name) + ".packages[" + std::to_string(index) + "]";
        const auto id = require_spec_string(require_spec_field(packages[index], "id", path), path + ".id");
        if (!package_ids.insert(id).second) {
            fail(path + ".id", "duplicate package id '" + id + "'");
        }
        validate_package(packages[index], layouts, path);
    }
    validate_companions(require_spec_field(spec, "companions", source_name), family, std::string(source_name) + ".companions");
    validate_ui(require_spec_field(spec, "ui", source_name), package_ids, std::string(source_name) + ".ui");

    const auto sources_path = std::string(source_name) + ".sources";
    const auto & sources_field = require_spec_field(spec, "sources", source_name);
    const auto & sources = require_spec_array(sources_field, sources_path);
    for (size_t index = 0; index < sources.size(); ++index) {
        validate_legacy_source(sources[index], std::string(source_name) + ".sources[" + std::to_string(index) + "]");
    }
}

void validate_legacy(const json::Value & spec, std::string_view source_name) {
    (void) require_spec_string(require_spec_field(spec, "family", source_name), std::string(source_name) + ".family");
    const auto sources_path = std::string(source_name) + ".sources";
    const auto & sources_field = require_spec_field(spec, "sources", source_name);
    const auto & sources = require_spec_array(sources_field, sources_path);
    for (size_t index = 0; index < sources.size(); ++index) {
        validate_legacy_source(sources[index], std::string(source_name) + ".sources[" + std::to_string(index) + "]");
    }
}

}  // namespace

void validate_spec(const json::Value & spec, std::string_view source_name) {
    require_spec_object(spec, source_name);
    const auto * version = spec.find("schema_version");
    if (version == nullptr) {
        validate_legacy(spec, source_name);
        return;
    }
    if (!version->is_number() || version->as_i64() != kModelSpecSchemaVersion) {
        fail(std::string(source_name) + ".schema_version",
             "expected " + std::to_string(kModelSpecSchemaVersion));
    }
    validate_v1(spec, source_name);
}

}  // namespace engine::model_spec
