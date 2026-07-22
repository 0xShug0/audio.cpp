#include "config.h"

#include "../cli/args.h"
#include "../cli/request.h"

#include "engine/framework/io/json.h"

#include <stdexcept>
#include <utility>

namespace minitts::server {
namespace {

std::filesystem::path resolve_path(const std::filesystem::path & base, const std::filesystem::path & path) {
    return path.is_absolute() ? path : base / path;
}

std::unordered_map<std::string, std::string> options_from_object(const engine::io::json::Value * value) {
    return minitts::cli::json_options_map(value);
}

ServerModelConfig::VoicePreset parse_voice_preset(
    const std::filesystem::path & base,
    const engine::io::json::Value & value,
    const std::string & context) {
    if (!value.is_object()) {
        throw std::runtime_error(context + " must be an object");
    }
    ServerModelConfig::VoicePreset preset;
    if (const auto * voice_id = value.find("voice_id")) {
        preset.voice_id = voice_id->as_string();
    }
    if (const auto * voice_ref = value.find("voice_ref")) {
        preset.voice_ref = resolve_path(base, voice_ref->as_string());
    }
    if (const auto * reference_text = value.find("reference_text")) {
        preset.reference_text = reference_text->as_string();
    }
    if (!preset.voice_id.has_value() && !preset.voice_ref.has_value() && !preset.reference_text.has_value()) {
        throw std::runtime_error(context + " must set voice_id, voice_ref, or reference_text");
    }
    return preset;
}

}  // namespace

engine::core::BackendType parse_server_backend(const std::string & value) {
    auto backend = minitts::cli::parse_backend(value);
    if (backend == engine::core::BackendType::BestAvailable) {
        throw std::runtime_error("unsupported server backend: " + value);
    }
    return backend;
}

ServerConfig load_server_config(const std::filesystem::path & path) {
    const auto root = engine::io::json::parse_file(path);
    const auto base = path.parent_path();
    ServerConfig config;
    config.host = engine::io::json::optional_string(root, "host", config.host);
    config.port = engine::io::json::optional_i32(root, "port", config.port);
    config.cors_origins = engine::io::json::optional_string(root, "cors_origins", config.cors_origins);
    config.backend = parse_server_backend(engine::io::json::optional_string(root, "backend", "cuda"));
    config.device = engine::io::json::optional_i32(root, "device", config.device);
    config.threads = engine::io::json::optional_i32(root, "threads", config.threads);
    config.lazy_load = engine::io::json::optional_bool(root, "lazy_load", config.lazy_load);
    config.busy_timeout_ms = engine::io::json::optional_i32(root, "busy_timeout_ms", config.busy_timeout_ms);
    if (const auto * value = root.find("model_spec_override")) {
        config.model_spec_override = resolve_path(base, value->as_string());
    }
    if (config.port <= 0 || config.port > 65535) {
        throw std::runtime_error("server port must be in 1..65535");
    }
    if (config.busy_timeout_ms < 0) {
        throw std::runtime_error("server busy_timeout_ms must be >= 0 (0 disables the guard)");
    }
    if (config.threads <= 0) {
        throw std::runtime_error("server threads must be positive");
    }

    const auto * models = root.find("models");
    if (models == nullptr || !models->is_array() || models->as_array().empty()) {
        throw std::runtime_error("server config requires a non-empty models array");
    }
    for (const auto & item : models->as_array()) {
        ServerModelConfig model;
        model.id = engine::io::json::require_string(item, "id");
        model.path = resolve_path(base, engine::io::json::require_string(item, "path"));
        if (const auto * value = item.find("model_spec_override")) {
            model.model_spec_override = resolve_path(base, value->as_string());
        }
        model.family = engine::io::json::require_string(item, "family");
        model.task = engine::io::json::optional_string(item, "task", model.task);
        model.mode = engine::io::json::optional_string(item, "mode", model.mode);
        model.lazy = engine::io::json::optional_bool(item, "lazy", config.lazy_load);
        if (item.find("busy_timeout_ms") != nullptr) {
            const auto busy_timeout_ms = engine::io::json::optional_i32(item, "busy_timeout_ms", 0);
            if (busy_timeout_ms < 0) {
                throw std::runtime_error(
                    "busy_timeout_ms for model " + model.id + " must be >= 0 (0 disables the guard)");
            }
            model.busy_timeout_ms = busy_timeout_ms;
        }
        if (const auto * value = item.find("config")) {
            model.config_id = value->as_string();
        }
        if (const auto * value = item.find("weight")) {
            model.weight_id = value->as_string();
        }
        model.load_options = options_from_object(item.find("load_options"));
        model.session_options = options_from_object(item.find("session_options"));
        if (const auto * voice_presets = item.find("voice_presets")) {
            if (!voice_presets->is_object()) {
                throw std::runtime_error("voice_presets for model " + model.id + " must be an object");
            }
            for (const auto & [name, preset_value] : voice_presets->as_object()) {
                if (name.empty()) {
                    throw std::runtime_error("voice_presets for model " + model.id + " cannot use an empty preset name");
                }
                auto [it, inserted] = model.voice_presets.emplace(
                    name,
                    parse_voice_preset(base, preset_value, "voice preset " + name + " for model " + model.id));
                if (!inserted) {
                    throw std::runtime_error("duplicate voice preset for model " + model.id + ": " + name);
                }
                (void) it;
            }
        }
        if (const auto * default_voice_preset = item.find("default_voice_preset")) {
            if (default_voice_preset->is_string()) {
                model.default_voice_preset_id = default_voice_preset->as_string();
                if (model.default_voice_preset_id->empty()) {
                    throw std::runtime_error("default_voice_preset for model " + model.id + " cannot be empty");
                }
                if (model.voice_presets.find(*model.default_voice_preset_id) == model.voice_presets.end()) {
                    throw std::runtime_error(
                        "default_voice_preset for model " + model.id +
                        " does not match a configured voice_presets entry: " +
                        *model.default_voice_preset_id);
                }
            } else {
                model.default_voice_preset =
                    parse_voice_preset(base, *default_voice_preset, "default_voice_preset for model " + model.id);
            }
        }
        config.models.push_back(std::move(model));
    }
    return config;
}

}  // namespace minitts::server
