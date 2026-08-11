#include "config.h"

#include "../cli/args.h"
#include "../cli/request.h"

#include "engine/framework/io/json.h"

#include <cmath>
#include <cstdint>
#include <limits>
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

uint64_t parse_max_request_body_bytes(const engine::io::json::Value & value) {
    if (!value.is_number()) {
        throw std::runtime_error("server max_request_body_bytes must be a number");
    }
    const double parsed = value.as_number();
    constexpr double kMaxSafeJsonInteger = 9007199254740991.0;  // 2^53 - 1
    if (parsed < 0.0) {
        throw std::runtime_error("server max_request_body_bytes must be non-negative");
    }
    if (std::floor(parsed) != parsed) {
        throw std::runtime_error("server max_request_body_bytes must be an integer");
    }
    if (parsed > kMaxSafeJsonInteger) {
        throw std::runtime_error("server max_request_body_bytes must be <= 2^53 - 1");
    }
    return static_cast<uint64_t>(parsed);
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

// Every live-ingest bound uses 0 to mean "disabled", matching busy_timeout_ms, so
// only a negative value is malformed. Rejected at parse time rather than clamped:
// a negative deadline is a typo, and silently treating it as "no bound" would
// remove a guard the operator believed they had set.
//
// Read and validated by hand rather than through optional_i32/optional_i64, both
// of which are wrong here in two ways. They return the supplied fallback for a
// present-but-wrong-typed field, so `"max_body_bytes": "oops"` in a MODEL override
// would silently record the compiled default as a deliberate override and widen a
// stricter server policy. And optional_i32 narrows to int before anything checks
// the range, so on a 32-bit int a value of 4294967296 becomes 0 — which here means
// "disabled", quietly removing the bound the operator was trying to set.
double live_ingest_number(
    const engine::io::json::Value & value,
    const char * key,
    const std::string & context) {
    const auto * field = value.find(key);
    if (field == nullptr || !field->is_number()) {
        throw std::runtime_error(context + " " + key + " must be a number");
    }
    const double parsed = field->as_number();
    constexpr double kMaxSafeJsonInteger = 9007199254740991.0;  // 2^53 - 1
    if (std::floor(parsed) != parsed) {
        throw std::runtime_error(context + " " + key + " must be an integer");
    }
    if (parsed < 0.0) {
        throw std::runtime_error(context + " " + key + " must be >= 0 (0 disables the bound)");
    }
    if (parsed > kMaxSafeJsonInteger) {
        throw std::runtime_error(context + " " + key + " must be <= 2^53 - 1");
    }
    return parsed;
}

int live_ingest_ms(
    const engine::io::json::Value & value,
    const char * key,
    int fallback,
    const std::string & context) {
    if (value.find(key) == nullptr) {
        return fallback;
    }
    const double parsed = live_ingest_number(value, key, context);
    if (parsed > static_cast<double>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            context + " " + key + " must be <= " + std::to_string(std::numeric_limits<int>::max()) + " ms");
    }
    return static_cast<int>(parsed);
}

size_t live_ingest_bytes(
    const engine::io::json::Value & value,
    const char * key,
    size_t fallback,
    const std::string & context) {
    if (value.find(key) == nullptr) {
        return fallback;
    }
    const double parsed = live_ingest_number(value, key, context);
    // Range-checked against size_t rather than cast blindly: on a 32-bit target a
    // legal-looking 4294967296 would otherwise truncate to 0 and disable the bound.
    if (parsed > static_cast<double>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error(context + " " + key + " is too large for this platform");
    }
    return static_cast<size_t>(parsed);
}

LiveIngestLimits parse_live_ingest_limits(
    const engine::io::json::Value & value,
    const LiveIngestLimits & fallback,
    const std::string & context) {
    if (!value.is_object()) {
        throw std::runtime_error(context + " must be an object");
    }
    LiveIngestLimits limits;
    limits.idle_timeout_ms = live_ingest_ms(value, "idle_timeout_ms", fallback.idle_timeout_ms, context);
    limits.total_timeout_ms = live_ingest_ms(value, "total_timeout_ms", fallback.total_timeout_ms, context);
    limits.max_body_bytes = live_ingest_bytes(value, "max_body_bytes", fallback.max_body_bytes, context);
    limits.max_chunk_bytes = live_ingest_bytes(value, "max_chunk_bytes", fallback.max_chunk_bytes, context);
    // The one bound that cannot be disabled. A chunk is materialized in memory
    // before it is served, so "unbounded" is not implementable — and a 0 here would
    // underflow the overflow guard in the chunk-size parser, re-admitting a declared
    // size of SIZE_MAX. Rejected rather than quietly substituted.
    if (limits.max_chunk_bytes == 0) {
        throw std::runtime_error(
            context + " max_chunk_bytes must be > 0: a chunk is held in memory, so it cannot be unbounded");
    }
    limits.send_timeout_ms = live_ingest_ms(value, "send_timeout_ms", fallback.send_timeout_ms, context);
    return limits;
}

LiveIngestOverrides parse_live_ingest_overrides(
    const engine::io::json::Value & value,
    const std::string & context) {
    if (!value.is_object()) {
        throw std::runtime_error(context + " must be an object");
    }
    // Parsed against the compiled-in defaults purely to reuse the validation; only
    // the keys actually present are recorded, so the rest still fall through to
    // whatever server policy is at the time the override is applied.
    const LiveIngestLimits defaults;
    const auto parsed = parse_live_ingest_limits(value, defaults, context);
    LiveIngestOverrides overrides;
    if (value.find("idle_timeout_ms") != nullptr) {
        overrides.idle_timeout_ms = parsed.idle_timeout_ms;
    }
    if (value.find("total_timeout_ms") != nullptr) {
        overrides.total_timeout_ms = parsed.total_timeout_ms;
    }
    if (value.find("max_body_bytes") != nullptr) {
        overrides.max_body_bytes = parsed.max_body_bytes;
    }
    if (value.find("max_chunk_bytes") != nullptr) {
        overrides.max_chunk_bytes = parsed.max_chunk_bytes;
    }
    if (value.find("send_timeout_ms") != nullptr) {
        overrides.send_timeout_ms = parsed.send_timeout_ms;
    }
    return overrides;
}

}  // namespace

LiveIngestLimits resolve_live_ingest_limits(
    const LiveIngestLimits & base,
    const LiveIngestOverrides & overrides) {
    LiveIngestLimits limits = base;
    if (overrides.idle_timeout_ms.has_value()) {
        limits.idle_timeout_ms = *overrides.idle_timeout_ms;
    }
    if (overrides.total_timeout_ms.has_value()) {
        limits.total_timeout_ms = *overrides.total_timeout_ms;
    }
    if (overrides.max_body_bytes.has_value()) {
        limits.max_body_bytes = *overrides.max_body_bytes;
    }
    if (overrides.max_chunk_bytes.has_value()) {
        limits.max_chunk_bytes = *overrides.max_chunk_bytes;
    }
    if (overrides.send_timeout_ms.has_value()) {
        limits.send_timeout_ms = *overrides.send_timeout_ms;
    }
    return limits;
}

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
    config.ui_enabled = engine::io::json::optional_bool(root, "ui", config.ui_enabled);
    config.ui_management = engine::io::json::optional_bool(root, "ui_management", config.ui_management);
    config.backend = parse_server_backend(engine::io::json::optional_string(root, "backend", "cuda"));
    config.device = engine::io::json::optional_i32(root, "device", config.device);
    config.threads = engine::io::json::optional_i32(root, "threads", config.threads);
    config.lazy_load = engine::io::json::optional_bool(root, "lazy_load", config.lazy_load);
    config.log_request_body = engine::io::json::optional_bool(root, "log_request_body", config.log_request_body);
    if (const auto * value = root.find("max_request_body_bytes")) {
        config.max_request_body_bytes = parse_max_request_body_bytes(*value);
    }
    config.busy_timeout_ms = engine::io::json::optional_i32(root, "busy_timeout_ms", config.busy_timeout_ms);
    if (const auto * value = root.find("live_ingest")) {
        config.live_ingest = parse_live_ingest_limits(*value, config.live_ingest, "server live_ingest");
    }
    if (const auto * value = root.find("model_spec_override")) {
        config.model_spec_override = resolve_path(base, value->as_string());
    }
    if (const auto * value = root.find("voice_dir")) {
        if (!value->is_string()) {
            throw std::runtime_error("server voice_dir must be a string");
        }
        config.voice_dir = resolve_path(base, value->as_string());
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
    if (models == nullptr || !models->is_array()) {
        throw std::runtime_error("server config requires a models array");
    }
    if (models->as_array().empty() && !config.ui_management) {
        throw std::runtime_error("server config requires a non-empty models array unless ui_management is enabled");
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
        if (const auto * value = item.find("live_ingest")) {
            model.live_ingest = parse_live_ingest_overrides(*value, "live_ingest for model " + model.id);
        }
        if (const auto * value = item.find("config")) {
            model.config_id = value->as_string();
        }
        if (const auto * value = item.find("weight")) {
            model.weight_id = value->as_string();
        }
        model.load_options = options_from_object(item.find("load_options"));
        model.session_options = options_from_object(item.find("session_options"));
        model.default_request_options = options_from_object(item.find("default_request_options"));
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
