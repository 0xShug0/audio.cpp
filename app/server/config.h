#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/framework/core/backend.h"

namespace minitts::server {

struct ServerModelConfig {
    struct VoicePreset {
        std::optional<std::string> voice_id;
        std::optional<std::filesystem::path> voice_ref;
        std::optional<std::string> reference_text;
    };

    std::string id;
    std::filesystem::path path;
    std::optional<std::filesystem::path> model_spec_override;
    std::string family;
    std::string task = "tts";
    std::string mode = "offline";
    bool lazy = false;
    // Overrides ServerConfig::busy_timeout_ms for this model, and acts as the ceiling
    // a per-request busy_timeout_ms is clamped to. Model runtimes differ by orders of
    // magnitude (a short TTS clip vs. minutes of music generation), so one fleet-wide
    // bound is either too tight for the slow models or useless for the fast ones.
    std::optional<int> busy_timeout_ms;
    std::optional<std::string> config_id;
    std::optional<std::string> weight_id;
    std::unordered_map<std::string, std::string> load_options;
    std::unordered_map<std::string, std::string> session_options;
    std::unordered_map<std::string, VoicePreset> voice_presets;
    std::optional<VoicePreset> default_voice_preset;
    std::optional<std::string> default_voice_preset_id;
};

struct ServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string cors_origins = "";
    engine::core::BackendType backend = engine::core::BackendType::Cuda;
    int device = 0;
    int threads = 1;
    bool lazy_load = false;
    // A single model runs one request at a time (serialized on model.mutex). If a
    // running inference wedges the GPU -- a CUDA call that never returns cannot be
    // cancelled from userspace -- later requests would otherwise block forever, so
    // once the current run has held the lock this long a new request fails fast with
    // 503 instead of parking a worker thread. Must exceed the slowest legitimate
    // single inference (music generation can take minutes). 0 disables the guard and
    // restores unbounded waiting.
    int busy_timeout_ms = 300000;
    std::optional<std::filesystem::path> model_spec_override;
    std::vector<ServerModelConfig> models;
};

engine::core::BackendType parse_server_backend(const std::string & value);
ServerConfig load_server_config(const std::filesystem::path & path);

}  // namespace minitts::server
