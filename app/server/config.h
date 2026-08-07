#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/framework/core/backend.h"

#include "http.h"

namespace minitts::server {

constexpr uint64_t kDefaultMaxRequestBodyBytes = 2ull * 1024ull * 1024ull * 1024ull;

// Per-model overrides of the server-level `live_ingest` block. Each field is
// independently optional: a model that only needs a longer deadline says so and
// inherits the rest, rather than restating the whole policy and drifting from it.
struct LiveIngestOverrides {
    std::optional<int> idle_timeout_ms;
    std::optional<int> total_timeout_ms;
    std::optional<size_t> max_body_bytes;
    std::optional<size_t> max_chunk_bytes;
    std::optional<int> send_timeout_ms;
};

// Server policy with any per-model override applied on top.
LiveIngestLimits resolve_live_ingest_limits(
    const LiveIngestLimits & base,
    const LiveIngestOverrides & overrides);

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
    // Only meaningful for a streaming model reachable over the live-ingest route;
    // ignored otherwise, since no other route delivers its body incrementally.
    LiveIngestOverrides live_ingest;
    std::optional<std::string> config_id;
    std::optional<std::string> weight_id;
    std::unordered_map<std::string, std::string> load_options;
    std::unordered_map<std::string, std::string> session_options;
    std::unordered_map<std::string, std::string> default_request_options;
    std::unordered_map<std::string, VoicePreset> voice_presets;
    std::optional<VoicePreset> default_voice_preset;
    std::optional<std::string> default_voice_preset_id;
};

struct ServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string cors_origins = "";
    bool ui_enabled = true;
    bool ui_management = false;
    engine::core::BackendType backend = engine::core::BackendType::Cuda;
    int device = 0;
    int threads = 1;
    bool lazy_load = false;
    bool log_request_body = false;
    uint64_t max_request_body_bytes = kDefaultMaxRequestBodyBytes;
    // A single model runs one request at a time (serialized on model.mutex). If a
    // running inference wedges the GPU -- a CUDA call that never returns cannot be
    // cancelled from userspace -- later requests would otherwise block forever, so
    // once the current run has held the lock this long a new request fails fast with
    // 503 instead of parking a worker thread. Must exceed the slowest legitimate
    // single inference (music generation can take minutes). 0 disables the guard and
    // restores unbounded waiting.
    int busy_timeout_ms = 300000;
    // Fleet-wide bounds for incrementally delivered request bodies. The defaults are
    // in LiveIngestLimits; a model entry may override any subset of them.
    LiveIngestLimits live_ingest;
    std::optional<std::filesystem::path> model_spec_override;
    // Voice library shared across all TTS models: *.wav files plus a `prompt_text`
    // mapping file (<basename>|<transcript>). A request `voice` name that is not a
    // model preset resolves to <voice_dir>/<name>.wav as the cloning reference.
    std::optional<std::filesystem::path> voice_dir;
    std::vector<ServerModelConfig> models;
};

engine::core::BackendType parse_server_backend(const std::string & value);
ServerConfig load_server_config(const std::filesystem::path & path);

}  // namespace minitts::server
