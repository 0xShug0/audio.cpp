#pragma once

#include "busy_guard.h"
#include "config.h"
#include "http.h"

#include "engine/framework/io/json.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace minitts::server {

class ServerState final : public IHttpHandler {
public:
    ServerState(ServerConfig config, std::filesystem::path request_base);

    HttpResponse handle(const HttpRequest & request) override;

private:
    struct LoadedModel {
        struct RuntimeVoicePreset {
            std::optional<std::string> voice_id;
            std::optional<engine::runtime::AudioBuffer> audio;
            std::optional<std::string> reference_text;
        };

        ServerModelConfig config;
        engine::runtime::TaskSpec task;
        std::unique_ptr<engine::runtime::ILoadedVoiceModel> model;
        std::unique_ptr<engine::runtime::IVoiceTaskSession> session;
        engine::runtime::IOfflineVoiceTaskSession * offline = nullptr;
        engine::runtime::IStreamingVoiceTaskSession * streaming = nullptr;
        std::unordered_map<std::string, RuntimeVoicePreset> voice_presets;
        std::optional<RuntimeVoicePreset> default_voice_preset;
        // Serializes runs on this model and bounds how long a caller waits for its
        // turn; see BusyGuard.
        BusyGuard busy;
    };

    // Acquire the model's run guard. `request_timeout_ms` is the caller-supplied
    // override, clamped by this model's configured ceiling. Throws ServerBusyError
    // (-> HTTP 503) once the effective timeout has elapsed.
    BusyGuard::Lock acquire_model_run(LoadedModel & model, std::optional<int> request_timeout_ms);

    // Server policy for this model: its own busy_timeout_ms if set, else the
    // top-level config value.
    int model_busy_timeout_ceiling(const LoadedModel & model) const;

    void load_models();
    LoadedModel::RuntimeVoicePreset load_runtime_voice_preset(const ServerModelConfig::VoicePreset & preset) const;
    void load_voice_presets(LoadedModel & model) const;
    void ensure_model_loaded_locked(LoadedModel & model);
    LoadedModel & require_model(const engine::io::json::Value & body);
    const LoadedModel::RuntimeVoicePreset * select_voice_preset(
        const LoadedModel & model,
        const engine::io::json::Value & body,
        bool & voice_field_is_preset) const;
    engine::runtime::TaskRequest build_speech_request(
        const LoadedModel & model,
        const engine::io::json::Value & body) const;
    struct TimedTaskResult;
    // `busy_timeout_ms` on each of these is the per-request override parsed from the
    // request body; nullopt means "use the model's configured ceiling".
    TimedTaskResult run_model(
        LoadedModel & model,
        const engine::runtime::TaskRequest & request,
        std::optional<int> busy_timeout_ms = std::nullopt);
    TimedTaskResult run_streaming_model(
        LoadedModel & model,
        const engine::runtime::TaskRequest & request,
        const std::function<void(const engine::runtime::StreamEvent &)> & event_sink = {},
        std::optional<int> busy_timeout_ms = std::nullopt);
    HttpResponse handle_speech(const std::string & body_text);
    HttpResponse handle_speech_stream(
        LoadedModel & model,
        const engine::runtime::TaskRequest & request,
        const engine::io::json::Value & body);
    HttpResponse handle_transcription(const HttpRequest & request);
    HttpResponse handle_transcription_json(const std::string & body_text);
    HttpResponse handle_transcription_multipart(const std::string & body_text, const std::string & boundary);
    HttpResponse run_transcription(
        LoadedModel & model,
        const engine::runtime::TaskRequest & request,
        std::optional<int> busy_timeout_ms = std::nullopt);
    HttpResponse run_transcription_stream(
        LoadedModel & model,
        const engine::runtime::TaskRequest & request,
        std::optional<int> busy_timeout_ms = std::nullopt);
    HttpResponse handle_generic_run(const std::string & body_text);
    HttpResponse handle_generic_stream(const std::string & body_text);
    HttpResponse handle_voices(const HttpRequest & request) const;
    std::string models_json() const;
    std::string get_allowed_origin(const HttpRequest & request) const;

    ServerConfig config_;
    std::filesystem::path request_base_;
    std::vector<std::unique_ptr<LoadedModel>> models_;
    std::unordered_map<std::string, size_t> model_index_;
};

}  // namespace minitts::server
