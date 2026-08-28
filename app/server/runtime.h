#pragma once

#include "busy_guard.h"
#include "config.h"
#include "frontend.h"
#include "http.h"
#if defined(AUDIOCPP_HAS_NATIVE_MODEL_MANAGER)
#include "model_installer.h"
#endif

#include "../streaming/streaming.h"

#include "engine/framework/io/json.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace minitts::server {

class ServerState final : public IHttpHandler, public ServerFrontendContext {
public:
    ServerState(
        ServerConfig config,
        std::filesystem::path request_base,
        std::filesystem::path ui_resource_anchor = {});
    ~ServerState() override;

    HttpResponse handle(const HttpRequest & request) override;
    HttpResponse forward_to_core(const HttpRequest & request) override;
    std::filesystem::path resolve_request_path(const std::filesystem::path & path) const override;
    std::filesystem::path make_frontend_temp_path(std::string_view filename) override;

    // Server-level `live_ingest` policy with this request's model override applied.
    // Deliberately does not reject an unknown or non-streaming model: it runs before
    // the handler, and rejecting here would turn a client's mistake into a dropped
    // connection instead of the 400 handle_transcription_live already produces.
    LiveIngestLimits live_ingest_limits(const HttpRequest & request) const override;

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
        std::atomic<bool> loaded{false};
        // Steady-clock ms of the most recent load or run of this model. Orders
        // eviction when max_loaded_models forces an unload: the least recently
        // used idle model goes first.
        std::atomic<std::int64_t> last_used_ms{0};
        mutable std::shared_mutex metadata_mutex;
        std::unordered_map<std::string, RuntimeVoicePreset> voice_presets;
        std::optional<RuntimeVoicePreset> default_voice_preset;
        // Whether this model's contract accepts the `reference_text` request
        // option, resolved once at registration (refresh_model_option_flags).
        // Resolving it per request re-reads the model file's embedded spec on
        // the request thread, which costs ~0.9 s per request for large GGUFs.
        // `true` mirrors model_accepts_request_option's no-contract behavior.
        bool accepts_reference_text = true;
        // Same treatment for `language`. Clients send the field on every
        // transcription whether or not the user chose one, so a model whose
        // contract omits it would reject the whole request over an option
        // nobody set. Resolved once at registration for the same cost reason.
        bool accepts_language = true;
        // Serializes runs on this model and bounds how long a caller waits for its
        // turn; see BusyGuard.
        BusyGuard busy;

        // Release the loaded model and session from memory (frees VRAM on GPU backends).
        // The next request will trigger a reload via ensure_model_loaded_locked().
        void unload();
    };

    // Acquire the model's run guard. `request_timeout_ms` is the caller-supplied
    // override, clamped by this model's configured ceiling. Throws ServerBusyError
    // (-> HTTP 503) once the effective timeout has elapsed.
    BusyGuard::Lock acquire_model_run(LoadedModel & model, std::optional<int> request_timeout_ms);

    // Server policy for this model: its own busy_timeout_ms if set, else the
    // top-level config value.
    engine::runtime::RunMode model_run_mode(const LoadedModel & model) const;

    void load_models();
    HttpResponse handle_request(const HttpRequest & request, bool use_frontends);
    std::unique_ptr<LoadedModel> make_model(ServerModelConfig config);
    // Recompute the per-model, config-derived request-option flags (currently
    // accepts_reference_text). Called at registration and on reconfiguration.
    void refresh_model_option_flags(LoadedModel & model);
    std::filesystem::path resolve_ui_model_path(const std::filesystem::path & path) const;
    HttpResponse handle_model_load(const std::string & body_text);
    HttpResponse handle_model_unload(const std::string & body_text);
    HttpResponse handle_path_status(const std::string & body_text) const;
    HttpResponse handle_ui_upload(const HttpRequest & request);
#if defined(AUDIOCPP_HAS_NATIVE_MODEL_MANAGER)
    HttpResponse handle_model_install(const std::string & body_text);
    HttpResponse handle_model_install_stop(const std::string & body_text);
    HttpResponse handle_model_clean_partial(const std::string & body_text);
    HttpResponse handle_model_remove(const std::string & body_text);
    HttpResponse handle_model_install_status(const HttpRequest & request) const;
    HttpResponse handle_model_package_sizes();
    HttpResponse handle_models_root_get() const;
    HttpResponse handle_models_root_set(const std::string & body_text);
    HttpResponse handle_directory_browser(const std::string & body_text) const;
#endif
    HttpResponse handle_ui_asset() const;
    HttpResponse handle_ui_voice_preview(const HttpRequest & request) const;
    LoadedModel::RuntimeVoicePreset load_runtime_voice_preset(const ServerModelConfig::VoicePreset & preset) const;
    void load_voice_presets(LoadedModel & model) const;
    void ensure_model_loaded_locked(LoadedModel & model);
    // With max_loaded_models set, unload least recently used idle models until
    // `loading` fits within the limit. A model mid-inference is never a victim;
    // when nothing can be evicted this throws ServerBusyError (-> HTTP 503).
    void evict_for_model_limit(const LoadedModel & loading);
    // Refuse the load with InsufficientMemoryError (-> HTTP 503) when the
    // estimated footprint plus configured headroom does not fit the free host
    // memory and (for GPU backends) the backend device memory.
    void ensure_model_fits_memory(const ServerModelConfig & model);
    LoadedModel & require_model(const engine::io::json::Value & body);
    const LoadedModel::RuntimeVoicePreset * select_voice_preset(
        const LoadedModel & model,
        const engine::io::json::Value & body,
        bool & voice_field_is_preset) const;
    engine::runtime::TaskRequest build_speech_request(
        const LoadedModel & model,
        const engine::io::json::Value & body) const;
    engine::runtime::TaskRequest apply_default_request_options(
        const LoadedModel & model,
        engine::runtime::TaskRequest request) const;
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
    // Shared body of the two entry points above/below; `audio` selects the source.
    TimedTaskResult run_streaming_model_impl(
        LoadedModel & model,
        const engine::runtime::TaskRequest & request,
        const minitts::app::AudioChunkStream * audio,
        const std::function<void(const engine::runtime::StreamEvent &)> & event_sink,
        std::optional<int> busy_timeout_ms);
    // Same as run_streaming_model, but pulls audio from `audio` instead of
    // `request.audio_input`, so the samples are never fully materialized.
    TimedTaskResult run_streaming_model_from(
        LoadedModel & model,
        const engine::runtime::TaskRequest & request,
        const minitts::app::AudioChunkStream & audio,
        const std::function<void(const engine::runtime::StreamEvent &)> & event_sink = {},
        std::optional<int> busy_timeout_ms = std::nullopt);
    HttpResponse handle_speech(const std::string & body_text);
    HttpResponse handle_speech_stream(
        LoadedModel & model,
        const engine::runtime::TaskRequest & request,
        const engine::io::json::Value & body);
    HttpResponse handle_speech_live(const HttpRequest & request);
    // detail selects the /v1/audio/transcriptions/details response, which adds the
    // segment, speaker-turn and word arrays the plain route drops.
    HttpResponse handle_transcription(const HttpRequest & request, bool detail = false);
    HttpResponse handle_transcription_json(const std::string & body_text, bool detail = false);
    HttpResponse handle_transcription_multipart(
        const std::string & body_text, const std::string & boundary, bool detail = false);
    HttpResponse run_transcription(
        LoadedModel & model,
        const engine::runtime::TaskRequest & request,
        std::optional<int> busy_timeout_ms = std::nullopt,
        bool detail = false);
    HttpResponse run_transcription_stream(
        LoadedModel & model,
        const engine::runtime::TaskRequest & request,
        std::optional<int> busy_timeout_ms = std::nullopt);
    HttpResponse handle_alignment(const HttpRequest & request);
    HttpResponse handle_alignment_multipart(const std::string & body_text, const std::string & boundary);
    HttpResponse run_alignment(
        LoadedModel & model,
        const engine::runtime::TaskRequest & request,
        std::optional<int> busy_timeout_ms = std::nullopt);
    HttpResponse handle_transcription_live(const HttpRequest & request);
    HttpResponse handle_generic_run(const std::string & body_text);
    HttpResponse handle_generic_stream(const std::string & body_text);
    HttpResponse handle_voices(const HttpRequest & request) const;
    HttpResponse handle_unload_models(const std::string & body_text);
    HttpResponse handle_unload_all_models();
    // Background loop started when idle_unload_ms > 0: when the server has gone
    // that long without a model load/run, unloads every resident (non-busy) model.
    void idle_unload_loop();
    void unload_idle_models();
    std::string models_json(bool include_session_options = false) const;
    std::string get_allowed_origin(const HttpRequest & request) const;

    ServerConfig config_;
    std::filesystem::path request_base_;
    std::vector<std::unique_ptr<LoadedModel>> models_;
    std::unordered_map<std::string, size_t> model_index_;
    mutable std::mutex models_mutex_;
    // Serializes framework loads while max_loaded_models or the memory guard
    // (min_free_memory_mb) is active, so two concurrent lazy loads cannot both
    // pass the eviction/memory check and overshoot. Not taken when both are off:
    // unrelated first loads stay concurrent there.
    std::mutex model_load_mutex_;
    std::filesystem::path upload_root_;
    std::mutex upload_root_mutex_;
    std::filesystem::path repository_root_;
#if defined(AUDIOCPP_HAS_NATIVE_MODEL_MANAGER)
    std::filesystem::path default_models_root_;
    std::filesystem::path models_root_;
    mutable std::mutex model_installer_mutex_;
    std::unique_ptr<ModelInstaller> model_installer_;
#endif
    std::atomic<uint64_t> next_upload_id_{1};
    // Steady-clock ms of the most recent model load/run completion; drives idle
    // unload. Updated at run start and again at completion so a long inference
    // does not read as idle the moment it finishes.
    std::atomic<std::int64_t> last_activity_ms_{0};
    std::atomic<bool> idle_unload_shutdown_{false};
    std::thread idle_unload_thread_;
    ServerFrontendRegistry frontends_;
};

}  // namespace minitts::server
