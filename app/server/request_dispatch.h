#pragma once

#include "busy_guard.h"
#include "model_memory.h"
#include "server_protocol.h"

#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/errors.h"

#include <string>

namespace minitts::server {

// Shares HTTP protocol dispatch without coupling the legacy and parallel
// runtime state or lifecycle implementations.
struct RequestDispatcher {
  template <typename State>
  static HttpResponse handle(State &state, const HttpRequest &request,
                             bool use_frontends) {
    HttpResponse response;
    const std::string allowed_origin = state.get_allowed_origin(request);
    try {
      log_request_body_if_enabled(state.config_, request);
      if (request.method == "OPTIONS" &&
          (!allowed_origin.empty() || state.config_.ui_enabled)) {
        response.status = 204;
        response.content_type = "text/plain";
        response.headers["Access-Control-Allow-Headers"] = "*";
        response.headers["Access-Control-Allow-Methods"] = "GET, POST";
      } else if (use_frontends && !state.frontends_.empty()) {
        response = state.frontends_.handle(state, request);
      } else if (request.method == "GET" &&
                 (request.path == "/" || request.path == "/index.html")) {
        response = state.handle_ui_asset();
      } else if (request.method == "GET" && state.config_.ui_enabled &&
                 (request.path == "/sw.js" ||
                  request.path == "/service-worker.js")) {
        response = ui_service_worker_retirement_response();
      } else if (request.method == "GET" && request.path == "/favicon.ico") {
        response.status = 204;
        response.content_type = "image/x-icon";
      } else if (request.method == "GET" && request.path == "/health") {
        response = state.handle_health();
      } else if (request.method == "GET" && request.path == "/v1/models") {
        const auto include_session_options =
            query_param(request.query, "include_session_options");
        const auto include_params =
            query_param(request.query, "include_params");
        response = json_response(state.models_json(
            include_session_options == "true" || include_session_options == "1",
            include_params == "true" || include_params == "1"));
      } else if (request.method == "GET" &&
                 request.path == "/v1/audio/voices") {
        response = state.handle_voices(request);
      } else if (request.method == "GET" &&
                 request.path == "/v1/ui/voice-preview") {
        response = state.handle_ui_voice_preview(request);
      } else if (request.method == "POST" &&
                 request.path == "/v1/models/load") {
        response = state.handle_model_load(request.body);
      } else if (request.method == "POST" &&
                 request.path == "/v1/models/unload") {
        response = state.handle_model_unload(request.body);
      } else if (request.method == "POST" &&
                 request.path == "/v1/ui/path-status") {
        response = state.handle_path_status(request.body);
      } else if (request.method == "POST" && request.path == "/v1/ui/upload") {
        response = state.handle_ui_upload(request);
      }
#if defined(AUDIOCPP_HAS_NATIVE_MODEL_MANAGER)
      else if (request.method == "GET" &&
               request.path == "/v1/ui/models-root") {
        response = state.handle_models_root_get();
      } else if (request.method == "POST" &&
                 request.path == "/v1/ui/models-root") {
        response = state.handle_models_root_set(request.body);
      } else if (request.method == "POST" &&
                 request.path == "/v1/ui/browse-directories") {
        response = state.handle_directory_browser(request.body);
      } else if (request.method == "POST" &&
                 request.path == "/v1/ui/models/install") {
        response = state.handle_model_install(request.body);
      } else if (request.method == "POST" &&
                 request.path == "/v1/ui/models/install/stop") {
        response = state.handle_model_install_stop(request.body);
      } else if (request.method == "POST" &&
                 request.path == "/v1/ui/models/clean-partial") {
        response = state.handle_model_clean_partial(request.body);
      } else if (request.method == "POST" &&
                 request.path == "/v1/ui/models/delete") {
        response = state.handle_model_remove(request.body);
      } else if (request.method == "GET" &&
                 request.path == "/v1/ui/models/install-status") {
        response = state.handle_model_install_status(request);
      } else if (request.method == "GET" &&
                 request.path == "/v1/ui/models/package-sizes") {
        response = state.handle_model_package_sizes();
      }
#endif
      else if (request.method == "POST" && request.path == "/v1/audio/speech") {
        response = state.handle_speech(request.body);
      } else if (request.method == "POST" &&
                 request.path == "/v1/audio/speech/live") {
        response = state.handle_speech_live(request);
      } else if (request.method == "POST" &&
                 request.path == "/v1/audio/transcriptions") {
        response = state.handle_transcription(request);
      }
      // Same request shape as the route above, opt-in richer response: the
      // models that align words or separate speakers report them through fields
      // the transcription response drops. A separate path rather than a flag
      // keeps the existing response schema fixed for every client already built
      // against it.
      else if (request.method == "POST" &&
               request.path == "/v1/audio/transcriptions/details") {
        response = state.handle_transcription(request, /*detail=*/true);
      } else if (request.method == "POST" &&
                 request.path == "/v1/batches/transcriptions") {
        response = state.handle_batch_transcriptions(request);
      } else if (request.method == "POST" &&
                 request.path == "/v1/audio/alignments") {
        response = state.handle_alignment(request);
      }
      // Separate path rather than a flag on the endpoint above: the input
      // transport differs (raw chunked PCM vs a complete upload), so keeping it
      // distinct leaves every existing transcription client untouched.
      else if (request.method == "POST" &&
               request.path == "/v1/audio/transcriptions/live") {
        response = state.handle_transcription_live(request);
      } else if (request.method == "POST" && request.path == "/v1/tasks/run") {
        response = state.handle_generic_run(request.body);
      } else if (request.method == "POST" &&
                 request.path == "/v1/tasks/batch") {
        response = state.handle_generic_batch(request.body);
      } else if (request.method == "POST" &&
                 request.path == "/v1/tasks/stream") {
        response = state.handle_generic_stream(request.body);
      } else if (request.method == "POST" &&
                 request.path == "/v1/tasks/unload_models") {
        response = state.handle_unload_models(request.body);
      } else if (request.method == "POST" &&
                 request.path == "/v1/tasks/unload_all_models") {
        response = state.handle_unload_all_models();
      } else {
        response = error_response(404, "unknown endpoint: " + request.path,
                                  "not_found");
      }
    } catch (const engine::runtime::CapacityError &ex) {
      // The request is too big for the device, which is the caller's to fix --
      // reporting it as 500 sends them looking for a server fault that is not
      // there. Checked before ServerBusyError only because both are
      // runtime_error; the two conditions are disjoint.
      response = error_response(400, ex.what(), "invalid_request_error");
    } catch (const engine::runtime::InvalidRequestError &ex) {
      // A request turned away as sent, such as an input the model does not
      // take: the caller's to fix as well.
      response = error_response(400, ex.what(), "invalid_request_error");
    } catch (const InsufficientMemoryError &ex) {
      response = error_response(503, ex.what(), "insufficient_memory");
    } catch (const ServerBusyError &ex) {
      // Non-streaming requests surface the busy state as 503 before any
      // response is sent. (Streaming requests acquire the lock inside the
      // stream body, after headers are sent, so there it becomes a stream error
      // event instead.)
      response = error_response(503, ex.what(), "server_busy");
    } catch (const std::exception &ex) {
      if (allowed_origin.empty()) {
        throw;
      }
      // Keep the transport's error response, but attach CORS before returning
      // it.
      if (engine::debug::log_enabled()) {
        engine::debug::log_message(
            std::string("[SERVER_HTTP_DEBUG] http.error ") + ex.what());
      }
      response = error_response(500, ex.what(), "server_error");
    }
    if (!allowed_origin.empty()) {
      response.headers["Access-Control-Allow-Origin"] = allowed_origin;
    }
    return response;
  }
};

} // namespace minitts::server
