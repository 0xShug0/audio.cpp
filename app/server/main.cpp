#include "config.h"
#include "http.h"
#include "runtime.h"

#include "engine/framework/debug/trace.h"

#include <csignal>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

volatile std::sig_atomic_t g_shutdown_requested = 0;

void request_shutdown(int) {
    g_shutdown_requested = 1;
}

bool shutdown_requested() {
    return g_shutdown_requested != 0;
}

std::optional<std::string> arg_value(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return std::string(argv[i + 1]);
        }
    }
    return std::nullopt;
}

bool has_arg(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == name) {
            return true;
        }
    }
    return false;
}

std::filesystem::path executable_directory(const char * argv0) {
    if (argv0 == nullptr || *argv0 == '\0') {
        return std::filesystem::current_path();
    }
    std::error_code ec;
    auto path = std::filesystem::absolute(std::filesystem::path(argv0), ec);
    if (ec) {
        return std::filesystem::current_path();
    }
    path = path.lexically_normal();
    if (std::filesystem::is_regular_file(path, ec)) {
        return path.parent_path();
    }
    return std::filesystem::current_path();
}

void print_help() {
    std::cout
        << "audiocpp_server [--config <server.json>] [--ui] [--host <ip>] [--port <port>] [--backend <backend>]\n"
        << "                [--device <id>] [--threads <n>] [--busy-timeout-ms <ms>]\n"
        << "                [--model-spec-override <json-or-directory>] [--voice-dir <directory>]\n"
        << "                [--log] [--log-file <path>]\n"
        << "                [--cors-origins <origins>]\n"
        << "  --ui                             serve the embedded WebUI\n"
        << "  --no-ui                          disable the embedded WebUI\n"
        << "  --ui-management                  allow WebUI model management and downloads; requires\n"
        << "                                   AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON at build time\n"
        << "  --backend cpu|cuda|hip|rocm|vulkan|metal  default cuda (rocm is an alias for hip)\n"
        << "  --busy-timeout-ms <ms>           fail a request with 503 when the model has been\n"
        << "                                   busy this long; default 300000, 0 disables\n"
        << "  --voice-dir <directory>          override the shared reference voice library directory\n"
        << "  --cors-origins \"*\"              experimental; disabled by default. Allows browser\n"
        << "                                   requests from any origin for trusted local demos only\n"
        << "\n"
        << "Endpoints:\n"
        << "  GET  /                           embedded WebUI (enabled by default with a config)\n"
        << "  GET  /health\n"
        << "  GET  /v1/models\n"
        << "  POST /v1/models/load             available with --ui-management\n"
        << "  POST /v1/models/unload           available with --ui-management\n"
        << "  POST /v1/ui/upload               temporary browser upload for WebUI requests\n"
        << "  POST /v1/ui/models/install       background package download/preparation\n"
        << "  POST /v1/ui/models/install/stop  stop one active package download\n"
        << "  POST /v1/ui/models/clean-partial remove abandoned package staging files\n"
        << "  POST /v1/ui/models/delete        remove one installed package precision\n"
        << "  GET  /v1/ui/models-root          current and binary-local default models folders\n"
        << "  POST /v1/ui/models-root          select a models folder (empty path restores default)\n"
        << "  POST /v1/ui/browse-directories   list local folders for the native folder picker\n"
        << "  GET  /v1/ui/models/install-status[?id=<package>]\n"
        << "  GET  /v1/ui/models/package-sizes package sizes from metadata-only checks\n"
        << "  GET  /v1/audio/voices?model=<id>\n"
        << "  POST /v1/audio/speech\n"
        << "  POST /v1/audio/speech/live?model=<id>\n"
        << "       raw PCM in a chunked body, speech audio deltas as SSE on the same connection\n"
        << "  POST /v1/audio/transcriptions\n"
        << "       fields: file, model, language, prompt, stream\n"
        << "       OpenAI-style streaming: speech stream_format=sse|audio, transcription stream=true\n"
        << "  POST /v1/audio/transcriptions/live?model=<id>\n"
        << "       raw PCM in a chunked body, transcript deltas as SSE on the same connection\n"
        << "  POST /v1/tasks/run\n";
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        if (has_arg(argc, argv, "--help") || has_arg(argc, argv, "-h")) {
            print_help();
            return 0;
        }
        const auto config_path = arg_value(argc, argv, "--config");
        const bool ui_requested = has_arg(argc, argv, "--ui");
        if (!config_path.has_value() && !ui_requested) {
            throw std::runtime_error("missing required --config argument (or use --ui for the native WebUI)");
        }
        const auto log_file = arg_value(argc, argv, "--log-file");
        engine::debug::configure_logging(engine::debug::LoggingConfig{
            has_arg(argc, argv, "--log") || log_file.has_value(),
            log_file,
        });
        std::signal(SIGINT, request_shutdown);
        std::signal(SIGTERM, request_shutdown);
#ifdef SIGPIPE
        // Writing to a socket whose peer has already disconnected (for example a
        // client that closed an SSE/chunked stream early) would otherwise deliver
        // SIGPIPE and terminate the whole server. Ignore it so the failed send
        // surfaces as an EPIPE error on that single request thread, which
        // handle_client already unwinds cleanly, instead of taking the process down.
        std::signal(SIGPIPE, SIG_IGN);
#endif

        auto config = config_path.has_value()
            ? minitts::server::load_server_config(*config_path)
            : minitts::server::ServerConfig{};
        if (!config_path.has_value()) {
            config.lazy_load = true;
        }
        if (ui_requested) {
            config.ui_enabled = true;
        }
        if (has_arg(argc, argv, "--no-ui")) {
            config.ui_enabled = false;
        }
        if (has_arg(argc, argv, "--ui-management")) {
            config.ui_management = true;
        }
#if !defined(AUDIOCPP_HAS_NATIVE_MODEL_MANAGER)
        if (config.ui_management) {
            throw std::runtime_error(
                "UI model management is not available in this build; reconfigure with "
                "-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON");
        }
#endif
        if (const auto host = arg_value(argc, argv, "--host")) {
            config.host = *host;
        }
        if (const auto port = arg_value(argc, argv, "--port")) {
            config.port = std::stoi(*port);
        }
        if (const auto cors_origins = arg_value(argc, argv, "--cors-origins")) {
            config.cors_origins = *cors_origins;
        }
        if (const auto backend = arg_value(argc, argv, "--backend")) {
            config.backend = minitts::server::parse_server_backend(*backend);
        }
        if (const auto device = arg_value(argc, argv, "--device")) {
            config.device = std::stoi(*device);
        }
        if (const auto threads = arg_value(argc, argv, "--threads")) {
            config.threads = std::stoi(*threads);
        }
        if (const auto busy_timeout = arg_value(argc, argv, "--busy-timeout-ms")) {
            config.busy_timeout_ms = std::stoi(*busy_timeout);
        }
        if (const auto model_spec = arg_value(argc, argv, "--model-spec-override")) {
            config.model_spec_override = std::filesystem::path(*model_spec);
        }
        if (const auto voice_dir = arg_value(argc, argv, "--voice-dir")) {
            config.voice_dir = std::filesystem::path(*voice_dir);
        }
        if (!(config.cors_origins == "*" || config.cors_origins == "")) {
            throw std::runtime_error("--cors-origins must be '*' (allow all origins) or '' (disabled)");
        }
        if (config.threads <= 0) {
            throw std::runtime_error("--threads must be positive");
        }
        if (config.busy_timeout_ms < 0) {
            throw std::runtime_error("--busy-timeout-ms must be >= 0 (0 disables the guard)");
        }

        const auto ui_resource_anchor = executable_directory(argc > 0 ? argv[0] : nullptr);
        minitts::server::ServerState state(
            config,
            std::filesystem::current_path(),
            ui_resource_anchor);
        minitts::server::serve_http(config.host, config.port, state, shutdown_requested, config.max_request_body_bytes);
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "audiocpp_server failed: " << ex.what() << "\n";
        return 1;
    }
}
