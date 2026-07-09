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

void print_help() {
    std::cout
        << "audiocpp_server --config <server.json> [--host <ip>] [--port <port>] [--backend <backend>]\n"
        << "                [--device <id>] [--threads <n>]\n"
        << "                [--log] [--log-file <path>]\n"
        << "  --backend cpu|cuda|vulkan|metal  default cuda\n"
        << "\n"
        << "Endpoints:\n"
        << "  GET  /health\n"
        << "  GET  /v1/models\n"
        << "  GET  /v1/audio/voices?model=<id>\n"
        << "  POST /v1/audio/speech\n"
        << "  POST /v1/audio/transcriptions\n"
        << "       OpenAI-style streaming: speech stream_format=sse|audio, transcription stream=true\n"
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
        if (!config_path.has_value()) {
            throw std::runtime_error("missing required --config argument");
        }
        const auto log_file = arg_value(argc, argv, "--log-file");
        engine::debug::configure_logging(engine::debug::LoggingConfig{
            has_arg(argc, argv, "--log") || log_file.has_value(),
            log_file,
        });
        std::signal(SIGINT, request_shutdown);
        std::signal(SIGTERM, request_shutdown);

        auto config = minitts::server::load_server_config(*config_path);
        if (const auto host = arg_value(argc, argv, "--host")) {
            config.host = *host;
        }
        if (const auto port = arg_value(argc, argv, "--port")) {
            config.port = std::stoi(*port);
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
        if (config.threads <= 0) {
            throw std::runtime_error("--threads must be positive");
        }

        minitts::server::ServerState state(config, std::filesystem::current_path());
        minitts::server::serve_http(config.host, config.port, state, shutdown_requested);
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "audiocpp_server failed: " << ex.what() << "\n";
        return 1;
    }
}
