#pragma once

#include "http.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace minitts::server {

class ServerFrontendContext {
public:
    virtual ~ServerFrontendContext() = default;

    virtual HttpResponse forward_to_core(const HttpRequest & request) = 0;
    virtual std::filesystem::path resolve_request_path(const std::filesystem::path & path) const = 0;
    virtual std::filesystem::path make_frontend_temp_path(std::string_view filename) = 0;
};

struct ServerFrontendRequest {
    HttpRequest request;
    std::optional<HttpResponse> response;
};

struct ServerFrontendResponse {
    const HttpRequest & original_request;
    const HttpRequest & core_request;
    HttpResponse response;
};

struct FrontendPreContract {
    std::string_view method;
    std::string_view path;
    std::string_view request_in;
    std::string_view request_out;
};

struct FrontendPostContract {
    std::string_view method;
    std::string_view path;
    std::string_view response_in;
    std::string_view response_out;
};

namespace frontend_contracts {
inline constexpr std::string_view any = "*";
inline constexpr std::string_view client_encoded_audio_request = "client_encoded_audio_request";
inline constexpr std::string_view core_wav_audio_request = "core_wav_audio_request";
inline constexpr std::string_view client_mp3_speech_request = "client_mp3_speech_request";
inline constexpr std::string_view core_wav_speech_request = "core_wav_speech_request";
inline constexpr std::string_view core_wav_speech_response = "core_wav_speech_response";
inline constexpr std::string_view client_mp3_speech_response = "client_mp3_speech_response";
} // namespace frontend_contracts

class ServerFrontendModule {
public:
    virtual ~ServerFrontendModule() = default;

    virtual std::string_view name() const = 0;
    virtual std::optional<FrontendPreContract> pre_contract() const;
    virtual std::optional<FrontendPostContract> post_contract() const;
    virtual void pre_process(ServerFrontendContext & context, ServerFrontendRequest & request);
    virtual void post_process(ServerFrontendContext & context, ServerFrontendResponse & response);
};

using ServerFrontendModuleFactory = std::unique_ptr<ServerFrontendModule> (*)();

struct ServerFrontendHttpsConfig {
    std::filesystem::path cert_file;
    std::filesystem::path key_file;
};

class ServerFrontendRegistry {
public:
    void add(ServerFrontendModuleFactory factory);
    bool empty() const;
    HttpResponse handle(ServerFrontendContext & context, const HttpRequest & request) const;

private:
    std::vector<std::unique_ptr<ServerFrontendModule>> modules_;
};

void register_static_server_frontends(ServerFrontendRegistry & registry);
void serve_frontend_https(
    const std::string & host,
    int port,
    IHttpHandler & handler,
    ShutdownRequested shutdown_requested,
    uint64_t max_request_body_bytes,
    const ServerFrontendHttpsConfig & config);

} // namespace minitts::server
