#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace minitts::server {

struct HttpRequest {
    std::string method;
    std::string path;
    std::string query;  // raw query string, e.g. "model=pocket-tts" (no leading '?')
    std::unordered_map<std::string, std::string> headers;
    std::string body;
};

struct HttpResponse {
    int status = 200;
    std::string content_type = "application/json";
    std::string body;
    std::unordered_map<std::string, std::string> headers;
    std::function<void(class HttpStreamWriter &)> stream_body;
};

class HttpStreamWriter {
public:
    virtual ~HttpStreamWriter() = default;
    virtual void write(std::string_view data) = 0;
};

class IHttpHandler {
public:
    virtual ~IHttpHandler() = default;
    virtual HttpResponse handle(const HttpRequest & request) = 0;
};

using ShutdownRequested = bool (*)();

HttpResponse json_response(std::string body, int status = 200);
HttpResponse error_response(int status, const std::string & message, const std::string & type);
void serve_http(const std::string & host, int port, IHttpHandler & handler, ShutdownRequested shutdown_requested);

}  // namespace minitts::server
