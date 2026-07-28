#pragma once

#include <cstdint>
#include <functional>
#include <iosfwd>
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

    // Non-null only for a route that opts into an incremental body AND whose
    // client declared `Transfer-Encoding: chunked`. The body is then NOT collected
    // into `body` before the handler runs — reads on this stream block until the
    // client sends more, and reach EOF at the terminating chunk. This is what lets a
    // handler consume audio while it is still being captured; every other request,
    // on every other route, still arrives fully buffered in `body`.
    //
    // A half-close WITHOUT that terminating chunk is an error, not EOF: a body cut
    // short by a dropped connection would otherwise be indistinguishable from one
    // the client finished deliberately.
    //
    // The stream reads directly from the connection, so it stays valid for the
    // whole exchange: through `handle()` and through any `HttpResponse::stream_body`
    // that call returns. A handler may therefore capture it in `stream_body`, which
    // is how the live route reads audio while writing SSE back. It must not outlive
    // that callback — nothing owns it once the connection is done.
    //
    // Reading it can throw: the stream is configured with `exceptions(badbit)` so a
    // stall, a disconnect before the terminating chunk, or a malformed frame surfaces
    // as an exception rather than as a silently short body.
    std::istream * body_stream = nullptr;
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
void serve_http(const std::string & host, int port, IHttpHandler & handler, ShutdownRequested shutdown_requested, uint64_t max_request_body_bytes);

}  // namespace minitts::server
