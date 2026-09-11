#pragma once

#include <cstdint>
#include <functional>
#include <iosfwd>
#include <string>
#include <string_view>
#include <unordered_map>

namespace minitts::server {

// Bounds on an incrementally delivered request body. Every one of them is really a
// bound on how long a single client can hold a model: this body is read while the
// model lock is held, so a peer that stalls, never stops, or names an enormous chunk
// would otherwise pin the model or exhaust the host.
//
// Uniformly, 0 disables that bound — the same convention `busy_timeout_ms` already
// uses. Disabling one is a deliberate choice for a specific deployment (an hour-long
// dictation seat behind a trusted proxy), not a default worth shipping.
struct LiveIngestLimits {
    // Longest gap between two receives before the body is declared stalled.
    int idle_timeout_ms = 30'000;
    // Wall-clock ceiling on the whole body, however steadily it arrives.
    int total_timeout_ms = 600'000;
    size_t max_body_bytes = 512u * 1024u * 1024u;
    // Bounds one chunk, independently of the total. A chunk is held in memory twice
    // while being de-framed, so without this a single legal chunk could name a size
    // large enough to exhaust the host — while holding a model lock. The one bound
    // that must stay non-zero: it also backs the overflow check in the chunk-size
    // parser, so a 0 would re-admit a declared size of SIZE_MAX. Config rejects it.
    size_t max_chunk_bytes = 8u * 1024u * 1024u;
    // SO_SNDTIMEO on the connection, so a client that stops reading the response
    // cannot hold the model open indefinitely. Per send() call, not an overall
    // response deadline.
    int send_timeout_ms = 30'000;
};

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

    // Bounds to apply to this request's incremental body, asked for before the body
    // stream is created and therefore before `handle()` runs. The transport has no
    // notion of models, so resolving a per-model override is the handler's job; the
    // default is the compiled-in policy above. Called only for a request that opts
    // into an incremental body.
    virtual LiveIngestLimits live_ingest_limits(const HttpRequest & request) const {
        (void) request;
        return {};
    }
};

using ShutdownRequested = bool (*)();

HttpResponse json_response(std::string body, int status = 200);
HttpResponse error_response(int status, const std::string & message, const std::string & type);
void serve_http(const std::string & host, int port, IHttpHandler & handler, ShutdownRequested shutdown_requested, uint64_t max_request_body_bytes);

}  // namespace minitts::server
