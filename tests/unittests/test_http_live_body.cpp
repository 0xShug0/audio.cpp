// Framing tests for the incrementally delivered request body used by
// POST /v1/audio/transcriptions/live.
//
// Driven over a real loopback socket against serve_http rather than against the
// streambuf directly: the behaviour under test is the interaction between the
// header parser, the undrained socket and the handler, and a direct unit test of
// the buffer would not cover the part most likely to break.
//
// The cases that matter are the ones where a wrong answer is INVISIBLE. A body
// truncated by a stall, a disconnect or a desynchronised sender must surface as
// an error, because for audio a silently short body is indistinguishable from a
// speaker who simply stopped talking.
//
// Scope, stated so nobody mistakes this for end-to-end coverage: it links
// http.cpp and not the server runtime, so it exercises the framing, the bounds
// and the isolation of other routes — and NOT the `/v1/audio/transcriptions/live`
// route itself, its query-parameter defaults, the PCM adapter, the model
// invocation, or the conversion of a body error into an SSE `error` event. Those
// need a loaded model, are not covered anywhere in this suite, and were verified
// by hand against a running server. Removing the route would not fail this test.

#include "../../app/server/http.h"

#include "test_assert.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <istream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {

using engine::test::require;
using engine::test::require_eq;

constexpr int kPort = 18094;
constexpr const char * kLivePath = "/v1/audio/transcriptions/live";

// The socket API's handle type is not `int` on Windows, and narrowing it there
// truncates a valid handle.
#ifdef _WIN32
using socket_t = SOCKET;
constexpr socket_t kInvalidSocket = INVALID_SOCKET;
#else
using socket_t = int;
constexpr socket_t kInvalidSocket = -1;
#endif

std::atomic<bool> g_stop{false};
bool stop_requested() {
    return g_stop.load();
}

// Reports what the body stream actually yielded, so a test can tell a complete
// body from a truncated one and a truncation from an error.
class EchoHandler final : public minitts::server::IHttpHandler {
public:
    minitts::server::HttpResponse handle(const minitts::server::HttpRequest & request) override {
        if (request.body_stream == nullptr) {
            return minitts::server::json_response(
                "{\"stream\":false,\"buffered\":" + std::to_string(request.body.size()) + "}");
        }
        std::string body;
        try {
            char buffer[4096];
            while (request.body_stream->read(buffer, sizeof(buffer)) || request.body_stream->gcount() > 0) {
                body.append(buffer, static_cast<size_t>(request.body_stream->gcount()));
            }
        } catch (const std::exception & ex) {
            return minitts::server::json_response(
                std::string("{\"error\":\"") + ex.what() + "\"}");
        }
        size_t sum = 0;
        for (const unsigned char byte : body) {
            sum += byte;
        }
        return minitts::server::json_response(
            "{\"stream\":true,\"bytes\":" + std::to_string(body.size()) +
            ",\"sum\":" + std::to_string(sum) + "}");
    }

    // Stands in for the server's per-model resolution: a request naming
    // `model=tight` gets tightened bounds, everything else the defaults. The real
    // handler reads them from config; what matters here is that the transport asks,
    // and that what it is told actually takes effect.
    minitts::server::LiveIngestLimits live_ingest_limits(
        const minitts::server::HttpRequest & request) const override {
        limits_queries.fetch_add(1);
        if (request.query.find("model=nocap") != std::string::npos) {
            // Not reachable through config, which rejects it — but the transport must
            // not depend on config having done so.
            minitts::server::LiveIngestLimits nocap;
            nocap.max_chunk_bytes = 0;
            return nocap;
        }
        if (request.query.find("model=tiny") != std::string::npos) {
            // Small enough that a single hex digit exceeds it, which is where the
            // chunk-size guard used to underflow.
            minitts::server::LiveIngestLimits tiny;
            tiny.max_chunk_bytes = 1;
            return tiny;
        }
        if (request.query.find("model=tight") == std::string::npos) {
            return {};
        }
        minitts::server::LiveIngestLimits tight;
        tight.max_chunk_bytes = 1024;
        tight.max_body_bytes = 4096;
        return tight;
    }

    mutable std::atomic<int> limits_queries{0};
};

void close_socket(socket_t handle) {
#ifdef _WIN32
    closesocket(handle);
#else
    close(handle);
#endif
}

// Retries rather than trusting a fixed startup sleep: the listener becomes ready
// on its own thread, and a sleep long enough to always win is either flaky or
// wasteful.
socket_t connect_to_server() {
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(kPort));
    inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
    for (int attempt = 0; attempt < 200; ++attempt) {
        const socket_t handle = socket(AF_INET, SOCK_STREAM, 0);
        require(handle != kInvalidSocket, "could not create client socket");
        if (connect(handle, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0) {
            return handle;
        }
        close_socket(handle);
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    require(false, "could not connect to the test server");
    return kInvalidSocket;
}

void send_raw(socket_t handle, const std::string & data) {
    size_t offset = 0;
    while (offset < data.size()) {
#ifdef _WIN32
        const int written = send(handle, data.data() + offset, static_cast<int>(data.size() - offset), 0);
#else
        const ssize_t written = send(handle, data.data() + offset, data.size() - offset, 0);
#endif
        require(written > 0, "send to the test server failed");
        offset += static_cast<size_t>(written);
    }
}

std::string read_reply(socket_t handle) {
    std::string reply;
    char buffer[4096];
    for (;;) {
#ifdef _WIN32
        const int received = recv(handle, buffer, sizeof(buffer), 0);
#else
        const ssize_t received = recv(handle, buffer, sizeof(buffer), 0);
#endif
        if (received <= 0) {
            break;
        }
        reply.append(buffer, static_cast<size_t>(received));
    }
    return reply;
}

std::string headers_for(const char * path, bool chunked) {
    std::string out = std::string("POST ") + path + " HTTP/1.1\r\nHost: test\r\n";
    out += chunked ? "Transfer-Encoding: chunked\r\n" : "Content-Length: 0\r\n";
    return out + "\r\n";
}

// Same, with a verbatim Transfer-Encoding value, for testing how the header is parsed.
std::string headers_with_encoding(const char * path, const std::string & encoding) {
    return std::string("POST ") + path + " HTTP/1.1\r\nHost: test\r\nTransfer-Encoding: " +
           encoding + "\r\n\r\n";
}

// Sends `body` as one chunk per entry and returns the server's reply.
std::string round_trip(const std::vector<std::string> & chunks, bool terminate = true,
                       const char * path = kLivePath, const std::string & raw_tail = {}) {
    const socket_t handle = connect_to_server();
    send_raw(handle, headers_for(path, true));
    for (const auto & piece : chunks) {
        std::ostringstream frame;
        frame << std::hex << piece.size() << "\r\n" << piece << "\r\n";
        send_raw(handle, frame.str());
    }
    if (!raw_tail.empty()) {
        send_raw(handle, raw_tail);
    }
    if (terminate) {
        send_raw(handle, "0\r\n\r\n");
    } else {
#ifdef _WIN32
        shutdown(handle, SD_SEND);
#else
        shutdown(handle, SHUT_WR);
#endif
    }
    const std::string reply = read_reply(handle);
    close_socket(handle);
    return reply;
}

bool contains(const std::string & haystack, const std::string & needle) {
    return haystack.find(needle) != std::string::npos;
}

std::string payload(size_t length, unsigned char seed) {
    std::string out;
    out.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        out.push_back(static_cast<char>((seed + i) % 251));
    }
    return out;
}

size_t byte_sum(const std::string & data) {
    size_t sum = 0;
    for (const unsigned char byte : data) {
        sum += byte;
    }
    return sum;
}

}  // namespace

int main() {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    EchoHandler handler;
    // The server default, spelled out rather than pulled from config.h so this test
    // keeps linking the transport alone. It bounds a Content-Length body and never
    // the incremental one under test, which has its own LiveIngestLimits.
    constexpr uint64_t kMaxBufferedBody = 2ull * 1024ull * 1024ull * 1024ull;
    std::thread server([&] {
        minitts::server::serve_http("127.0.0.1", kPort, handler, stop_requested, kMaxBufferedBody);
    });
    // No startup sleep: connect_to_server() retries until the listener is up.

    // A body split into several chunks must arrive byte-identical and complete.
    {
        const std::string a = payload(5000, 7);
        const std::string b = payload(3000, 61);
        const std::string reply = round_trip({a, b});
        require(contains(reply, "\"stream\":true"), "live path should expose a body stream");
        require(
            contains(reply, "\"bytes\":" + std::to_string(a.size() + b.size())),
            "multi-chunk body should arrive complete: " + reply);
        require(
            contains(reply, "\"sum\":" + std::to_string(byte_sum(a) + byte_sum(b))),
            "multi-chunk body should arrive byte-identical: " + reply);
    }

    // A single chunk far larger than the 8 KiB receive buffer, so the chunk is
    // assembled across many recv() calls rather than found whole in one.
    {
        const std::string big = payload(300'000, 11);
        const std::string reply = round_trip({big});
        require(contains(reply, "\"bytes\":" + std::to_string(big.size())),
                "a chunk spanning many receives should arrive complete: " + reply);
        require(contains(reply, "\"sum\":" + std::to_string(byte_sum(big))),
                "a chunk spanning many receives should arrive intact: " + reply);
    }

    // Drives total consumed bytes past the 64 KiB mark at which the de-framer
    // compacts its pending buffer. Compaction shifts the bytes the reader has not
    // consumed yet, so a mistake here silently corrupts or drops audio rather than
    // failing loudly — and it cannot happen at all in the small-payload cases.
    {
        std::vector<std::string> chunks;
        size_t total = 0;
        size_t sum = 0;
        for (int i = 0; i < 80; ++i) {
            chunks.push_back(payload(4096, static_cast<unsigned char>(i * 7)));
            total += chunks.back().size();
            sum += byte_sum(chunks.back());
        }
        require(total > 4 * 64 * 1024, "compaction case must exceed the compaction threshold");
        const std::string reply = round_trip(chunks);
        require(contains(reply, "\"bytes\":" + std::to_string(total)),
                "body spanning several compactions should be complete: " + reply);
        require(contains(reply, "\"sum\":" + std::to_string(sum)),
                "compaction must not corrupt or drop unconsumed bytes: " + reply);
    }

    // Many small chunks exercise the path where several chunks arrive inside a
    // single recv().
    {
        std::vector<std::string> chunks;
        size_t total = 0;
        size_t sum = 0;
        for (int i = 0; i < 40; ++i) {
            chunks.push_back(payload(97, static_cast<unsigned char>(i)));
            total += chunks.back().size();
            sum += byte_sum(chunks.back());
        }
        const std::string reply = round_trip(chunks);
        require(contains(reply, "\"bytes\":" + std::to_string(total)), "many small chunks: " + reply);
        require(contains(reply, "\"sum\":" + std::to_string(sum)), "many small chunks intact: " + reply);
    }

    // Chunk extensions are legal and must be ignored rather than parsed as size.
    {
        const socket_t handle = connect_to_server();
        send_raw(handle, headers_for(kLivePath, true));
        send_raw(handle, "4;name=value\r\nabcd\r\n0\r\n\r\n");
        const std::string reply = read_reply(handle);
        close_socket(handle);
        require(contains(reply, "\"bytes\":4"), "chunk extensions should be ignored: " + reply);
    }

    // Trailer headers after the terminating chunk are legal.
    {
        const socket_t handle = connect_to_server();
        send_raw(handle, headers_for(kLivePath, true));
        send_raw(handle, "4\r\nwxyz\r\n0\r\nX-Trailer: value\r\n\r\n");
        const std::string reply = read_reply(handle);
        close_socket(handle);
        require(contains(reply, "\"bytes\":4"), "trailers should be accepted: " + reply);
    }

    // Each malformed case asserts the SPECIFIC diagnostic, not merely that something
    // failed. Half-closing the connection to end these requests would itself produce
    // an error, so a bare "did it fail" check would pass even with the guard removed
    // — it would just be failing later, for the wrong reason.

    // A peer that closes without the terminating chunk must NOT look like a clean
    // end of body — that is the silent-truncation failure this endpoint cannot have.
    {
        const std::string reply = round_trip({payload(2000, 3)}, /*terminate=*/false);
        require(contains(reply, "closed before the terminating chunk"),
                "premature close must be an error, not a short body: " + reply);
    }

    // A declared size that would overflow size_t must be rejected before any
    // arithmetic on it, not wrap into a tiny length.
    {
        const std::string reply = round_trip({}, /*terminate=*/false, kLivePath, "ffffffffffffffff\r\nX");
        require(contains(reply, "chunk size exceeds the maximum"),
                "overflowing chunk size must be rejected by the size guard: " + reply);
    }

    // A chunk larger than the per-chunk cap must be rejected by its declared size,
    // before the data behind it is waited for.
    {
        const std::string reply = round_trip({}, /*terminate=*/false, kLivePath, "1000000\r\n");
        require(contains(reply, "chunk size exceeds the maximum"),
                "oversized chunk must be rejected by the size guard: " + reply);
    }

    // Chunk data must be CRLF-terminated; otherwise a desynchronised sender's
    // payload can be silently reinterpreted as framing.
    {
        const std::string reply = round_trip({}, /*terminate=*/false, kLivePath, "4\r\nabcdXX0\r\n\r\n");
        require(contains(reply, "not terminated by CRLF"),
                "chunk not CRLF-terminated must be rejected: " + reply);
    }

    // A non-hex size is a protocol error rather than an empty body.
    {
        const std::string reply = round_trip({}, /*terminate=*/false, kLivePath, "zz\r\n");
        require(contains(reply, "invalid chunk size"),
                "invalid chunk size must be rejected: " + reply);
    }

    // The header names a transfer-coding, so it must be matched as a token rather
    // than searched for as a substring. "notchunked" is not chunked, and a chain
    // like "gzip, chunked" is chunked framing around bytes this server cannot
    // decode — de-framing that and handing the result to the PCM decoder would
    // turn compressed data into confident nonsense. Both must decline the
    // incremental path rather than accept it.
    for (const std::string encoding : {"notchunked", "chunked-garbage", "gzip, chunked"}) {
        const socket_t handle = connect_to_server();
        send_raw(handle, headers_with_encoding(kLivePath, encoding));
        send_raw(handle, "4\r\nabcd\r\n0\r\n\r\n");
        const std::string reply = read_reply(handle);
        close_socket(handle);
        require(contains(reply, "\"stream\":false"),
                "Transfer-Encoding \"" + encoding + "\" must not select the incremental path: " + reply);
    }

    // The same unsupported chain split across two header lines must be rejected the
    // same way. Repeated field lines mean one comma-separated list, so a parser that
    // overwrites instead of combining sees a bare "chunked" and accepts what it just
    // rejected on a single line.
    {
        const socket_t handle = connect_to_server();
        send_raw(handle,
                 std::string("POST ") + kLivePath +
                     " HTTP/1.1\r\nHost: test\r\nTransfer-Encoding: gzip\r\n"
                     "Transfer-Encoding: chunked\r\n\r\n");
        send_raw(handle, "4\r\nabcd\r\n0\r\n\r\n");
        const std::string reply = read_reply(handle);
        close_socket(handle);
        require(contains(reply, "\"stream\":false"),
                "a transfer-coding chain split across header lines must not select the "
                "incremental path: " + reply);
    }

    // A chunk header line past the cap must be rejected, including when its
    // terminator arrives in the same receive as the bytes that exceed it.
    {
        const std::string reply =
            round_trip({}, /*terminate=*/false, kLivePath, std::string(9000, 'a') + ";x\r\n");
        require(contains(reply, "oversized chunk header"),
                "an over-long chunk header line must be rejected: " + reply);
    }

    // Every other endpoint must be untouched: a chunked request elsewhere still
    // goes through the ordinary buffered path, with no body stream published.
    {
        const socket_t handle = connect_to_server();
        send_raw(handle, headers_for("/v1/audio/speech", true));
        send_raw(handle, "4\r\nabcd\r\n0\r\n\r\n");
        const std::string reply = read_reply(handle);
        close_socket(handle);
        require(
            contains(reply, "\"stream\":false"),
            "only the live endpoint may consume an incremental body: " + reply);
        // Pins pre-existing behaviour rather than endorsing it: the buffered path
        // sizes the body from Content-Length, which a chunked request does not send,
        // so it has always yielded an empty body there. Asserted so that a future
        // change to chunked handling elsewhere is a deliberate one, and to show this
        // change did not introduce it.
        require(
            contains(reply, "\"buffered\":0"),
            "chunked bodies on other routes must keep their existing handling: " + reply);
    }

    // Limits are resolved per request, not compiled in. The same 2000-byte chunk is
    // fine under the defaults and rejected once the handler tightens max_chunk_bytes
    // for this request — which is the whole point of the per-model override.
    {
        const std::string chunk = payload(2000, 3);
        const std::string relaxed = round_trip({chunk}, true, kLivePath);
        require(contains(relaxed, "\"bytes\":2000"),
                "a 2000-byte chunk is within the default per-chunk cap: " + relaxed);

        const std::string tightened =
            round_trip({chunk}, true, "/v1/audio/transcriptions/live?model=tight");
        require(contains(tightened, "chunk size exceeds the maximum"),
                "a handler-supplied per-chunk cap must be enforced: " + tightened);
    }

    // Same for the whole-body cap, which is a separate axis: each chunk is legal on
    // its own and only their total crosses the bound.
    {
        const std::vector<std::string> chunks(6, payload(1000, 11));
        const std::string relaxed = round_trip(chunks, true, kLivePath);
        require(contains(relaxed, "\"bytes\":6000"),
                "6000 bytes is within the default body cap: " + relaxed);

        const std::string tightened =
            round_trip(chunks, true, "/v1/audio/transcriptions/live?model=tight");
        require(contains(tightened, "exceeded its maximum size"),
                "a handler-supplied body cap must be enforced: " + tightened);
    }

    // A cap smaller than one hex digit's value must still reject. `cap - value`
    // underflows for any cap below the digit, so at cap=1 a declared `f` used to be
    // accepted outright — a 15-byte chunk against a 1-byte bound.
    {
        const std::string reply =
            round_trip({}, /*terminate=*/false, "/v1/audio/transcriptions/live?model=tiny",
                       "f\r\n123456789012345\r\n");
        require(contains(reply, "chunk size exceeds the maximum"),
                "a chunk cap below the digit value must not underflow: " + reply);
    }

    // The body cap must count bytes that arrived alongside the headers. A body short
    // enough to fit in the first receive never calls receive_more(), so a cap checked
    // only there is skipped by exactly the request most likely to be probing it.
    {
        std::string request = headers_for("/v1/audio/transcriptions/live?model=tight", true);
        for (int i = 0; i < 6; ++i) {
            std::ostringstream frame;
            frame << std::hex << 1000 << "\r\n" << payload(1000, 23) << "\r\n";
            request += frame.str();
        }
        request += "0\r\n\r\n";
        // One write, so headers and the whole 6000-byte body land in a single recv().
        const socket_t handle = connect_to_server();
        send_raw(handle, request);
        const std::string reply = read_reply(handle);
        close_socket(handle);
        require(contains(reply, "exceeded its maximum size"),
                "a fully prefetched body must still be measured against the cap: " + reply);
    }

    // A zero per-chunk cap must not disable the overflow guard. Config rejects a 0,
    // but a Limits built in code could still carry one, and this parse is the last
    // thing between a declared SIZE_MAX and `size + 2` wrapping to 1.
    {
        const std::string reply =
            round_trip({}, /*terminate=*/false, "/v1/audio/transcriptions/live?model=nocap",
                       "ffffffffffffffff\r\n");
        require(contains(reply, "chunk size exceeds the maximum"),
                "a zero max_chunk_bytes must fall back to a real cap, not disable it: " + reply);
    }

    // Asked for only when the request actually opts into an incremental body: every
    // other route must not pay for a lookup it cannot use.
    {
        const int before = handler.limits_queries.load();
        const socket_t handle = connect_to_server();
        send_raw(handle, headers_for("/v1/audio/speech", true));
        send_raw(handle, "4\r\nabcd\r\n0\r\n\r\n");
        (void) read_reply(handle);
        close_socket(handle);
        require_eq(
            handler.limits_queries.load(),
            before,
            "live-ingest limits must not be resolved for a non-live route");
    }

    g_stop.store(true);
    server.join();
    std::cout << "http_live_body_test: all cases passed\n";
    return 0;
}
