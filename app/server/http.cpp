#include "http.h"

#include "engine/framework/io/json.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cerrno>
#include <iostream>
#include <istream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using SocketHandle = SOCKET;
constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
using SocketHandle = int;
constexpr SocketHandle kInvalidSocket = -1;
#endif

namespace minitts::server {
namespace {

std::string lower_ascii(std::string value) {
    for (char & ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

std::string trim(std::string value) {
    auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [&](char ch) { return !is_space(ch); }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [&](char ch) { return !is_space(ch); }).base(), value.end());
    return value;
}

std::string json_quote(std::string_view value) {
    return engine::io::json::stringify_string(value);
}

const char * status_text(int status) noexcept {
    switch (status) {
    case 200:
        return "OK";
    case 204:
        return "No Content";
    case 400:
        return "Bad Request";
    case 403:
        return "Forbidden";
    case 404:
        return "Not Found";
    case 405:
        return "Method Not Allowed";
    case 413:
        return "Payload Too Large";
    case 500:
        return "Internal Server Error";
    case 503:
        return "Service Unavailable";
    default:
        return "Error";
    }
}

class SocketRuntime {
public:
    SocketRuntime() {
#ifdef _WIN32
        WSADATA data;
        const int rc = WSAStartup(MAKEWORD(2, 2), &data);
        if (rc != 0) {
            throw std::runtime_error("WSAStartup failed: " + std::to_string(rc));
        }
#endif
    }

    ~SocketRuntime() {
#ifdef _WIN32
        WSACleanup();
#endif
    }
};

void close_socket(SocketHandle socket) {
    if (socket == kInvalidSocket) {
        return;
    }
#ifdef _WIN32
    closesocket(socket);
#else
    close(socket);
#endif
}

class UniqueSocket {
public:
    UniqueSocket() = default;
    explicit UniqueSocket(SocketHandle socket)
        : socket_(socket) {}
    UniqueSocket(const UniqueSocket &) = delete;
    UniqueSocket & operator=(const UniqueSocket &) = delete;
    UniqueSocket(UniqueSocket && other) noexcept
        : socket_(std::exchange(other.socket_, kInvalidSocket)) {}
    UniqueSocket & operator=(UniqueSocket && other) noexcept {
        if (this != &other) {
            close_socket(socket_);
            socket_ = std::exchange(other.socket_, kInvalidSocket);
        }
        return *this;
    }
    ~UniqueSocket() {
        close_socket(socket_);
    }
    SocketHandle get() const noexcept {
        return socket_;
    }

private:
    SocketHandle socket_ = kInvalidSocket;
};

void send_all(SocketHandle socket, const std::string & data) {
    size_t offset = 0;
    while (offset < data.size()) {
        const auto remaining = data.size() - offset;
#ifdef _WIN32
        const int written = send(
            socket,
            data.data() + offset,
            static_cast<int>(std::min<size_t>(remaining, std::numeric_limits<int>::max())),
            0);
#else
        const ssize_t written = send(socket, data.data() + offset, remaining, 0);
#endif
        if (written <= 0) {
            throw std::runtime_error("socket send failed");
        }
        offset += static_cast<size_t>(written);
    }
}

// An SSE body is written while a model lock is held, so an unbounded blocking
// send() lets a client that uploads but never reads fill the kernel send buffer
// and pin that model indefinitely. A send timeout turns it into a failed write.
void set_send_timeout(SocketHandle socket, int timeout_ms) {
#ifdef _WIN32
    const DWORD timeout = static_cast<DWORD>(timeout_ms);
    setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char *>(&timeout), sizeof(timeout));
#else
    timeval timeout{};
    timeout.tv_sec = timeout_ms / 1000;
    timeout.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
#endif
}

// The endpoints that consume their body incrementally. Gating on the path as
// well as the encoding keeps every other endpoint's request handling bit-for-bit
// unchanged, instead of silently altering how any chunked request is read.
constexpr std::string_view kLiveTranscriptionPath = "/v1/audio/transcriptions/live";
constexpr std::string_view kLiveSpeechPath = "/v1/audio/speech/live";

// True only when the header names exactly one transfer-coding and that coding is
// "chunked". A substring test would accept "notchunked" as well as chains like
// "gzip, chunked" — and for the latter, stripping the chunk framing would hand the
// PCM decoder compressed bytes, which it would happily interpret as audio.
bool is_chunked_only(std::string_view value) {
    bool saw_coding = false;
    size_t start = 0;
    while (start <= value.size()) {
        const size_t comma = value.find(',', start);
        const auto end = comma == std::string_view::npos ? value.size() : comma;
        const std::string coding = trim(std::string(value.substr(start, end - start)));
        if (!coding.empty()) {
            if (saw_coding || lower_ascii(coding) != "chunked") {
                return false;
            }
            saw_coding = true;
        }
        if (comma == std::string_view::npos) {
            break;
        }
        start = comma + 1;
    }
    return saw_coding;
}

bool wants_incremental_body(const HttpRequest & request) {
    if (request.path != kLiveTranscriptionPath && request.path != kLiveSpeechPath) {
        return false;
    }
    const auto it = request.headers.find("transfer-encoding");
    if (it == request.headers.end()) {
        return false;
    }
    return is_chunked_only(it->second);
}

// De-frames an HTTP/1.1 chunked request body off a live socket into a byte
// stream. `underflow()` blocks in recv() waiting for the next chunk, which is
// precisely the contract `AudioChunkReader` documents for a live source — so a
// streaming task can pull capture-time audio straight through it without the
// body ever being fully materialized.
//
// Bounded on four axes, because this stream is read while a model lock is held:
// an idle timeout between reads, an absolute deadline for the whole body, a cap on
// total bytes, and a cap on any single chunk. A client that opens a body and then
// stalls, never stops, or names an enormous chunk therefore cannot pin the model
// indefinitely or exhaust the host. The values come from `LiveIngestLimits`
// (app/server/http.h), which the handler resolves per request.
class ChunkedSocketStreambuf final : public std::streambuf {
public:
    ChunkedSocketStreambuf(SocketHandle socket, std::string prefetched, LiveIngestLimits limits)
        : socket_(socket),
          pending_(std::move(prefetched)),
          limits_(limits),
          started_(std::chrono::steady_clock::now()),
          // Bytes that arrived alongside the headers still count toward the total,
          // or the cap could be overshot by one receive before it is ever consulted.
          received_bytes_(pending_.size()) {}

protected:
    int_type underflow() override {
        if (gptr() < egptr()) {
            return traits_type::to_int_type(*gptr());
        }
        // Covers the bytes that arrived alongside the headers. A body short enough to
        // fit in that first receive never calls receive_more(), so a cap enforced only
        // there would be skipped by the single request most likely to be probing it.
        require_within_body_cap();
        // A chunk is handed out in windows rather than all at once. Reads that stay
        // inside the published get area never re-enter underflow(), so publishing a
        // whole 8 MiB chunk would let the body keep advancing across many model
        // iterations without the deadline ever being consulted again. Windowing puts
        // a bound on how far past the deadline consumption can get.
        if (publish_window()) {
            return traits_type::to_int_type(*gptr());
        }
        if (!next_chunk()) {
            return traits_type::eof();
        }
        return traits_type::to_int_type(*gptr());
    }

private:
    // poll() rather than select(): an accepted descriptor can be >= FD_SETSIZE
    // once enough connections are open, and FD_SET on such a descriptor is
    // undefined behaviour that corrupts the stack.
    bool wait_readable() const {
        int wait_ms = limits_.idle_timeout_ms;
        // Clamp to whatever is left of the total deadline, so a read starting just
        // before it expires cannot overshoot by a further idle period.
        if (limits_.total_timeout_ms > 0) {
            const auto remaining = limits_.total_timeout_ms -
                static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - started_)
                                     .count());
            if (remaining <= 0) {
                return false;
            }
            wait_ms = (wait_ms <= 0) ? remaining : std::min(wait_ms, remaining);
        }
        if (wait_ms <= 0) {
            return true;
        }
        pollfd descriptor{};
        descriptor.fd = socket_;
        descriptor.events = POLLIN;
#ifdef _WIN32
        return WSAPoll(&descriptor, 1, wait_ms) > 0;
#else
        return poll(&descriptor, 1, wait_ms) > 0;
#endif
    }

    void require_within_body_cap() const {
        if (limits_.max_body_bytes > 0 && received_bytes_ > limits_.max_body_bytes) {
            throw std::runtime_error("live request body exceeded its maximum size");
        }
    }

    // Enforced at every point where the body advances, not only before a receive:
    // checking it in receive_more() alone lets an already-buffered tail — the rest
    // of a large chunk, trailers, the terminating chunk — be consumed after the
    // deadline has passed, which is not what "deadline" means to a caller reasoning
    // about how long the model can be held.
    void require_within_deadline() const {
        if (limits_.total_timeout_ms <= 0) {
            return;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - started_)
                                 .count();
        if (elapsed >= limits_.total_timeout_ms) {
            throw std::runtime_error("live request body exceeded its total deadline");
        }
    }

    // Returns false on a clean half-close; throws when a bound is exceeded, so a
    // stalled client surfaces as a stream error rather than a silent truncation
    // that would look like a legitimate end of audio.
    bool receive_more() {
        require_within_deadline();
        if (!wait_readable()) {
            // The wait is clamped to whatever is left of the total deadline, so it can
            // end because that expired rather than because the peer went quiet. Report
            // which one it was instead of always blaming the idle timeout.
            require_within_deadline();
            throw std::runtime_error("live request body stalled: no data within the idle timeout");
        }
        std::array<char, 8192> buffer{};
#ifdef _WIN32
        const int received = recv(socket_, buffer.data(), static_cast<int>(buffer.size()), 0);
#else
        const ssize_t received = recv(socket_, buffer.data(), buffer.size(), 0);
#endif
        // Re-checked after the wait: poll() can return readable just before the
        // deadline, so without this a final receive would be processed past it.
        require_within_deadline();
        if (received <= 0) {
            return false;
        }
        received_bytes_ += static_cast<size_t>(received);
        require_within_body_cap();
        pending_.append(buffer.data(), static_cast<size_t>(received));
        return true;
    }

    bool ensure_available(size_t count) {
        while (pending_.size() - pending_pos_ < count) {
            if (!receive_more()) {
                return false;
            }
        }
        return true;
    }

    static constexpr size_t kMaxLineBytes = 8192;

    bool read_line(std::string & line) {
        for (;;) {
            const auto eol = pending_.find("\r\n", pending_pos_);
            if (eol != std::string::npos) {
                // Checked on the found line too, not only while still searching: a
                // receive can deliver the terminator along with enough bytes to put
                // the line well past the cap, which would otherwise be accepted and
                // make the stated bound roughly double what it claims.
                if (eol - pending_pos_ > kMaxLineBytes) {
                    throw std::runtime_error("chunked request body: oversized chunk header");
                }
                line = pending_.substr(pending_pos_, eol - pending_pos_);
                pending_pos_ = eol + 2;
                return true;
            }
            if (pending_.size() - pending_pos_ > kMaxLineBytes) {
                throw std::runtime_error("chunked request body: oversized chunk header");
            }
            if (!receive_more()) {
                return false;
            }
        }
    }

    bool next_chunk() {
        if (finished_) {
            return false;
        }
        // Bounds consumption of already-buffered framing as well as waiting for more,
        // so the deadline holds even for a body that arrives faster than it is read.
        require_within_deadline();
        std::string header;
        if (!read_line(header)) {
            // The peer closed without the mandatory terminating 0-chunk. Treating
            // that as a clean end would hand the caller a truncated body that looks
            // byte-for-byte like a complete one — for audio, a cut-off sentence
            // indistinguishable from the speaker stopping. Fail instead.
            throw std::runtime_error(
                "chunked request body: connection closed before the terminating chunk");
        }
        if (const auto extension = header.find(';'); extension != std::string::npos) {
            header.resize(extension);  // chunk extensions are unused here
        }
        header = trim(header);
        // Parsed by hand rather than with stoull: the size is attacker-controlled, and
        // "ffffffffffffffff" would otherwise yield SIZE_MAX, whose `size + 2` wraps to
        // 1 and slips past the availability check into invalid iterator arithmetic.
        if (header.empty()) {
            throw std::runtime_error("chunked request body: empty chunk size");
        }
        // Unlike the other bounds, a per-chunk cap cannot be disabled: the chunk is
        // materialized in memory, so "unbounded" is not implementable. 0 falls back
        // to the default rather than meaning "no limit".
        const size_t chunk_cap =
            limits_.max_chunk_bytes > 0 ? limits_.max_chunk_bytes : LiveIngestLimits{}.max_chunk_bytes;
        size_t size = 0;
        for (const char digit : header) {
            int value = 0;
            if (digit >= '0' && digit <= '9') {
                value = digit - '0';
            } else if (digit >= 'a' && digit <= 'f') {
                value = digit - 'a' + 10;
            } else if (digit >= 'A' && digit <= 'F') {
                value = digit - 'A' + 10;
            } else {
                throw std::runtime_error(
                    "chunked request body: invalid chunk size \"" + header + "\"");
            }
            // Checked in two steps, each on operands already known to be in range.
            // The single-expression form `size > (cap - value) / 16` underflows
            // whenever cap < value — at cap = 1 a digit of 'f' wraps to a huge
            // quotient and is accepted — which is the same class of bug this
            // hand-rolled parse exists to prevent.
            if (size > chunk_cap / 16) {
                throw std::runtime_error("chunked request body: chunk size exceeds the maximum");
            }
            size *= 16;
            // size <= (cap / 16) * 16 <= cap here, so the subtraction cannot wrap.
            if (static_cast<size_t>(value) > chunk_cap - size) {
                throw std::runtime_error("chunked request body: chunk size exceeds the maximum");
            }
            size += static_cast<size_t>(value);
        }
        if (size == 0) {
            finished_ = true;
            // Trailers are bounded: they arrive after the terminating chunk, so
            // without a cap a peer could stream them indefinitely into pending_,
            // which is never compacted on this path.
            size_t trailer_bytes = 0;
            std::string trailer;
            for (;;) {
                if (!read_line(trailer)) {
                    throw std::runtime_error(
                        "chunked request body: connection closed inside the trailer");
                }
                if (trailer.empty()) {
                    break;
                }
                trailer_bytes += trailer.size();
                if (trailer_bytes > 8192) {
                    throw std::runtime_error("chunked request body: trailer section too large");
                }
            }
            return false;
        }
        if (!ensure_available(size + 2)) {
            throw std::runtime_error("chunked request body: connection closed mid-chunk");
        }
        // The two bytes after chunk data must be CRLF; accepting anything else lets a
        // desynchronised sender's payload be silently reinterpreted as framing.
        if (pending_.compare(pending_pos_ + size, 2, "\r\n") != 0) {
            throw std::runtime_error("chunked request body: chunk data not terminated by CRLF");
        }
        // Dropped before the assign, which may reallocate and free what the get area
        // still points at. publish_window() can throw on the deadline before it calls
        // setg(), and leaving stale pointers across that throw would hand anyone who
        // caught the exception and read again a dangling comparison.
        setg(nullptr, nullptr, nullptr);
        chunk_.assign(
            pending_.begin() + static_cast<std::ptrdiff_t>(pending_pos_),
            pending_.begin() + static_cast<std::ptrdiff_t>(pending_pos_ + size));
        pending_pos_ += size + 2;  // also skip the chunk's trailing CRLF
        if (pending_pos_ > 64 * 1024) {
            pending_.erase(0, pending_pos_);
            pending_pos_ = 0;
        }
        chunk_pos_ = 0;
        publish_window();
        return true;
    }

    // Exposes the next slice of the current chunk, re-checking the deadline each
    // time. Returns false once the chunk is spent, which is the caller's signal to
    // read the next one off the socket.
    bool publish_window() {
        if (chunk_pos_ >= chunk_.size()) {
            return false;
        }
        require_within_deadline();
        const size_t window = std::min(kWindowBytes, chunk_.size() - chunk_pos_);
        setg(chunk_.data() + chunk_pos_,
             chunk_.data() + chunk_pos_,
             chunk_.data() + chunk_pos_ + window);
        chunk_pos_ += window;
        return true;
    }

    static constexpr size_t kWindowBytes = 64 * 1024;

    SocketHandle socket_;
    std::string pending_;   // received but not yet de-framed
    size_t pending_pos_ = 0;
    std::vector<char> chunk_;  // the de-framed chunk currently being served
    size_t chunk_pos_ = 0;     // how much of chunk_ has been published so far
    LiveIngestLimits limits_;
    std::chrono::steady_clock::time_point started_;
    size_t received_bytes_;
    bool finished_ = false;
};

// `leftover` receives any body bytes that arrived alongside the headers, but only
// for a chunked body — that case returns with the socket deliberately undrained
// so the handler can consume the rest as it is sent.
HttpRequest read_http_request(
    SocketHandle socket,
    uint64_t max_request_body_bytes,
    std::string & leftover) {
    std::string data;
    std::array<char, 8192> buffer{};
    size_t header_end = std::string::npos;
    while (header_end == std::string::npos) {
#ifdef _WIN32
        const int received = recv(socket, buffer.data(), static_cast<int>(buffer.size()), 0);
#else
        const ssize_t received = recv(socket, buffer.data(), buffer.size(), 0);
#endif
        if (received <= 0) {
            throw std::runtime_error("socket receive failed before HTTP headers");
        }
        data.append(buffer.data(), static_cast<size_t>(received));
        header_end = data.find("\r\n\r\n");
        if (data.size() > 1024 * 1024 && header_end == std::string::npos) {
            throw std::runtime_error("HTTP headers exceed 1 MiB");
        }
    }

    std::istringstream header_stream(data.substr(0, header_end));
    HttpRequest request;
    std::string line;
    if (!std::getline(header_stream, line)) {
        throw std::runtime_error("empty HTTP request");
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    std::istringstream request_line(line);
    request_line >> request.method >> request.path;
    if (request.method.empty() || request.path.empty()) {
        throw std::runtime_error("invalid HTTP request line");
    }
    const auto query = request.path.find('?');
    if (query != std::string::npos) {
        request.query = request.path.substr(query + 1);
        request.path = request.path.substr(0, query);
    }

    while (std::getline(header_stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const auto pos = line.find(':');
        if (pos == std::string::npos) {
            continue;
        }
        const std::string name = lower_ascii(trim(line.substr(0, pos)));
        const std::string value = trim(line.substr(pos + 1));
        // Repeated field lines are equivalent to one comma-separated list (RFC 9110
        // 5.2), and for Transfer-Encoding that equivalence is load-bearing: splitting
        // "gzip, chunked" across two lines would otherwise overwrite the first with
        // the second and leave a bare "chunked" that passes the coding check. Only
        // this header is combined, so no other endpoint's parsing changes.
        if (name == "transfer-encoding") {
            auto & slot = request.headers[name];
            // Appended in place rather than rebuilt: `slot = slot + ", " + value`
            // recopies everything accumulated so far on each line, so a header block
            // packed with repeated fields costs quadratic time — and this runs before
            // dispatch, on every endpoint, on attacker-supplied input.
            if (!slot.empty()) {
                slot.append(", ");
            }
            slot.append(value);
            continue;
        }
        request.headers[name] = value;
    }

    if (wants_incremental_body(request)) {
        leftover = data.substr(header_end + 4);
        return request;
    }

    size_t content_length = 0;
    if (const auto it = request.headers.find("content-length"); it != request.headers.end()) {
        // std::stoull throws on a non-numeric or overflowing header, and the
        // body loop below grows request.body until it reaches whatever this
        // says. An absurd Content-Length is therefore an unbounded in-memory
        // accumulation driven by a single request.
        //
        const std::string & raw = it->second;
        unsigned long long parsed = 0;
        const auto * first = raw.data();
        const auto * last = raw.data() + raw.size();
        const auto [ptr, ec] = std::from_chars(first, last, parsed);
        if (ec != std::errc{} || ptr != last) {
            throw std::runtime_error("invalid Content-Length header");
        }
        if (parsed > max_request_body_bytes) {
            throw std::runtime_error(
                "request body exceeds the maximum of " + std::to_string(max_request_body_bytes) + " bytes");
        }
        if (parsed > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("request body exceeds the maximum addressable size");
        }
        content_length = static_cast<size_t>(parsed);
    }
    request.body = data.substr(header_end + 4);
    while (request.body.size() < content_length) {
#ifdef _WIN32
        const int received = recv(socket, buffer.data(), static_cast<int>(buffer.size()), 0);
#else
        const ssize_t received = recv(socket, buffer.data(), buffer.size(), 0);
#endif
        if (received <= 0) {
            throw std::runtime_error("socket receive failed while reading HTTP body");
        }
        request.body.append(buffer.data(), static_cast<size_t>(received));
    }
    if (request.body.size() > content_length) {
        request.body.resize(content_length);
    }
    return request;
}

std::string serialize_response(const HttpResponse & response) {
    std::ostringstream out;
    out << "HTTP/1.1 " << response.status << " " << status_text(response.status) << "\r\n"
        << "Content-Type: " << response.content_type << "\r\n"
        << "Content-Length: " << response.body.size() << "\r\n"
        << "Connection: close\r\n";
    for (const auto & [key, value] : response.headers) {
        out << key << ": " << value << "\r\n";
    }
    out << "\r\n";
    std::string header = out.str();
    header += response.body;
    return header;
}

std::string serialize_stream_headers(const HttpResponse & response) {
    std::ostringstream out;
    out << "HTTP/1.1 " << response.status << " " << status_text(response.status) << "\r\n"
        << "Content-Type: " << response.content_type << "\r\n"
        << "Transfer-Encoding: chunked\r\n"
        << "Cache-Control: no-cache\r\n"
        << "Connection: close\r\n";
    for (const auto & [key, value] : response.headers) {
        out << key << ": " << value << "\r\n";
    }
    out << "\r\n";
    return out.str();
}

class ChunkedHttpStreamWriter final : public HttpStreamWriter {
public:
    explicit ChunkedHttpStreamWriter(SocketHandle socket)
        : socket_(socket) {}

    void write(std::string_view data) override {
        if (data.empty()) {
            return;
        }
        std::ostringstream header;
        header << std::hex << data.size() << "\r\n";
        send_all(socket_, header.str());
        send_all(socket_, std::string(data));
        send_all(socket_, "\r\n");
    }

    void finish() {
        send_all(socket_, "0\r\n\r\n");
    }

private:
    SocketHandle socket_;
};

UniqueSocket bind_listen_socket(const std::string & host, int port) {
    UniqueSocket socket_handle(socket(AF_INET, SOCK_STREAM, 0));
    if (socket_handle.get() == kInvalidSocket) {
        throw std::runtime_error("could not create listen socket");
    }
    int yes = 1;
#ifdef _WIN32
    setsockopt(socket_handle.get(), SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char *>(&yes), sizeof(yes));
#else
    setsockopt(socket_handle.get(), SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#endif
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        throw std::runtime_error("server host must be an IPv4 address: " + host);
    }
    if (bind(socket_handle.get(), reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        throw std::runtime_error("could not bind " + host + ":" + std::to_string(port));
    }
    if (listen(socket_handle.get(), 16) != 0) {
        throw std::runtime_error("could not listen on " + host + ":" + std::to_string(port));
    }
    return socket_handle;
}

void handle_client(SocketHandle client, IHttpHandler & handler, uint64_t max_request_body_bytes) {
    UniqueSocket socket(client);
    try {
        std::string leftover;
        auto request = read_http_request(socket.get(), max_request_body_bytes, leftover);
        const bool incremental_body = wants_incremental_body(request);
        // Asked for once, before the stream exists, because the streambuf takes its
        // bounds at construction. The handler resolves them from server config and
        // any per-model override — the transport has no notion of models.
        const LiveIngestLimits limits =
            incremental_body ? handler.live_ingest_limits(request) : LiveIngestLimits{};
        if (incremental_body && limits.send_timeout_ms > 0) {
            // Scoped to this endpoint: it is the only one whose response is written
            // while a model lock is held, so it is the only one where a client that
            // stops reading can pin a model. Applying it server-wide would risk
            // truncating a large ordinary response to a merely slow client.
            set_send_timeout(socket.get(), limits.send_timeout_ms);
        }
        // Constructed unconditionally so it outlives the handler call, but only
        // published on `request` when the client actually declared a chunked body.
        ChunkedSocketStreambuf body_buffer(socket.get(), std::move(leftover), limits);
        std::istream body_stream(&body_buffer);
        // Without this, a throw from underflow() is caught by istream and turned into
        // badbit. A reader that checks gcount() then sees a short read and reports a
        // clean end of input, so a stall, an oversize chunk or a mid-body disconnect
        // would all be delivered as a successful, silently truncated body.
        body_stream.exceptions(std::ios::badbit);
        if (incremental_body) {
            request.body_stream = &body_stream;
        }
        const auto response = handler.handle(request);
        if (response.stream_body) {
            send_all(socket.get(), serialize_stream_headers(response));
            ChunkedHttpStreamWriter writer(socket.get());
            try {
                response.stream_body(writer);
            } catch (const std::exception & ex) {
                if (response.content_type.rfind("text/event-stream", 0) == 0) {
                    const std::string data =
                        "data: {\"type\":\"error\",\"error\":{\"message\":" +
                        json_quote(ex.what()) +
                        "}}\n\n";
                    writer.write(data);
                } else {
                    std::cerr << "audiocpp_server streaming response failed: " << ex.what() << "\n";
                }
            }
            writer.finish();
        } else {
            send_all(socket.get(), serialize_response(response));
        }
    } catch (const std::exception & ex) {
        try {
            send_all(socket.get(), serialize_response(error_response(500, ex.what(), "server_error")));
        } catch (const std::exception & send_error) {
            std::cerr << "audiocpp_server failed to send error response: " << send_error.what() << "\n";
        }
    }
}

bool wait_for_client(SocketHandle socket, int timeout_ms) {
    fd_set read_set;
    FD_ZERO(&read_set);
    FD_SET(socket, &read_set);

    timeval timeout{};
    timeout.tv_sec = timeout_ms / 1000;
    timeout.tv_usec = (timeout_ms % 1000) * 1000;

#ifdef _WIN32
    const int ready = select(0, &read_set, nullptr, nullptr, &timeout);
#else
    const int ready = select(socket + 1, &read_set, nullptr, nullptr, &timeout);
#endif
    if (ready < 0) {
#ifdef _WIN32
        if (WSAGetLastError() == WSAEINTR) {
#else
        if (errno == EINTR) {
#endif
            return false;
        }
        throw std::runtime_error("server select failed");
    }
    return ready > 0 && FD_ISSET(socket, &read_set);
}

}  // namespace

HttpResponse json_response(std::string body, int status) {
    return HttpResponse{status, "application/json", std::move(body), {}, {}};
}

HttpResponse error_response(int status, const std::string & message, const std::string & type) {
    const std::string body = std::string("{\"error\":{\"message\":") + json_quote(message) +
        ",\"type\":" + json_quote(type) + "}}";
    return json_response(body, status);
}

void serve_http(const std::string & host, int port, IHttpHandler & handler, ShutdownRequested shutdown_requested, uint64_t max_request_body_bytes) {
    SocketRuntime sockets;
    auto listen_socket = bind_listen_socket(host, port);
    std::cout << "audiocpp_server listening on http://" << host << ":" << port << "\n";
    while (!shutdown_requested()) {
        if (!wait_for_client(listen_socket.get(), 250)) {
            continue;
        }
        sockaddr_in client_addr{};
#ifdef _WIN32
        int client_len = sizeof(client_addr);
#else
        socklen_t client_len = sizeof(client_addr);
#endif
        const SocketHandle client = accept(
            listen_socket.get(),
            reinterpret_cast<sockaddr *>(&client_addr),
            &client_len);
        if (client == kInvalidSocket) {
#ifdef _WIN32
            const int error = WSAGetLastError();
            const bool transient = error == WSAEINTR || error == WSAEWOULDBLOCK;
#else
            const bool transient = errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK;
#endif
            if (shutdown_requested() || transient) {
                continue;
            }
            throw std::runtime_error("accept failed");
        }
        std::thread(handle_client, client, std::ref(handler), max_request_body_bytes).detach();
    }
    std::cout << "audiocpp_server stopped\n";
}

}  // namespace minitts::server
