#include "multipart.h"

// This is a transport/parser integration test: it sends a real Content-Length
// multipart request through serve_http into a minimal IHttpHandler fixture. It
// intentionally does not construct ServerState, load an ASR model, or claim
// full inference-route verification.

#include "http.h"

#include <atomic>
#include <chrono>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

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

#ifdef _WIN32
using socket_t = SOCKET;
constexpr socket_t kInvalidSocket = INVALID_SOCKET;
#else
using socket_t = int;
constexpr socket_t kInvalidSocket = -1;
#endif

constexpr int kMultipartHttpTestPort = 18095;
constexpr int kSocketTimeoutMs = 5000;
std::atomic<bool> g_stop_http{false};

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool stop_http_requested() {
    return g_stop_http.load();
}

void close_socket(socket_t handle) {
    if (handle == kInvalidSocket) {
        return;
    }
#ifdef _WIN32
    closesocket(handle);
#else
    close(handle);
#endif
}

class SocketRuntime {
public:
    SocketRuntime() {
#ifdef _WIN32
        WSADATA data;
        const int result = WSAStartup(MAKEWORD(2, 2), &data);
        require(result == 0, "multipart HTTP test WSAStartup failed");
#endif
    }

    ~SocketRuntime() {
#ifdef _WIN32
        WSACleanup();
#endif
    }

    SocketRuntime(const SocketRuntime &) = delete;
    SocketRuntime & operator=(const SocketRuntime &) = delete;
};

class UniqueSocket {
public:
    UniqueSocket() = default;
    explicit UniqueSocket(socket_t handle)
        : handle_(handle) {}

    ~UniqueSocket() {
        close_socket(handle_);
    }

    UniqueSocket(const UniqueSocket &) = delete;
    UniqueSocket & operator=(const UniqueSocket &) = delete;

    UniqueSocket(UniqueSocket && other) noexcept
        : handle_(std::exchange(other.handle_, kInvalidSocket)) {}

    UniqueSocket & operator=(UniqueSocket && other) noexcept {
        if (this != &other) {
            close_socket(handle_);
            handle_ = std::exchange(other.handle_, kInvalidSocket);
        }
        return *this;
    }

    socket_t get() const noexcept {
        return handle_;
    }

private:
    socket_t handle_ = kInvalidSocket;
};

void set_socket_timeouts(socket_t handle) {
#ifdef _WIN32
    const DWORD timeout = kSocketTimeoutMs;
    const auto * timeout_value = reinterpret_cast<const char *>(&timeout);
    require(
        setsockopt(handle, SOL_SOCKET, SO_SNDTIMEO, timeout_value, sizeof(timeout)) == 0,
        "multipart HTTP client send timeout setup failed");
    require(
        setsockopt(handle, SOL_SOCKET, SO_RCVTIMEO, timeout_value, sizeof(timeout)) == 0,
        "multipart HTTP client receive timeout setup failed");
#else
    timeval timeout{};
    timeout.tv_sec = kSocketTimeoutMs / 1000;
    timeout.tv_usec = (kSocketTimeoutMs % 1000) * 1000;
    require(
        setsockopt(handle, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0,
        "multipart HTTP client send timeout setup failed");
    require(
        setsockopt(handle, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0,
        "multipart HTTP client receive timeout setup failed");
#endif
}

void send_all(socket_t handle, const std::string & data) {
    size_t offset = 0;
    while (offset < data.size()) {
#ifdef _WIN32
        const int written = send(handle, data.data() + offset, static_cast<int>(data.size() - offset), 0);
#else
        const ssize_t written = send(handle, data.data() + offset, data.size() - offset, 0);
#endif
        require(written > 0, "multipart HTTP client send failed");
        offset += static_cast<size_t>(written);
    }
}

class HttpServerSession {
public:
    explicit HttpServerSession(minitts::server::IHttpHandler & handler) {
        g_stop_http.store(false);
        server_ = std::thread([this, &handler] {
            try {
                minitts::server::serve_http(
                    "127.0.0.1",
                    kMultipartHttpTestPort,
                    handler,
                    stop_http_requested,
                    2 * 1024 * 1024);
            } catch (...) {
                server_error_ = std::current_exception();
                server_failed_.store(true);
            }
        });
    }

    ~HttpServerSession() {
        stop_and_join();
    }

    HttpServerSession(const HttpServerSession &) = delete;
    HttpServerSession & operator=(const HttpServerSession &) = delete;

    void rethrow_if_failed() const {
        if (server_failed_.load()) {
            std::rethrow_exception(server_error_);
        }
    }

    void finish() {
        stop_and_join();
        rethrow_if_failed();
    }

private:
    void stop_and_join() {
        g_stop_http.store(true);
        if (server_.joinable()) {
            server_.join();
        }
    }

    std::exception_ptr server_error_;
    std::atomic<bool> server_failed_{false};
    std::thread server_;
};

UniqueSocket connect_to_multipart_server(const HttpServerSession & server) {
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(kMultipartHttpTestPort));
    require(inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) == 1, "loopback address setup failed");
    for (int attempt = 0; attempt < 200; ++attempt) {
        server.rethrow_if_failed();
        UniqueSocket handle(socket(AF_INET, SOCK_STREAM, 0));
        require(handle.get() != kInvalidSocket, "multipart HTTP client socket failed");
        if (connect(handle.get(), reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0) {
            set_socket_timeouts(handle.get());
            return handle;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    server.rethrow_if_failed();
    require(false, "could not connect to multipart HTTP test server");
    return {};
}

std::string read_http_reply(socket_t handle) {
    std::string reply;
    char buffer[4096];
    for (;;) {
#ifdef _WIN32
        const int received = recv(handle, buffer, sizeof(buffer), 0);
#else
        const ssize_t received = recv(handle, buffer, sizeof(buffer), 0);
#endif
        if (received == 0) {
            break;
        }
        require(received > 0, "multipart HTTP client receive failed or timed out");
        reply.append(buffer, static_cast<size_t>(received));
    }
    return reply;
}

std::string make_multipart_body(
    const std::string & boundary,
    const std::string & filename,
    const std::string & payload) {
    const std::string marker = "--" + boundary;
    std::string body = "preamble " + marker + " bytes\r\n";
    body += "still preamble\r\n";
    body += marker + "-not-a-delimiter\r\n";
    body += marker + "\r\n";
    body += "Content-Disposition: form-data; name=\"file\"; filename=\"";
    body += filename + "\"\r\n";
    body += "Content-Type: application/octet-stream\r\n\r\n";
    body += payload;
    body += "\r\n" + marker + "\r\n";
    body += "Content-Disposition: form-data; name=\"later\"\n\n";
    body += "kept-after-inline-markers\n";
    body += marker + "--\n";
    return body;
}

class MultipartHttpHandler final : public minitts::server::IHttpHandler {
public:
    MultipartHttpHandler(std::string expected_payload, std::string expected_filename)
        : expected_payload_(std::move(expected_payload)),
          expected_filename_(std::move(expected_filename)) {}

    minitts::server::HttpResponse handle(const minitts::server::HttpRequest & request) override {
        const auto content_type = request.headers.find("content-type");
        require(content_type != request.headers.end(), "loopback request lost content type");
        const auto boundary = minitts::server::extract_multipart_boundary(content_type->second);
        require(boundary.has_value(), "loopback request boundary was not extracted");
        const auto parts = minitts::server::parse_multipart_body(request.body, *boundary);
        require(parts.size() == 2, "loopback parser should return both multipart parts");
        require(parts[0].name == "file", "loopback file part name");
        require(parts[0].filename == expected_filename_, "loopback file part filename");
        require(parts[0].data == expected_payload_, "loopback file part payload");
        require(parts[1].name == "later", "loopback following part name");
        require(parts[1].filename.empty(), "loopback following part should not have a filename");
        require(parts[1].data == "kept-after-inline-markers", "loopback following part data");
        return minitts::server::json_response("{\"parts\":2,\"exact_match\":true}");
    }

private:
    const std::string expected_payload_;
    const std::string expected_filename_;
};

void test_loopback_http_multipart_transport_and_parser() {
    const std::string boundary = "LOOPBACK_BINARY_BOUNDARY";
    const std::string marker = "--" + boundary;
    std::string payload;
    payload.push_back('\x00');
    payload.push_back('\x80');
    payload.push_back('\xff');
    payload += "prefix" + marker + "-owned\rbare-cr\nowned-newline\n";
    payload += "x" + marker + "--inline-closing";
    for (int i = 0; i < 4096; ++i) {
        payload += "inline-" + marker + (i % 2 == 0 ? "--" : "-owned") + "-marker";
    }
    payload.push_back('\x01');
    const std::string filename = "inline-" + marker + "--name.bin";
    const std::string body = make_multipart_body(boundary, filename, payload);

    MultipartHttpHandler handler(payload, filename);
    HttpServerSession server(handler);

    UniqueSocket client = connect_to_multipart_server(server);
    std::ostringstream request;
    request << "POST /v1/audio/transcriptions HTTP/1.1\r\n"
            << "Host: 127.0.0.1\r\n"
            << "Content-Type: multipart/form-data; boundary=\"" << boundary << "\"\r\n"
            << "Content-Length: " << body.size() << "\r\n\r\n";
    send_all(client.get(), request.str() + body);
    const std::string reply = read_http_reply(client.get());

    require(reply.find("HTTP/1.1 200 OK") != std::string::npos, "loopback multipart status");
    require(reply.find("\"parts\":2") != std::string::npos, "loopback multipart part count");
    require(
        reply.find("\"exact_match\":true") != std::string::npos,
        "loopback multipart exact payload and filename match");

    server.finish();
}

}  // namespace

int main() {
    try {
        SocketRuntime sockets;
        test_loopback_http_multipart_transport_and_parser();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "server_multipart_http_test passed\n";
    return 0;
}
