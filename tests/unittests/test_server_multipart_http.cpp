#include "multipart.h"

// This is a transport/parser integration test: it sends a real Content-Length
// multipart request through serve_http into a minimal IHttpHandler fixture. It
// intentionally does not construct ServerState, load an ASR model, or claim
// full inference-route verification.

#include "http.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

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
#ifdef _WIN32
    closesocket(handle);
#else
    close(handle);
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

socket_t connect_to_multipart_server() {
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(kMultipartHttpTestPort));
    require(inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) == 1, "loopback address setup failed");
    for (int attempt = 0; attempt < 200; ++attempt) {
        const socket_t handle = socket(AF_INET, SOCK_STREAM, 0);
        require(handle != kInvalidSocket, "multipart HTTP client socket failed");
        if (connect(handle, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0) {
            return handle;
        }
        close_socket(handle);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    require(false, "could not connect to multipart HTTP test server");
    return kInvalidSocket;
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
        if (received <= 0) {
            break;
        }
        reply.append(buffer, static_cast<size_t>(received));
    }
    return reply;
}

std::string make_multipart_body(const std::string & boundary, const std::string & payload) {
    const std::string marker = "--" + boundary;
    std::string body = "preamble " + marker + " bytes\r\n";
    body += "still preamble\r\n";
    body += marker + "-not-a-delimiter\r\n";
    body += marker + "\r\n";
    body += "Content-Disposition: form-data; name=\"file\"; filename=\"inline-";
    body += marker + "--name.bin\"\r\n";
    body += "Content-Type: application/octet-stream\r\n\r\n";
    body += payload;
    body += "\r\n" + marker + "\r\n";
    body += "Content-Disposition: form-data; name=\"later\"\n\n";
    body += "kept-after-inline-markers\n";
    body += marker + "--\n";
    return body;
}

size_t byte_sum(const std::string & value) {
    size_t sum = 0;
    for (const unsigned char byte : value) {
        sum += byte;
    }
    return sum;
}

class MultipartHttpHandler final : public minitts::server::IHttpHandler {
public:
    minitts::server::HttpResponse handle(const minitts::server::HttpRequest & request) override {
        const auto content_type = request.headers.find("content-type");
        require(content_type != request.headers.end(), "loopback request lost content type");
        const auto boundary = minitts::server::extract_multipart_boundary(content_type->second);
        require(boundary.has_value(), "loopback request boundary was not extracted");
        const auto parts = minitts::server::parse_multipart_body(request.body, *boundary);
        require(parts.size() == 2, "loopback parser should return both multipart parts");
        require(parts[0].name == "file", "loopback file part name");
        require(parts[1].name == "later", "loopback following part name");
        require(parts[1].data == "kept-after-inline-markers", "loopback following part data");
        return minitts::server::json_response(
            "{\"parts\":2,\"file_bytes\":" + std::to_string(parts[0].data.size()) +
            ",\"file_sum\":" + std::to_string(byte_sum(parts[0].data)) + "}");
    }
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
    const std::string body = make_multipart_body(boundary, payload);

    g_stop_http.store(false);
    MultipartHttpHandler handler;
    std::thread server([&] {
        minitts::server::serve_http(
            "127.0.0.1", kMultipartHttpTestPort, handler, stop_http_requested, 2 * 1024 * 1024);
    });

    const socket_t client = connect_to_multipart_server();
    std::ostringstream request;
    request << "POST /v1/audio/transcriptions HTTP/1.1\r\n"
            << "Host: 127.0.0.1\r\n"
            << "Content-Type: multipart/form-data; boundary=\"" << boundary << "\"\r\n"
            << "Content-Length: " << body.size() << "\r\n\r\n";
    send_all(client, request.str() + body);
    const std::string reply = read_http_reply(client);
    close_socket(client);

    require(reply.find("HTTP/1.1 200 OK") != std::string::npos, "loopback multipart status");
    require(reply.find("\"parts\":2") != std::string::npos, "loopback multipart part count");
    require(
        reply.find("\"file_bytes\":" + std::to_string(payload.size())) != std::string::npos,
        "loopback multipart binary length");
    require(
        reply.find("\"file_sum\":" + std::to_string(byte_sum(payload))) != std::string::npos,
        "loopback multipart binary checksum");

    g_stop_http.store(true);
    server.join();
}

}  // namespace

int main() {
    try {
        test_loopback_http_multipart_transport_and_parser();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "server_multipart_http_test passed\n";
    return 0;
}
