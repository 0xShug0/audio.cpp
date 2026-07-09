#include "http.h"

#include "engine/framework/io/json.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cerrno>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
using SocketHandle = SOCKET;
constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
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
    case 400:
        return "Bad Request";
    case 404:
        return "Not Found";
    case 405:
        return "Method Not Allowed";
    case 500:
        return "Internal Server Error";
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

HttpRequest read_http_request(SocketHandle socket) {
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
        request.headers[lower_ascii(trim(line.substr(0, pos)))] = trim(line.substr(pos + 1));
    }

    size_t content_length = 0;
    if (const auto it = request.headers.find("content-length"); it != request.headers.end()) {
        content_length = static_cast<size_t>(std::stoull(it->second));
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

void handle_client(SocketHandle client, IHttpHandler & handler) {
    UniqueSocket socket(client);
    try {
        const auto request = read_http_request(socket.get());
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
    return HttpResponse{status, "application/json", std::move(body), {}};
}

HttpResponse error_response(int status, const std::string & message, const std::string & type) {
    const std::string body = std::string("{\"error\":{\"message\":") + json_quote(message) +
        ",\"type\":" + json_quote(type) + "}}";
    return json_response(body, status);
}

void serve_http(const std::string & host, int port, IHttpHandler & handler, ShutdownRequested shutdown_requested) {
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
        std::thread(handle_client, client, std::ref(handler)).detach();
    }
    std::cout << "audiocpp_server stopped\n";
}

}  // namespace minitts::server
