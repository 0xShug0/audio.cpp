// Exercise the real POSIX HTTP listener with ordinary and high descriptors.
// No model weights or inference backend are involved in this transport test.
#include "../../app/server/http.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/resource.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr int kPort = 18095;
std::atomic<bool> stop{false};
bool stop_requested() { return stop.load(); }

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

class DescriptorPressure {
public:
    ~DescriptorPressure() {
        for (int fd : descriptors_) close(fd);
    }
    bool reserve() {
        constexpr int target = FD_SETSIZE + 16;
        rlimit limits{};
        if (getrlimit(RLIMIT_NOFILE, &limits) != 0) return false;
        if (limits.rlim_cur < static_cast<rlim_t>(target + 32)) {
            if (limits.rlim_max < static_cast<rlim_t>(target + 32)) return false;
            limits.rlim_cur = target + 32;
            if (setrlimit(RLIMIT_NOFILE, &limits) != 0) return false;
        }
        for (;;) {
            const int fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
            if (fd < 0) return false;
            descriptors_.push_back(fd);
            if (fd >= target) {
                std::cout << "Listener will use a descriptor above " << fd << std::endl;
                return true;
            }
        }
    }
private:
    std::vector<int> descriptors_;
};

class Handler final : public minitts::server::IHttpHandler {
public:
    minitts::server::HttpResponse handle(const minitts::server::HttpRequest &) override {
        return minitts::server::json_response("{\"ok\":true}");
    }
};

class Socket {
public:
    explicit Socket(int fd) : fd_(fd) {}
    Socket(const Socket &) = delete;
    Socket & operator=(const Socket &) = delete;
    ~Socket() { if (fd_ >= 0) close(fd_); }
    int get() const { return fd_; }
private:
    int fd_;
};

int connect_to_server() {
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(kPort);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    for (int attempt = 0; attempt < 100; ++attempt) {
        const int fd = socket(AF_INET, SOCK_STREAM, 0);
        require(fd >= 0, "could not create test socket");
        if (connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0) return fd;
        close(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    throw std::runtime_error("listener did not become available");
}

void exchange() {
    Socket client(connect_to_server());
    timeval timeout{3, 0};
    require(setsockopt(client.get(), SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0,
            "could not set receive timeout");
    require(setsockopt(client.get(), SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0,
            "could not set send timeout");
    const std::string request = "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    size_t sent = 0;
    while (sent < request.size()) {
        const ssize_t n = send(client.get(), request.data() + sent, request.size() - sent, 0);
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, "could not send request");
        sent += static_cast<size_t>(n);
    }
    std::string response;
    char buffer[1024];
    for (;;) {
        const ssize_t n = recv(client.get(), buffer, sizeof(buffer), 0);
        if (n < 0 && errno == EINTR) continue;
        require(n >= 0, "response timed out or failed");
        if (n == 0) break;
        response.append(buffer, static_cast<size_t>(n));
        require(response.size() < 8192, "response exceeded test bound");
    }
    require(response.find("HTTP/1.1 200 OK") == 0, "request was not accepted");
    require(response.find("{\"ok\":true}") != std::string::npos, "response body was lost");
}
} // namespace

int main(int argc, char ** argv) {
    if (argc > 2 || (argc == 2 && std::string(argv[1]) != "--high-fd")) return 2;
    DescriptorPressure pressure;
    if (argc == 2 && !pressure.reserve()) {
        std::cout << "SKIP: cannot reserve descriptors above FD_SETSIZE\n";
        return 77;
    }
    Handler handler;
    std::exception_ptr server_error;
    std::thread server([&] {
        try {
            minitts::server::serve_http("127.0.0.1", kPort, handler, stop_requested, 4096);
        } catch (...) {
            server_error = std::current_exception();
        }
    });
    std::exception_ptr client_error;
    try {
        for (int i = 0; i < 10; ++i) exchange();
    } catch (...) {
        client_error = std::current_exception();
    }
    const auto shutdown_start = std::chrono::steady_clock::now();
    stop.store(true);
    server.join(); // server_error is read only after the writer is joined.
    try {
        if (server_error) std::rethrow_exception(server_error);
        if (client_error) std::rethrow_exception(client_error);
        require(std::chrono::steady_clock::now() - shutdown_start < std::chrono::seconds(3),
                "idle listener did not stop promptly");
        std::cout << "PASS: ten HTTP exchanges and idle shutdown\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
